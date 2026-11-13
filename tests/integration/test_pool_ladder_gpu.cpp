// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// The CUDA pool correctness ladder: the same pool, the same graph, one rising
// concurrency axis per rung. Every rung reports the full counter set and the
// exactly-once sink accounting, and the test names the first rung that
// diverges instead of failing somewhere inside a single large scenario.
//
// The ladder deliberately forces the worker count through
// `PoolConfig::testWorkerLimit` (a documented test-only knob, not a public API
// change) so a rung is reproducible on any device. Rung 7 uses the full launch
// geometry and the counts of the original failing case (17 descriptors, 712
// items, chunk 32).

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/pool.hpp>

#if GRAPH_HAS_CUDA

#    include "toy_ops.hpp"

#    include <chrono>
#    include <cstdint>
#    include <cstdio>
#    include <cstring>
#    include <span>
#    include <string>
#    include <utility>
#    include <vector>

namespace
{
    using ToyItem = graph::toy::ToyItem;
    using ToyResult = graph::toy::ToyResult;
    using ToyInstrumentation = graph::toy::ToyInstrumentation;

    /** Bind the toy operations to real device function pointers in device code. */
    struct ToyBinder
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC static bool bind(T_Acc const&, graph::OperationTable<T_Acc>& table)
        {
            using Ids = graph::toy::ToyOperationIds;
            Ids const ids{};
            bool ok = true;
            for(std::uint32_t index = 0u; index < table.count; ++index)
            {
                graph::OperationId const id = table.entries[index].id;
                if(id == ids.source)
                    ok = table.bindOperation(id, &graph::toy::ToySource<T_Acc>::operation) && ok;
                else if(id == ids.revisit)
                    ok = table.bindOperation(id, &graph::toy::ToyRevisit<T_Acc>::operation) && ok;
                else if(id == ids.sink)
                    ok = table.bindOperation(id, &graph::toy::ToySink<T_Acc>::operation) && ok;
                else
                    ok = false;
            }
            return ok;
        }
    };

    graph::ElementSchema toySchema(std::uint32_t chunk)
    {
        return graph::makeElementSchema<ToyItem>(alpaka::Vec<std::size_t, 1u>{chunk});
    }

    /** Source -> pass-through -> sink, one stream per edge. */
    graph::CompiledGraph buildLinearToyGraph(std::uint32_t chunk)
    {
        graph::GraphModel model;
        auto const schema = toySchema(chunk);
        graph::StreamId const sourceToRevisit{1u};
        graph::StreamId const revisitToSink{2u};
        model.addStream(graph::StreamDescriptor{sourceToRevisit, 0u, chunk, 1u});
        model.addStream(graph::StreamDescriptor{revisitToSink, 0u, chunk, 1u});

        graph::NodeDescriptor source;
        source.name = "ToySource";
        source.operation = graph::toy::defaultToyOperationIds.source;
        source.outputs.push_back(graph::OutputPortDescriptor{schema, sourceToRevisit, 0u});
        graph::NodeDescriptor revisit;
        revisit.name = "ToyRevisit";
        revisit.operation = graph::toy::defaultToyOperationIds.revisit;
        revisit.inputs.push_back(graph::InputPortDescriptor{schema});
        revisit.outputs.push_back(graph::OutputPortDescriptor{schema, revisitToSink, 0u});
        graph::NodeDescriptor sink;
        sink.name = "ToySink";
        sink.operation = graph::toy::defaultToyOperationIds.sink;
        sink.inputs.push_back(graph::InputPortDescriptor{schema});

        auto const sourceNode = model.addNode(std::move(source));
        auto const revisitNode = model.addNode(std::move(revisit));
        auto const sinkNode = model.addNode(std::move(sink));
        model.addEdge(graph::EdgeDescriptor{sourceNode, 0u, revisitNode, 0u, sourceToRevisit, 0u, false});
        model.addEdge(graph::EdgeDescriptor{revisitNode, 0u, sinkNode, 0u, revisitToSink, 0u, false});

        auto outcome = model.buildGraph();
        REQUIRE(graph::succeeded(outcome));
        return std::get<graph::CompiledGraph>(outcome);
    }

    /** One ladder rung. */
    struct Rung
    {
        char const* name;
        /** Test-only worker cap; zero means the full launch geometry. */
        std::uint32_t workers;
        std::uint32_t chunk;
        /** Item count of each submitted source request. */
        std::vector<std::uint64_t> requests;
        std::uint64_t seed;
    };

    /** Per-rung sink accounting from the per-id write counters. */
    struct SinkAudit
    {
        std::uint64_t correct{0u};
        std::uint64_t duplicated{0u};
        std::uint64_t missing{0u};
        std::uint64_t wrong{0u};
    };

    struct RungOutcome
    {
        bool green{false};
        graph::QuiescenceSnapshot counters{};
        graph::PoolStopReason reason{graph::PoolStopReason::neverStarted};
        std::uint32_t errorWord{0u};
        std::uint32_t geometryWorkers{0u};
        std::uint64_t totalItems{0u};
        SinkAudit sink{};
        std::uint64_t sinkWrites{0u};
        std::uint64_t controlErrorWord{0u};
        std::uint32_t stopStage{0u};
        std::uint64_t importBlocked{0u};
        std::uint64_t importEmpty{0u};
        std::string failure;
    };

    auto makeCudaHandles()
    {
        auto const spec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
        auto selector = alpaka::onHost::makeDeviceSelector(spec);
        REQUIRE(selector.isAvailable());
        REQUIRE(selector.getDeviceCount() > 0u);

        struct Handles
        {
            decltype(selector.makeDevice(0)) device;
            decltype(alpaka::onHost::defaultExecutor(selector.makeDevice(0))) executor;
        };

        auto device = selector.makeDevice(0);
        auto executor = alpaka::onHost::defaultExecutor(device);
        return Handles{std::move(device), std::move(executor)};
    }

    template<typename T_Device, typename T_Executor>
    RungOutcome runRung(T_Device& device, T_Executor executor, Rung const& rung, std::uint64_t maxItemId)
    {
        RungOutcome outcome;
        auto const chunk = rung.chunk;
        auto const resultCount = static_cast<std::uint32_t>(maxItemId);

        graph::PoolConfig config{};
        config.packetSchema = toySchema(chunk);
        config.chunkCapacity = chunk;
        config.sourceOperation = graph::toy::defaultToyOperationIds.source;
        config.budget.liveParticles = 64u;
        config.budget.controlAllowance = 8u;
        config.maxImportBatch = 8u;
        config.idLimit = 0u;
        config.testWorkerLimit = rung.workers;

        auto controlBuffer = alpaka::onHost::alloc<ToyInstrumentation>(device, 1u);
        auto hostControl = alpaka::onHost::allocHost<ToyInstrumentation>(1u);
        auto resultBuffer = alpaka::onHost::alloc<ToyResult>(device, resultCount);
        auto writeBuffer = alpaka::onHost::alloc<std::uint32_t>(device, resultCount);
        auto hostResult = alpaka::onHost::allocMapped<ToyResult>(device, resultCount);
        auto hostWrites = alpaka::onHost::allocHost<std::uint32_t>(resultCount);
        auto hostResults = alpaka::onHost::allocHost<ToyResult>(resultCount);
        auto queue = device.makeQueue();

        {
            std::vector<ToyResult> const zeroResults(resultCount, ToyResult{});
            std::vector<std::uint32_t> const zeroWrites(resultCount, 0u);
            std::memcpy(alpaka::onHost::data(hostResults), zeroResults.data(), zeroResults.size() * sizeof(ToyResult));
            std::memcpy(
                alpaka::onHost::data(hostWrites),
                zeroWrites.data(),
                zeroWrites.size() * sizeof(std::uint32_t));
            alpaka::onHost::memcpy(queue, resultBuffer, hostResults);
            alpaka::onHost::memcpy(queue, writeBuffer, hostWrites);
        }

        config.controlBlock = alpaka::onHost::data(controlBuffer);
        config.resultDevice = alpaka::onHost::data(resultBuffer);
        config.resultBytes = static_cast<std::uint64_t>(resultCount) * sizeof(ToyResult);

        ToyInstrumentation control{};
        control.results = alpaka::onHost::data(resultBuffer);
        control.resultCount = resultCount;
        control.sinkWrites = alpaka::onHost::data(writeBuffer);
        control.sinkWriteCount = resultCount;
        *alpaka::onHost::data(hostControl) = control;
        alpaka::onHost::memcpy(queue, controlBuffer, hostControl);
        alpaka::onHost::wait(queue);

        auto const compiled = buildLinearToyGraph(chunk);
        auto pool = graph::startPool(device, executor, compiled, config, ToyBinder{});
        outcome.geometryWorkers = pool.geometry().workerCount;

        std::vector<graph::RequestSpec> specs;
        specs.reserve(rung.requests.size());
        std::uint64_t firstId = 0u;
        for(auto const count : rung.requests)
        {
            specs.push_back(graph::RequestSpec{firstId, count, rung.seed});
            firstId += count;
            outcome.totalItems += count;
        }

        auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{specs});
        CHECK(submitted.accepted == specs.size());
        CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

        auto waited = pool.waitFor(std::chrono::seconds{120});
        if(!waited.has_value())
        {
            pool.abort();
            (void) pool.waitFor(std::chrono::seconds{10});
            outcome.failure = "pool did not stop within the deadline";
            return outcome;
        }

        pool.copyResultsToHost(alpaka::onHost::data(hostResult), sizeof(ToyResult) * resultCount);
        pool.waitCopies();
        alpaka::onHost::memcpy(queue, hostControl, controlBuffer);
        alpaka::onHost::memcpy(queue, hostWrites, writeBuffer);
        alpaka::onHost::wait(queue);

        auto const& result = *waited;
        outcome.reason = result.reason;
        outcome.errorWord = result.errorWord;
        outcome.counters = result.counters;
        outcome.stopStage = result.counters.debugStopStage;
        outcome.importBlocked = result.counters.importBlockedPolls;
        outcome.importEmpty = result.counters.importEmptyPolls;
        outcome.controlErrorWord = alpaka::onHost::data(hostControl)->errorWord;

        auto const* writes = alpaka::onHost::data(hostWrites);
        auto const* results = alpaka::onHost::data(hostResult);
        for(std::uint64_t id = 0u; id < outcome.totalItems; ++id)
        {
            auto const index = static_cast<std::uint32_t>(id);
            std::uint32_t const count = writes[index];
            outcome.sinkWrites += count;
            if(count == 0u)
            {
                ++outcome.sink.missing;
            }
            else if(count > 1u)
            {
                ++outcome.sink.duplicated;
            }
            else if(results[index].sinkVisits == 1u && results[index].value == graph::toy::toyValue(id, rung.seed))
            {
                ++outcome.sink.correct;
            }
            else
            {
                ++outcome.sink.wrong;
            }
        }

        bool const countersGreen = result.counters.outstanding == 0u && result.counters.pendingSources == 0u
                                   && result.counters.queued == 0u && result.counters.active == 0u;
        outcome.green = result.reason == graph::PoolStopReason::stoppedByDrain && result.errorWord == 0u
                        && outcome.controlErrorWord == 0u && countersGreen
                        && outcome.sink.correct == outcome.totalItems && outcome.sink.duplicated == 0u
                        && outcome.sink.missing == 0u && outcome.sink.wrong == 0u;
        if(!outcome.green)
        {
            outcome.failure = "not green";
        }
        return outcome;
    }

    void printRow(std::size_t index, Rung const& rung, RungOutcome const& out)
    {
        auto const& c = out.counters;
        std::printf(
            "[ladder] rung=%zu %-26s workers=%u accepted=%llu imported=%llu queued=%llu claimed=%llu "
            "executed=%llu published=%llu retired=%llu distinctWorkers=%llu items=%llu "
            "sink(correct/dup/missing/wrong)=%llu/%llu/%llu/%llu writes=%llu errorWord=0x%x ctlError=0x%llx "
            "stopReason=%u stopStage=%u blocked=%llu empty=%llu %s\n",
            index,
            rung.name,
            static_cast<unsigned>(out.geometryWorkers),
            static_cast<unsigned long long>(c.accepted),
            static_cast<unsigned long long>(c.imported),
            static_cast<unsigned long long>(c.queued),
            static_cast<unsigned long long>(c.claimAttempts),
            static_cast<unsigned long long>(c.completedJobs),
            static_cast<unsigned long long>(c.publications),
            static_cast<unsigned long long>(c.retiredPackets),
            static_cast<unsigned long long>(c.distinctWorkers),
            static_cast<unsigned long long>(out.totalItems),
            static_cast<unsigned long long>(out.sink.correct),
            static_cast<unsigned long long>(out.sink.duplicated),
            static_cast<unsigned long long>(out.sink.missing),
            static_cast<unsigned long long>(out.sink.wrong),
            static_cast<unsigned long long>(out.sinkWrites),
            out.errorWord,
            static_cast<unsigned long long>(out.controlErrorWord),
            static_cast<unsigned>(out.reason),
            static_cast<unsigned>(out.stopStage),
            static_cast<unsigned long long>(out.importBlocked),
            static_cast<unsigned long long>(out.importEmpty),
            out.green ? "OK" : "DIVERGED");
        std::fflush(stdout);
    }
} // namespace

TEST_CASE("pool gpu correctness ladder: rising worker count, exactly-once sink", "[graph][pool][ladder][gpu]")
{
    auto handles = makeCudaHandles();

    std::vector<Rung> rungs;
    rungs.push_back(Rung{"1w/1job/8items/chunk8", 1u, 8u, std::vector<std::uint64_t>{8u}, 17u});
    rungs.push_back(Rung{"1w/4jobs/8items/chunk8", 1u, 8u, std::vector<std::uint64_t>(4u, 8u), 17u});
    rungs.push_back(Rung{"1w/17jobs/8items/chunk8", 1u, 8u, std::vector<std::uint64_t>(17u, 8u), 17u});
    rungs.push_back(Rung{"2w/17jobs/8items/chunk8", 2u, 8u, std::vector<std::uint64_t>(17u, 8u), 17u});
    rungs.push_back(Rung{"8w/17jobs/8items/chunk8", 8u, 8u, std::vector<std::uint64_t>(17u, 8u), 17u});
    rungs.push_back(Rung{"64w/17jobs/8items/chunk8", 64u, 8u, std::vector<std::uint64_t>(17u, 8u), 17u});
    {
        // The original failing shape: 17 descriptors, 712 items, full geometry.
        std::vector<std::uint64_t> requests;
        requests.push_back(200u);
        for(std::uint32_t index = 0u; index < 16u; ++index)
            requests.push_back(32u);
        rungs.push_back(Rung{"full/17jobs/712items/chunk32", 0u, 32u, std::move(requests), 17u});
    }

    for(std::size_t index = 0u; index < rungs.size(); ++index)
    {
        auto const& rung = rungs[index];
        auto const maxItemId = [&rung]()
        {
            std::uint64_t total = 0u;
            for(auto const count : rung.requests)
                total += count;
            return total;
        }();
        auto const outcome = runRung(handles.device, handles.executor, rung, maxItemId);
        printRow(index + 1u, rung, outcome);
        INFO("rung " << (index + 1u) << " (" << rung.name << "): " << outcome.failure);
        CHECK(outcome.reason == graph::PoolStopReason::stoppedByDrain);
        CHECK(outcome.errorWord == 0u);
        CHECK(outcome.controlErrorWord == 0u);
        CHECK(outcome.counters.outstanding == 0u);
        CHECK(outcome.counters.pendingSources == 0u);
        CHECK(outcome.counters.queued == 0u);
        CHECK(outcome.counters.active == 0u);
        CHECK(outcome.sink.correct == outcome.totalItems);
        CHECK(outcome.sink.duplicated == 0u);
        CHECK(outcome.sink.missing == 0u);
        CHECK(outcome.sink.wrong == 0u);
    }
}

#endif // GRAPH_HAS_CUDA
