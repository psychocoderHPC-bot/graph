// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// CUDA evidence for the persistent warp pool: one real launch serves several
// batches, at least two distinct device function pointers execute consecutive
// jobs with correct results, more than one worker warp consumes work, an idle
// pool resumes, a deliberately small pool survives full-capacity pressure, and
// the final shutdown drains cleanly. Every wait has a finite deadline.

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/pool.hpp>

#if GRAPH_HAS_CUDA

#    include "toy_ops.hpp"

#    include <algorithm>
#    include <array>
#    include <chrono>
#    include <cstdint>
#    include <cstdio>
#    include <cstring>
#    include <span>
#    include <thread>
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

    template<typename T_Device, typename T_Executor>
    class ToyHarness
    {
    public:
        ToyHarness(T_Device device, T_Executor executor, std::uint32_t chunk, std::uint32_t resultCount)
            : m_device(device)
            , m_executor(executor)
            , m_chunk(chunk)
            , m_resultCount(resultCount)
            , m_controlBuffer(alpaka::onHost::alloc<ToyInstrumentation>(m_device, 1u))
            , m_hostControl(alpaka::onHost::allocHost<ToyInstrumentation>(1u))
            , m_resultBuffer(alpaka::onHost::alloc<ToyResult>(m_device, std::max(resultCount, 1u)))
            , m_writeBuffer(alpaka::onHost::alloc<std::uint32_t>(m_device, std::max(resultCount, 1u)))
            , m_hostResult(alpaka::onHost::allocMapped<ToyResult>(m_device, std::max(resultCount, 1u)))
            , m_queue(m_device.makeQueue())
        {
            m_config.packetSchema = toySchema(chunk);
            m_config.chunkCapacity = chunk;
            m_config.sourceOperation = graph::toy::defaultToyOperationIds.source;
            m_config.budget.liveParticles = 64u;
            m_config.budget.controlAllowance = 8u;
            m_config.maxImportBatch = 8u;
            m_config.idLimit = 0u;
            m_config.controlBlock = alpaka::onHost::data(m_controlBuffer);
            m_config.resultDevice = alpaka::onHost::data(m_resultBuffer);
            m_config.resultBytes = static_cast<std::uint64_t>(resultCount) * sizeof(ToyResult);

            auto hostResults = alpaka::onHost::allocHost<ToyResult>(std::max(resultCount, 1u));
            auto hostWrites = alpaka::onHost::allocHost<std::uint32_t>(std::max(resultCount, 1u));
            std::vector<ToyResult> const zeroResults(std::max(resultCount, 1u), ToyResult{});
            std::vector<std::uint32_t> const zeroWrites(std::max(resultCount, 1u), 0u);
            std::memcpy(alpaka::onHost::data(hostResults), zeroResults.data(), zeroResults.size() * sizeof(ToyResult));
            std::memcpy(
                alpaka::onHost::data(hostWrites),
                zeroWrites.data(),
                zeroWrites.size() * sizeof(std::uint32_t));
            alpaka::onHost::memcpy(m_queue, m_resultBuffer, hostResults);
            alpaka::onHost::memcpy(m_queue, m_writeBuffer, hostWrites);

            m_control = ToyInstrumentation{};
            m_control.results = alpaka::onHost::data(m_resultBuffer);
            m_control.resultCount = resultCount;
            m_control.sinkWrites = alpaka::onHost::data(m_writeBuffer);
            m_control.sinkWriteCount = resultCount;
            *alpaka::onHost::data(m_hostControl) = m_control;
            alpaka::onHost::memcpy(m_queue, m_controlBuffer, m_hostControl);
            alpaka::onHost::wait(m_queue);
        }

        ToyHarness(ToyHarness const&) = delete;
        ToyHarness& operator=(ToyHarness const&) = delete;

        graph::PoolConfig& config()
        {
            return m_config;
        }

        graph::Pool start()
        {
            auto const compiled = buildLinearToyGraph(m_chunk);
            return graph::startPool(m_device, m_executor, compiled, m_config, ToyBinder{});
        }

        graph::PoolResult waitBounded(graph::Pool& pool, std::chrono::milliseconds timeout)
        {
            auto result = pool.waitFor(timeout);
            if(!result.has_value())
            {
                pool.abort();
                (void) pool.waitFor(std::chrono::seconds{10});
                FAIL("pool did not stop within the deadline");
                return graph::PoolResult{};
            }
            return *result;
        }

        void fetch(graph::Pool& pool)
        {
            pool.copyResultsToHost(alpaka::onHost::data(m_hostResult), sizeof(ToyResult) * m_resultCount);
            pool.waitCopies();
            alpaka::onHost::memcpy(m_queue, m_hostControl, m_controlBuffer);
            alpaka::onHost::wait(m_queue);
            m_control = *alpaka::onHost::data(m_hostControl);
        }

        ToyResult result(std::uint32_t id) const
        {
            return alpaka::onHost::data(m_hostResult)[id];
        }

        ToyInstrumentation const& control() const
        {
            return m_control;
        }

        std::uint64_t sourceInvocations() const
        {
            return m_control.invocations[static_cast<std::uint32_t>(graph::toy::ToyOpSlot::Source)];
        }

        std::uint64_t revisitInvocations() const
        {
            return m_control.invocations[static_cast<std::uint32_t>(graph::toy::ToyOpSlot::Revisit)];
        }

        std::uint64_t sinkInvocations() const
        {
            return m_control.invocations[static_cast<std::uint32_t>(graph::toy::ToyOpSlot::Sink)];
        }

    private:
        T_Device m_device;
        T_Executor m_executor;
        std::uint32_t m_chunk;
        std::uint32_t m_resultCount;
        graph::PoolConfig m_config{};
        decltype(alpaka::onHost::alloc<ToyInstrumentation>(std::declval<T_Device>(), 1u)) m_controlBuffer;
        decltype(alpaka::onHost::allocHost<ToyInstrumentation>(1u)) m_hostControl;
        decltype(alpaka::onHost::alloc<ToyResult>(std::declval<T_Device>(), 1u)) m_resultBuffer;
        decltype(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device>(), 1u)) m_writeBuffer;
        decltype(alpaka::onHost::allocMapped<ToyResult>(std::declval<T_Device>(), 1u)) m_hostResult;
        decltype(std::declval<T_Device&>().makeQueue()) m_queue;
        ToyInstrumentation m_control{};
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
        return Handles{std::move(device), executor};
    }

    template<typename T_Harness>
    void checkResults(T_Harness& harness, std::uint64_t count, std::uint64_t seed)
    {
        std::uint64_t mismatches = 0u;
        for(std::uint64_t id = 0u; id < count; ++id)
        {
            auto const stored = harness.result(static_cast<std::uint32_t>(id));
            if(stored.sinkVisits != 1u || stored.value != graph::toy::toyValue(id, seed))
            {
                if(mismatches < 8u)
                    std::printf(
                        "[pool-gpu] mismatch id=%llu visits=%u value=0x%x expected=0x%x\n",
                        static_cast<unsigned long long>(id),
                        stored.sinkVisits,
                        stored.value,
                        graph::toy::toyValue(id, seed));
                ++mismatches;
            }
        }
        std::printf(
            "[pool-gpu] mismatches=%llu of %llu\n",
            static_cast<unsigned long long>(mismatches),
            static_cast<unsigned long long>(count));
        std::fflush(stdout);
        CHECK(mismatches == 0u);
        CHECK(harness.control().errorWord == 0u);
    }

    std::vector<graph::RequestSpec> makeRequests(
        std::uint64_t firstId,
        std::uint64_t countPerRequest,
        std::uint32_t requestCount,
        std::uint64_t seed)
    {
        std::vector<graph::RequestSpec> requests;
        requests.reserve(requestCount);
        for(std::uint32_t index = 0u; index < requestCount; ++index)
        {
            requests.push_back(
                graph::RequestSpec{
                    firstId + static_cast<std::uint64_t>(index) * countPerRequest,
                    countPerRequest,
                    seed});
        }
        return requests;
    }

    void printSnapshot(char const* name, graph::Pool const& pool, graph::PoolResult const& result)
    {
        auto const geometry = pool.geometry();
        std::printf(
            "[pool-gpu] %s geometry blocks=%u threadsPerBlock=%u warpsPerBlock=%u workers=%u smCount=%u "
            "residentsPerSm=%u launchCount=%llu distinctWorkers=%llu accepted=%llu imported=%llu completed=%llu "
            "publications=%llu pendingSources=%llu outstanding=%llu queued=%llu active=%llu errorWord=0x%x "
            "stopReason=%u readyViolations=%llu stopStage=%u\n",
            name,
            static_cast<unsigned>(geometry.blocks),
            static_cast<unsigned>(geometry.threadsPerBlock),
            static_cast<unsigned>(geometry.warpsPerBlock),
            static_cast<unsigned>(geometry.workerCount),
            static_cast<unsigned>(geometry.smCount),
            static_cast<unsigned>(geometry.residentsPerSm),
            static_cast<unsigned long long>(result.counters.launchCount),
            static_cast<unsigned long long>(result.counters.distinctWorkers),
            static_cast<unsigned long long>(result.counters.accepted),
            static_cast<unsigned long long>(result.counters.imported),
            static_cast<unsigned long long>(result.counters.completedJobs),
            static_cast<unsigned long long>(result.counters.publications),
            static_cast<unsigned long long>(result.counters.pendingSources),
            static_cast<unsigned long long>(result.counters.outstanding),
            static_cast<unsigned long long>(result.counters.queued),
            static_cast<unsigned long long>(result.counters.active),
            result.counters.errorWord,
            result.counters.stopReason,
            static_cast<unsigned long long>(result.counters.readyViolations),
            static_cast<unsigned>(result.counters.debugStopStage));
        std::fflush(stdout);
    }
} // namespace

TEST_CASE("pool gpu one launch, two function pointers, many workers, clean drain", "[graph][pool][gpu]")
{
    auto handles = makeCudaHandles();
    ToyHarness harness(handles.device, handles.executor, 32u, 1024u);
    auto pool = harness.start();

    // 1) A single request exercises ToySource and ToySink (two distinct device
    //    function pointers) with a resumable continuation.
    std::array<graph::RequestSpec, 1u> const single{graph::RequestSpec{0u, 200u, 17u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{single}).accepted == 1u);

    // 2) Many batches while the same kernel stays resident: launch-once proof.
    for(std::uint32_t batch = 0u; batch < 16u; ++batch)
    {
        auto const requests = makeRequests(200u + batch * 32u, 32u, 1u, 17u);
        REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }

    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{120});
    harness.fetch(pool);

    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.errorWord == 0u);
    CHECK(result.counters.launchCount == 1u);
    CHECK(result.counters.distinctWorkers >= 2u);
    CHECK(result.counters.outstanding == 0u);
    CHECK(result.counters.pendingSources == 0u);
    CHECK(result.counters.queued == 0u);
    CHECK(result.counters.active == 0u);
    CHECK(harness.sourceInvocations() == harness.revisitInvocations());
    CHECK(harness.revisitInvocations() == harness.sinkInvocations());
    CHECK(harness.sourceInvocations() > 1u);
    checkResults(harness, 200u + 16u * 32u, 17u);
    printSnapshot("drain", pool, result);
}

TEST_CASE("pool gpu resumes from idle and reports the resume latency", "[graph][pool][gpu]")
{
    auto handles = makeCudaHandles();
    ToyHarness harness(handles.device, handles.executor, 32u, 256u);
    auto pool = harness.start();

    std::this_thread::sleep_for(std::chrono::milliseconds{150});
    CHECK(pool.completedJobs() == 0u);

    auto const requests = makeRequests(0u, 32u, 8u, 23u);
    auto const started = std::chrono::steady_clock::now();
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == requests.size());

    // Wait until every source job and its successor chain retired.
    std::uint64_t const expectedJobs = 8u * 3u;
    while(pool.completedJobs() < expectedJobs)
    {
        REQUIRE(std::chrono::steady_clock::now() - started < std::chrono::seconds{60});
        std::this_thread::sleep_for(std::chrono::microseconds{50});
    }
    auto const resumed = std::chrono::steady_clock::now();
    double const latencyMs = std::chrono::duration<double, std::milli>(resumed - started).count();

    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{120});
    harness.fetch(pool);

    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.counters.launchCount == 1u);
    checkResults(harness, 256u, 23u);
    std::printf(
        "[pool-gpu] idle-then-resume latency for %llu jobs: %.3f ms\n",
        static_cast<unsigned long long>(expectedJobs),
        latencyMs);
    std::fflush(stdout);
    printSnapshot("idle-resume", pool, result);
}

TEST_CASE("pool gpu full-capacity pressure does not deadlock", "[graph][pool][gpu]")
{
    auto handles = makeCudaHandles();
    ToyHarness harness(handles.device, handles.executor, 4u, 2048u);
    harness.config().maxReadyInFlight = 2u;
    harness.config().maxImportBatch = 4u;
    auto pool = harness.start();

    auto const requests = makeRequests(0u, 8u, 256u, 41u);
    auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    REQUIRE(submitted.accepted == requests.size());
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

    auto const result = harness.waitBounded(pool, std::chrono::seconds{120});
    harness.fetch(pool);
    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.errorWord == 0u);
    CHECK(result.counters.outstanding == 0u);
    CHECK(result.counters.pendingSources == 0u);
    checkResults(harness, 2048u, 41u);
    printSnapshot("pressure", pool, result);
}

#endif // GRAPH_HAS_CUDA
