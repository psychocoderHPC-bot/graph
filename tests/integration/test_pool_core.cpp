// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// CPU backend integration test of the persistent warp pool: a real single
// launch, many jobs, multiple worker warps, source continuation re-queue,
// idle-then-resume, zero-work close, full-capacity pressure, and abort.
//
// The graph is a linear acyclic toy chain:
//
//     ToySource -> ToyRevisit -> ToySink
//
// ToySource is a resumable id range, ToyRevisit is a one-to-one operation (its
// feedback port is disabled in this configuration), and ToySink records one
// result per id. Pool-core forwards one packet in place per outgoing edge;
// per-destination output-size accounting belongs to the chunk-assembly
// milestone, so the fan-out ToySplit operation is not part of this chain.

#include "toy_ops.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/graph_model.hpp>
#include <graph/pool.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace
{
    using ToyItem = graph::toy::ToyItem;
    using ToyResult = graph::toy::ToyResult;
    using ToyInstrumentation = graph::toy::ToyInstrumentation;

    /** Binds the three toy operations to real device function pointers.
     *
     * The table is filled in device code; the host only publishes ids. The
     * indirect call in the scheduler goes through these pointers.
     */
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

    /** Owns every host/device buffer a test needs and the pool lifetime. */
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

            std::vector<ToyResult> const zeroResults(std::max(resultCount, 1u), ToyResult{});
            std::vector<std::uint32_t> const zeroWrites(std::max(resultCount, 1u), 0u);
            auto hostResults = alpaka::onHost::allocHost<ToyResult>(std::max(resultCount, 1u));
            auto hostWrites = alpaka::onHost::allocHost<std::uint32_t>(std::max(resultCount, 1u));
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

        /** Block until the pool stops or the deadline expires.
         *
         * On timeout the pool is aborted so a broken test cannot leave a
         * persistent launch behind, and the deadline is reported as a failure.
         */
        graph::PoolResult waitBounded(graph::Pool& pool, std::chrono::milliseconds timeout)
        {
            auto result = pool.waitFor(timeout);
            if(!result.has_value())
            {
                std::printf(
                    "[pool] WAIT TIMEOUT, aborting (completed=%llu)\n",
                    static_cast<unsigned long long>(pool.completedJobs()));
                std::fflush(stdout);
                pool.abort();
                auto second = pool.waitFor(std::chrono::seconds{10});
                FAIL("pool did not stop within the deadline");
                return second.has_value() ? *second : graph::PoolResult{};
            }
            return *result;
        }

        /** Copy the control block and the result buffer back to the host. */
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

    auto makeCpuDevice()
    {
        auto const spec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
        auto selector = alpaka::onHost::makeDeviceSelector(spec);
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
            auto const offset = static_cast<std::uint64_t>(index) * countPerRequest;
            requests.push_back(graph::RequestSpec{firstId + offset, countPerRequest, seed});
        }
        return requests;
    }

    template<typename T_Harness>
    void checkResults(T_Harness& harness, std::uint64_t count, std::uint64_t seed)
    {
        for(std::uint64_t id = 0u; id < count; ++id)
        {
            auto const stored = harness.result(static_cast<std::uint32_t>(id));
            if(stored.sinkVisits != 1u || stored.value != graph::toy::toyValue(id, seed))
            {
                CAPTURE(id, stored.sinkVisits, stored.value);
                FAIL("sink result mismatch");
            }
        }
        CHECK(harness.control().errorWord == 0u);
    }
} // namespace

TEST_CASE("pool core serves many jobs on multiple CPU workers with one launch", "[graph][pool][core][cpu]")
{
    auto cpu = makeCpuDevice();
    ToyHarness harness(cpu.device, cpu.executor, 16u, 512u);
    auto pool = harness.start();

    auto const requests = makeRequests(0u, 64u, 8u, 5u);
    auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    REQUIRE(submitted.status == graph::PoolEnqueueStatus::accepted);
    REQUIRE(submitted.accepted == requests.size());
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.errorWord == 0u);
    CHECK(result.counters.launchCount == 1u);
    CHECK(result.counters.distinctWorkers >= 2u);
    CHECK(result.counters.outstanding == 0u);
    CHECK(result.counters.pendingSources == 0u);
    CHECK(result.counters.queued == 0u);
    CHECK(result.counters.active == 0u);
    CHECK(pool.geometry().warpsPerBlock == 1u);
    CHECK(pool.geometry().workerCount == result.counters.workerCount);
    CHECK(harness.sourceInvocations() == 32u);
    checkResults(harness, 512u, 5u);

    std::printf(
        "[pool-core] backend=cpu workers=%u distinctWorkers=%llu launchCount=%llu sourceInvocations=%llu\n",
        static_cast<unsigned>(pool.geometry().workerCount),
        static_cast<unsigned long long>(result.counters.distinctWorkers),
        static_cast<unsigned long long>(result.counters.launchCount),
        static_cast<unsigned long long>(harness.sourceInvocations()));
    std::fflush(stdout);
}

TEST_CASE("pool core accepts a late submission to an already running idle pool", "[graph][pool][core][cpu]")
{
    auto cpu = makeCpuDevice();
    ToyHarness harness(cpu.device, cpu.executor, 16u, 128u);
    auto pool = harness.start();

    // The pool is running and completely idle here.
    std::this_thread::sleep_for(std::chrono::milliseconds{150});
    CHECK(pool.completedJobs() == 0u);

    auto const requests = makeRequests(0u, 32u, 4u, 9u);
    auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    REQUIRE(submitted.status == graph::PoolEnqueueStatus::accepted);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.counters.launchCount == 1u);
    checkResults(harness, 128u, 9u);
}

TEST_CASE("pool core closes with zero work", "[graph][pool][core][cpu]")
{
    auto cpu = makeCpuDevice();
    ToyHarness harness(cpu.device, cpu.executor, 8u, 4u);
    auto pool = harness.start();
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.counters.completedJobs == 0u);
    CHECK(result.counters.launchCount == 1u);
    CHECK(result.counters.outstanding == 0u);
    CHECK(result.counters.pendingSources == 0u);
}

TEST_CASE("pool core re-queues a source continuation without flooding", "[graph][pool][core][cpu]")
{
    auto cpu = makeCpuDevice();
    ToyHarness harness(cpu.device, cpu.executor, 8u, 100u);
    auto pool = harness.start();

    std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, 100u, 21u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.counters.pendingSources == 0u);
    CHECK(result.counters.outstanding == 0u);
    // 100 items in chunks of 8: twelve full chunks and one four-item chunk.
    CHECK(harness.sourceInvocations() == 13u);
    CHECK(harness.control().lastEmittedCount <= 8u);
    checkResults(harness, 100u, 21u);
}

TEST_CASE("pool core survives full-capacity pressure on a deliberately small pool", "[graph][pool][core][cpu]")
{
    auto cpu = makeCpuDevice();
    ToyHarness harness(cpu.device, cpu.executor, 2u, 1024u);
    harness.config().maxReadyInFlight = 2u;
    harness.config().maxImportBatch = 4u;
    auto pool = harness.start();

    auto const requests = makeRequests(0u, 8u, 128u, 33u);
    auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    REQUIRE(submitted.accepted == requests.size());
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);

    auto const result = harness.waitBounded(pool, std::chrono::seconds{120});
    harness.fetch(pool);
    CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
    CHECK(result.errorWord == 0u);
    CHECK(result.counters.outstanding == 0u);
    CHECK(result.counters.pendingSources == 0u);
    checkResults(harness, 1024u, 33u);
}

TEST_CASE("pool core abort stops a pool with pending work and reports an abort", "[graph][pool][core][cpu]")
{
    auto cpu = makeCpuDevice();
    ToyHarness harness(cpu.device, cpu.executor, 1u, 1u);
    auto pool = harness.start();

    std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, 2'000'000u, 1u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
    pool.abort();

    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    CHECK(result.reason == graph::PoolStopReason::stoppedByAbort);
    CHECK(graph::poolErrorSet(result.errorWord, graph::PoolError::aborted));
}
