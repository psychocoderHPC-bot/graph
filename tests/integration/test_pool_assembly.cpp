// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Pool-level evidence for the per-destination chunk assembly contract (design
// sections 6 and 7). The scripted toy graph runs through the real persistent
// pool: the scheduler reserves destination ranges, copies the functor output,
// rotates the A/B chunks, publishes full chunks with their real element count,
// and drains partial buffers with a repeated quiescence observation.
//
// Every check is behavioural: the sink records observed chunk sizes and one
// result per id, so a wrong counter, a missing rotation, a duplicated element,
// or a partial that never drains is visible without inspecting scheduler state.

#include "toy_ops.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/graph_model.hpp>
#include <graph/pool.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

namespace
{
    using ToyItem = graph::toy::ToyItem;
    using ToyResult = graph::toy::ToyResult;
    using ToyInstrumentation = graph::toy::ToyInstrumentation;

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
                else if(id == ids.split)
                    ok = table.bindOperation(id, &graph::toy::ToySplit<T_Acc>::operation) && ok;
                else if(id == ids.revisit)
                    ok = table.bindOperation(id, &graph::toy::ToyRevisit<T_Acc>::operation) && ok;
                else if(id == ids.sink)
                    ok = table.bindOperation(id, &graph::toy::ToySink<T_Acc>::operation) && ok;
                else if(id == ids.abort)
                    ok = table.bindOperation(id, &graph::toy::ToyAbort<T_Acc>::operation) && ok;
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

    graph::CompiledGraph compile(graph::GraphModel const& model)
    {
        auto outcome = model.buildGraph();
        REQUIRE(graph::succeeded(outcome));
        return std::get<graph::CompiledGraph>(outcome);
    }

    /** Linear source -> revisit(no feedback) -> sink chain. */
    graph::CompiledGraph linearGraph(std::uint32_t chunk)
    {
        graph::GraphModel model;
        auto const schema = toySchema(chunk);
        graph::StreamId const first{1u};
        graph::StreamId const second{2u};
        model.addStream(graph::StreamDescriptor{first, 0u, chunk, 2u});
        model.addStream(graph::StreamDescriptor{second, 0u, chunk, 2u});

        graph::NodeDescriptor source;
        source.name = "ToySource";
        source.operation = graph::toy::defaultToyOperationIds.source;
        source.outputs.push_back(graph::OutputPortDescriptor{schema, first, 0u});
        graph::NodeDescriptor revisit;
        revisit.name = "ToyRevisit";
        revisit.operation = graph::toy::defaultToyOperationIds.revisit;
        revisit.inputs.push_back(graph::InputPortDescriptor{schema});
        revisit.outputs.push_back(graph::OutputPortDescriptor{schema, second, 0u});
        graph::NodeDescriptor sink;
        sink.name = "ToySink";
        sink.operation = graph::toy::defaultToyOperationIds.sink;
        sink.inputs.push_back(graph::InputPortDescriptor{schema});

        auto const sourceNode = model.addNode(std::move(source));
        auto const revisitNode = model.addNode(std::move(revisit));
        auto const sinkNode = model.addNode(std::move(sink));
        model.addEdge(graph::EdgeDescriptor{sourceNode, 0u, revisitNode, 0u, first, 0u, false});
        model.addEdge(graph::EdgeDescriptor{revisitNode, 0u, sinkNode, 0u, second, 0u, false});
        return compile(model);
    }

    graph::CompiledGraph feedbackGraph(std::uint32_t chunk, std::uint32_t revisitCount)
    {
        graph::toy::ToyGraphConfig config;
        config.chunkSize = chunk;
        config.revisitCount = revisitCount;
        config.feedback = true;
        return compile(graph::toy::makeToyGraph(config));
    }

    template<typename T_Device, typename T_Executor>
    class ToyAssemblyHarness
    {
    public:
        ToyAssemblyHarness(
            T_Device device,
            T_Executor executor,
            std::uint32_t chunk,
            std::uint32_t resultCount,
            std::uint32_t revisitCount)
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
            m_config.budget.liveParticles = std::max<std::uint32_t>(512u, resultCount);
            m_config.budget.controlAllowance = 16u;
            m_config.budget.replacementsPerWorker = 4u;
            m_config.maxImportBatch = 8u;
            // Keep the packet budget small enough for any backend while still
            // letting several warps compete on the destination streams.
            m_config.testWorkerLimit = 8u;
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
            m_control.revisitCount = revisitCount;
            *alpaka::onHost::data(m_hostControl) = m_control;
            alpaka::onHost::memcpy(m_queue, m_controlBuffer, m_hostControl);
            alpaka::onHost::wait(m_queue);
        }

        graph::PoolConfig& config()
        {
            return m_config;
        }

        graph::Pool start(graph::CompiledGraph const& compiled)
        {
            return graph::startPool(m_device, m_executor, compiled, m_config, ToyBinder{});
        }

        graph::PoolResult waitBounded(graph::Pool& pool, std::chrono::milliseconds timeout)
        {
            auto result = pool.waitFor(timeout);
            if(!result.has_value())
            {
                pool.abort();
                auto second = pool.waitFor(std::chrono::seconds{20});
                if(second.has_value())
                {
                    auto const& c = second->counters;
                    std::printf(
                        "[assembly] TIMEOUT errorWord=0x%x imported=%llu completed=%llu outstanding=%llu "
                        "pendingSources=%llu partial=%llu queued=%llu active=%llu publications=%llu "
                        "drainRounds=%llu stopReason=%u\n",
                        second->errorWord,
                        static_cast<unsigned long long>(c.imported),
                        static_cast<unsigned long long>(c.completedJobs),
                        static_cast<unsigned long long>(c.outstanding),
                        static_cast<unsigned long long>(c.pendingSources),
                        static_cast<unsigned long long>(c.partialBuffers),
                        static_cast<unsigned long long>(c.queued),
                        static_cast<unsigned long long>(c.active),
                        static_cast<unsigned long long>(c.publications),
                        static_cast<unsigned long long>(c.drainRounds),
                        c.stopReason);
                    std::fflush(stdout);
                }
                FAIL("assembly pool did not stop within the deadline");
                return second.has_value() ? *second : graph::PoolResult{};
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

        std::vector<std::uint32_t> chunkSizes() const
        {
            auto const count = std::min(m_control.chunkSizeCount, m_control.chunkSizeCapacity);
            return std::vector<std::uint32_t>(m_control.chunkSizes, m_control.chunkSizes + count);
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

    auto makeCpuHandles()
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

#if GRAPH_HAS_CUDA
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
#endif

    template<typename T_Harness>
    void checkEveryIdOnce(T_Harness& harness, std::uint64_t count, std::uint64_t seed, std::uint32_t visit)
    {
        std::uint64_t mismatches = 0u;
        for(std::uint64_t id = 0u; id < count; ++id)
        {
            auto const stored = harness.result(static_cast<std::uint32_t>(id));
            bool const ok
                = stored.sinkVisits == 1u && stored.value == graph::toy::toyValue(id, seed) && stored.visits == visit;
            if(!ok && mismatches < 8u)
                std::printf(
                    "[assembly] mismatch id=%llu sinkVisits=%u visits=%u value=0x%x expected=0x%x\n",
                    static_cast<unsigned long long>(id),
                    stored.sinkVisits,
                    stored.visits,
                    stored.value,
                    graph::toy::toyValue(id, seed));
            mismatches += ok ? 0u : 1u;
        }
        CHECK(mismatches == 0u);
        CHECK(harness.control().errorWord == 0u);
    }

    template<typename T_Harness>
    void drainOk(T_Harness& harness, graph::PoolResult const& result)
    {
        CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
        CHECK(result.errorWord == 0u);
        CHECK(result.counters.outstanding == 0u);
        CHECK(result.counters.pendingSources == 0u);
        CHECK(result.counters.partialBuffers == 0u);
        CHECK(result.counters.queued == 0u);
        CHECK(result.counters.active == 0u);
        if(result.errorWord != 0u)
            std::printf(
                "[assembly] errorWord=0x%x outstanding=%llu queued=%llu stopStage=%u\n",
                result.errorWord,
                static_cast<unsigned long long>(result.counters.outstanding),
                static_cast<unsigned long long>(result.counters.queued),
                result.counters.debugStopStage);
        (void) harness;
    }

    template<typename T_Device, typename T_Executor>
    void runLinearChunkCase(
        T_Device device,
        T_Executor executor,
        std::uint32_t chunk,
        std::uint64_t count,
        std::vector<std::uint32_t> const& expectedSizes)
    {
        ToyAssemblyHarness<T_Device, T_Executor>
            harness(device, executor, chunk, static_cast<std::uint32_t>(std::max<std::uint64_t>(count, 1u)), 0u);
        auto pool = harness.start(linearGraph(chunk));
        std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, count, 42u}};
        REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
        CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
        auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
        harness.fetch(pool);
        drainOk(harness, result);
        if(count != 0u)
            checkEveryIdOnce(harness, count, 42u, 0u);
        auto const sizes = harness.chunkSizes();
        std::vector<std::uint32_t> sortedExpected = expectedSizes;
        std::vector<std::uint32_t> sortedActual = sizes;
        std::sort(sortedExpected.begin(), sortedExpected.end());
        std::sort(sortedActual.begin(), sortedActual.end());
        CHECK(sortedActual == sortedExpected);
    }
} // namespace

TEST_CASE("pool assembly counter four plus input 128 publishes full and carries four", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    ToyAssemblyHarness harness(cpu.device, cpu.executor, 128u, 132u, 0u);
    auto pool = harness.start(linearGraph(128u));

    // Append four elements first and wait until that source job retired, so
    // the next append is deterministic: c == 4, k == 128, c + k == 132.
    std::array<graph::RequestSpec, 1u> const first{graph::RequestSpec{0u, 4u, 42u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{first}).accepted == 1u);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while(pool.completedJobs() < 1u && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    REQUIRE(pool.completedJobs() >= 1u);

    std::array<graph::RequestSpec, 1u> const second{graph::RequestSpec{4u, 128u, 42u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{second}).accepted == 1u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    drainOk(harness, result);
    checkEveryIdOnce(harness, 132u, 42u, 0u);
    auto sizes = harness.chunkSizes();
    std::sort(sizes.begin(), sizes.end());
    CHECK(sizes == std::vector<std::uint32_t>{4u, 128u});
}

TEST_CASE("pool assembly exactly full publishes the full chunk", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    runLinearChunkCase(cpu.device, cpu.executor, 32u, 32u, {32u});
}

TEST_CASE("pool assembly partial input stays buffered until drain", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    runLinearChunkCase(cpu.device, cpu.executor, 16u, 7u, {7u});
}

TEST_CASE("pool assembly capacity one rotates on every append", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    runLinearChunkCase(cpu.device, cpu.executor, 1u, 8u, std::vector<std::uint32_t>(8u, 1u));
}

TEST_CASE("pool assembly non-warp-multiple capacity Y equals 100", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    runLinearChunkCase(cpu.device, cpu.executor, 100u, 250u, {100u, 100u, 50u});
}

TEST_CASE("pool assembly zero-count source port publishes nothing", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    ToyAssemblyHarness harness(cpu.device, cpu.executor, 16u, 1u, 0u);
    auto pool = harness.start(linearGraph(16u));
    std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, 0u, 42u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    drainOk(harness, result);
    CHECK(harness.chunkSizes().empty());
    CHECK(harness.control().invocations[static_cast<std::uint32_t>(graph::toy::ToyOpSlot::Sink)] == 0u);
}

TEST_CASE("pool assembly both destinations with competing writers", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    std::uint32_t const chunk = 32u;
    std::uint64_t const count = 256u;
    ToyAssemblyHarness harness(cpu.device, cpu.executor, chunk, static_cast<std::uint32_t>(count), 0u);
    harness.config().maxReadyInFlight = 64u;
    auto pool = harness.start(feedbackGraph(chunk, 0u));

    std::vector<graph::RequestSpec> requests;
    for(std::uint64_t first = 0u; first < count; first += 32u)
        requests.push_back(graph::RequestSpec{first, 32u, 7u});
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == requests.size());
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    drainOk(harness, result);
    checkEveryIdOnce(harness, count, 7u, 0u);
    auto const sizes = harness.chunkSizes();
    CHECK(!sizes.empty());
    for(auto const size : sizes)
        CHECK(size <= chunk);
    CHECK(std::count(sizes.begin(), sizes.end(), chunk) > 0);
    CHECK(result.counters.distinctWorkers >= 2u);
}

TEST_CASE("pool assembly toy feedback revisits exactly and drains returned particles", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    std::uint32_t const chunk = 8u;
    std::uint32_t const revisitCount = 3u;
    std::uint64_t const count = 64u;
    ToyAssemblyHarness harness(cpu.device, cpu.executor, chunk, static_cast<std::uint32_t>(count), revisitCount);
    harness.config().maxReadyInFlight = 8u;
    harness.config().budget.liveParticles = 64u;
    auto pool = harness.start(feedbackGraph(chunk, revisitCount));

    std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, count, 11u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{120});
    harness.fetch(pool);
    drainOk(harness, result);
    checkEveryIdOnce(harness, count, 11u, revisitCount);
}

TEST_CASE("pool assembly returned particle forces another drain round", "[graph][pool][assembly][cpu]")
{
    auto cpu = makeCpuHandles();
    std::uint32_t const chunk = 8u;
    std::uint32_t const revisitCount = 3u;
    // One particle only: every append is partial, so the particle is carried
    // from partial buffer to partial buffer and can only finish through a
    // sequence of drain rounds. A returned particle that created a new partial
    // would be lost if quiescence were not re-observed after draining.
    ToyAssemblyHarness harness(cpu.device, cpu.executor, chunk, 1u, revisitCount);
    harness.config().maxReadyInFlight = 4u;
    harness.config().budget.liveParticles = 8u;
    auto pool = harness.start(feedbackGraph(chunk, revisitCount));

    std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, 1u, 23u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{60});
    harness.fetch(pool);
    drainOk(harness, result);
    checkEveryIdOnce(harness, 1u, 23u, revisitCount);
    CHECK(result.counters.drainRounds >= 2u);
}

#if GRAPH_HAS_CUDA
TEST_CASE("pool assembly runs on the CUDA backend", "[graph][pool][assembly][cuda]")
{
    auto cuda = makeCudaHandles();
    std::uint32_t const chunk = 16u;
    std::uint64_t const count = 128u;
    ToyAssemblyHarness harness(cuda.device, cuda.executor, chunk, static_cast<std::uint32_t>(count), 2u);
    harness.config().maxReadyInFlight = 16u;
    harness.config().budget.liveParticles = 256u;
    auto pool = harness.start(feedbackGraph(chunk, 2u));
    std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, count, 5u}};
    REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const result = harness.waitBounded(pool, std::chrono::seconds{20});
    harness.fetch(pool);
    drainOk(harness, result);
    checkEveryIdOnce(harness, count, 5u, 2u);
}
#endif
