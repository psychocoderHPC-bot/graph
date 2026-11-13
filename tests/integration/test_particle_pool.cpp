// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// End-to-end particle-feedback example through the real persistent pool.
//
// The five named functors are registered under the operation ABI and the
// scheduler assembles every stream. The test compares, for N=4096 seed 42 and
// chunk size 128, the per-id final weight and the exact sparse histogram against
// the committed fixture `tests/reference/test-vectors.md`; a local scalar
// reference using the same documented RNG stream checks the per-id values. The
// same body runs on the CPU backend and, when enabled, on the CUDA backend.

#include "oracle.hpp"
#include "toy_ops.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/graph_model.hpp>
#include <graph/particle_feedback_graph.hpp>
#include <graph/particle_functors.hpp>
#include <graph/pool.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <span>
#include <thread>
#include <vector>

namespace
{
    using Particle = graph::Particle;
    using ResultBlock = graph::particle_feedback::ResultBlock;
    using Weight = std::uint32_t;

    struct ParticleBinder
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC static bool bind(T_Acc const&, graph::OperationTable<T_Acc>& table)
        {
            using Ids = graph::particle_feedback::OperationIds;
            Ids const ids{};
            bool ok = true;
            for(std::uint32_t index = 0u; index < table.count; ++index)
            {
                graph::OperationId const id = table.entries[index].id;
                if(id == ids.generateParticles)
                    ok = table.bindOperation(id, &graph::GenerateParticles::template operation<T_Acc>) && ok;
                else if(id == ids.filterByProperty)
                    ok = table.bindOperation(id, &graph::FilterByProperty::template operation<T_Acc>) && ok;
                else if(id == ids.processA)
                    ok = table.bindOperation(id, &graph::ProcessA::template operation<T_Acc>) && ok;
                else if(id == ids.processB)
                    ok = table.bindOperation(id, &graph::ProcessB::template operation<T_Acc>) && ok;
                else if(id == ids.destroyParticles)
                    ok = table.bindOperation(id, &graph::DestroyParticles::template operation<T_Acc>) && ok;
                else
                    ok = false;
            }
            return ok;
        }
    };

    graph::CompiledGraph compileParticleGraph(std::uint32_t chunk)
    {
        auto const model = graph::makeParticleFeedbackGraph(chunk);
        auto outcome = model.buildGraph();
        REQUIRE(graph::succeeded(outcome));
        return std::get<graph::CompiledGraph>(outcome);
    }

    template<typename T_Device, typename T_Executor>
    class ParticlePoolHarness
    {
    public:
        ParticlePoolHarness(T_Device device, T_Executor executor, std::uint32_t chunk, std::uint32_t count)
            : m_device(device)
            , m_executor(executor)
            , m_chunk(chunk)
            , m_count(count)
            , m_controlBuffer(alpaka::onHost::alloc<ResultBlock>(m_device, 1u))
            , m_hostControl(alpaka::onHost::allocHost<ResultBlock>(1u))
            , m_weights(alpaka::onHost::alloc<std::int64_t>(m_device, std::max(count, 1u)))
            , m_sinkWrites(alpaka::onHost::alloc<std::uint32_t>(m_device, std::max(count, 1u)))
            , m_hostWeights(alpaka::onHost::allocHost<std::int64_t>(std::max(count, 1u)))
            , m_hostSinkWrites(alpaka::onHost::allocHost<std::uint32_t>(std::max(count, 1u)))
            , m_queue(m_device.makeQueue())
        {
            m_config.packetSchema
                = graph::makeElementSchema<Particle>(alpaka::Vec<std::size_t, 1u>{static_cast<std::size_t>(chunk)});
            m_config.chunkCapacity = chunk;
            m_config.sourceOperation = graph::particle_feedback::defaultOperationIds.generateParticles;
            m_config.budget.liveParticles = count + chunk;
            m_config.budget.controlAllowance = 16u;
            m_config.budget.replacementsPerWorker = 4u;
            m_config.maxImportBatch = 8u;
            m_config.testWorkerLimit = 1u;
            m_config.controlBlock = alpaka::onHost::data(m_controlBuffer);

            std::vector<std::int64_t> const zeroWeights(std::max(count, 1u), std::int64_t{-1});
            std::vector<std::uint32_t> const zeroWrites(std::max(count, 1u), 0u);
            auto hostWeights = alpaka::onHost::allocHost<std::int64_t>(std::max(count, 1u));
            auto hostWrites = alpaka::onHost::allocHost<std::uint32_t>(std::max(count, 1u));
            std::memcpy(
                alpaka::onHost::data(hostWeights),
                zeroWeights.data(),
                zeroWeights.size() * sizeof(std::int64_t));
            std::memcpy(
                alpaka::onHost::data(hostWrites),
                zeroWrites.data(),
                zeroWrites.size() * sizeof(std::uint32_t));
            alpaka::onHost::memcpy(m_queue, m_weights, hostWeights);
            alpaka::onHost::memcpy(m_queue, m_sinkWrites, hostWrites);

            ResultBlock control{};
            control.weights = alpaka::onHost::data(m_weights);
            control.sinkWrites = alpaka::onHost::data(m_sinkWrites);
            control.slotCount = count;
            control.sinkWriteCount = count;
            *alpaka::onHost::data(m_hostControl) = control;
            alpaka::onHost::memcpy(m_queue, m_controlBuffer, m_hostControl);
            alpaka::onHost::wait(m_queue);
        }

        graph::PoolConfig& config()
        {
            return m_config;
        }

        graph::Pool start()
        {
            return graph::startPool(m_device, m_executor, compileParticleGraph(m_chunk), m_config, ParticleBinder{});
        }

        graph::PoolResult waitBounded(graph::Pool& pool, std::chrono::milliseconds timeout)
        {
            auto result = pool.waitFor(timeout);
            if(!result.has_value())
            {
                pool.abort();
                auto second = pool.waitFor(std::chrono::seconds{10});
                FAIL("particle pool did not stop within the deadline");
                return second.has_value() ? *second : graph::PoolResult{};
            }
            return *result;
        }

        void fetch()
        {
            alpaka::onHost::memcpy(m_queue, m_hostWeights, m_weights);
            alpaka::onHost::memcpy(m_queue, m_hostSinkWrites, m_sinkWrites);
            alpaka::onHost::memcpy(m_queue, m_hostControl, m_controlBuffer);
            alpaka::onHost::wait(m_queue);
        }

        std::vector<std::int64_t> weights() const
        {
            auto const* data = alpaka::onHost::data(m_hostWeights);
            return std::vector<std::int64_t>(data, data + m_count);
        }

        std::vector<std::uint32_t> sinkWrites() const
        {
            auto const* data = alpaka::onHost::data(m_hostSinkWrites);
            return std::vector<std::uint32_t>(data, data + m_count);
        }

        ResultBlock control() const
        {
            return *alpaka::onHost::data(m_hostControl);
        }

    private:
        T_Device m_device;
        T_Executor m_executor;
        std::uint32_t m_chunk;
        std::uint32_t m_count;
        graph::PoolConfig m_config{};
        decltype(alpaka::onHost::alloc<ResultBlock>(std::declval<T_Device>(), 1u)) m_controlBuffer;
        decltype(alpaka::onHost::allocHost<ResultBlock>(1u)) m_hostControl;
        decltype(alpaka::onHost::alloc<std::int64_t>(std::declval<T_Device>(), 1u)) m_weights;
        decltype(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device>(), 1u)) m_sinkWrites;
        decltype(alpaka::onHost::allocHost<std::int64_t>(1u)) m_hostWeights;
        decltype(alpaka::onHost::allocHost<std::uint32_t>(1u)) m_hostSinkWrites;
        decltype(std::declval<T_Device&>().makeQueue()) m_queue;
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

    /** Independent scalar reference for the per-id final weight. */
    std::vector<std::int64_t> referenceWeights(std::uint64_t count, std::uint64_t seed)
    {
        std::vector<std::int64_t> result(count, std::int64_t{-1});
        std::vector<std::uint32_t> property(count);
        std::vector<Weight> weight(count, 0u);
        std::vector<std::uint64_t> visit(count, 0u);
        std::vector<std::uint8_t> destroyed(count, 0u);
        for(std::uint64_t id = 0u; id < count; ++id)
            property[id] = rng_oracle::draw32(seed, id, 0u, rng_oracle::Purpose::Property);

        std::vector<std::uint64_t> queue(count);
        for(std::uint64_t id = 0u; id < count; ++id)
            queue[id] = id;
        for(std::size_t head = 0u; head < queue.size(); ++head)
        {
            auto const id = queue[head];
            bool const bin1 = rng_oracle::draw32(seed, id, visit[id], rng_oracle::Purpose::Filter) >= property[id];
            auto const purpose = bin1 ? rng_oracle::Purpose::DecisionB : rng_oracle::Purpose::DecisionA;
            bool const survive = rng_oracle::draw32(seed, id, visit[id], purpose) >= rng_oracle::kHalfScale;
            if(survive)
            {
                weight[id] += bin1 ? 1u : 1000u;
                visit[id] += 1u;
                queue.push_back(id);
            }
            else
            {
                REQUIRE_FALSE(destroyed[id] != 0u);
                destroyed[id] = 1u;
                result[id] = weight[id];
            }
        }
        return result;
    }

    std::map<std::int64_t, std::int64_t> histogramOf(std::vector<std::int64_t> const& weights)
    {
        std::map<std::int64_t, std::int64_t> histogram;
        for(auto const weight : weights)
            if(weight >= 0)
                ++histogram[weight];
        return histogram;
    }

    std::uint64_t fnv1aOf(std::vector<std::int64_t> const& weights)
    {
        std::uint64_t hash = 1'469'598'103'934'665'603ull;
        for(auto const weight : weights)
        {
            if(weight < 0)
                continue;
            hash ^= static_cast<std::uint64_t>(weight);
            hash *= 1'099'511'628'211ull;
        }
        return hash;
    }

    /** The committed N=4096 seed=42 fixture (tests/reference/test-vectors.md). */
    std::map<std::int64_t, std::int64_t> const& committedHistogram()
    {
        static std::map<std::int64_t, std::int64_t> const histogram{
            {0, 2081}, {1, 480},  {2, 170},   {3, 72},     {4, 20},     {5, 12},    {6, 7},
            {7, 3},    {9, 1},    {10, 1},    {1000, 478}, {1001, 215}, {1002, 78}, {1003, 25},
            {1004, 8}, {1005, 4}, {1006, 2},  {2000, 136}, {2001, 71},  {2002, 30}, {2003, 10},
            {2004, 6}, {2005, 2}, {3000, 62}, {3001, 28},  {3002, 12},  {3003, 3},  {3004, 4},
            {3005, 1}, {3006, 1}, {4000, 24}, {4001, 11},  {4002, 9},   {4004, 1},  {5000, 7},
            {5001, 4}, {5002, 4}, {6000, 6},  {6001, 3},   {6002, 1},   {7000, 2},  {7001, 1}};
        return histogram;
    }

    template<typename T_Device, typename T_Executor>
    void runParticleCase(T_Device device, T_Executor executor, char const* backend)
    {
        std::uint32_t const chunk = 128u;
        std::uint32_t const count = 4096u;
        std::uint64_t const seed = 42u;
        ParticlePoolHarness<T_Device, T_Executor> harness(device, executor, chunk, count);
        auto pool = harness.start();

        std::array<graph::RequestSpec, 1u> const requests{graph::RequestSpec{0u, count, seed}};
        REQUIRE(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
        CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
        auto const result = harness.waitBounded(pool, std::chrono::seconds{300});
        harness.fetch();

        CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
        CHECK(result.errorWord == 0u);
        CHECK(result.counters.outstanding == 0u);
        CHECK(result.counters.pendingSources == 0u);
        CHECK(result.counters.partialBuffers == 0u);

        auto const control = harness.control();
        CHECK(control.errorWord == 0u);
        auto const weights = harness.weights();
        auto const sinkWrites = harness.sinkWrites();
        auto const expected = referenceWeights(count, seed);

        std::uint64_t mismatches = 0u;
        std::uint64_t duplicates = 0u;
        std::uint64_t missing = 0u;
        for(std::uint32_t id = 0u; id < count; ++id)
        {
            if(sinkWrites[id] == 0u)
                ++missing;
            else if(sinkWrites[id] != 1u)
                ++duplicates;
            if(weights[id] != expected[id])
            {
                if(mismatches < 8u)
                    std::printf(
                        "[particle-pool] backend=%s id=%u weight=%lld expected=%lld sinkWrites=%u\n",
                        backend,
                        id,
                        static_cast<long long>(weights[id]),
                        static_cast<long long>(expected[id]),
                        sinkWrites[id]);
                ++mismatches;
            }
        }
        CHECK(missing == 0u);
        CHECK(duplicates == 0u);
        CHECK(mismatches == 0u);

        auto const histogram = histogramOf(weights);
        CHECK(histogram == committedHistogram());
        CHECK(fnv1aOf(weights) == 0xFBBC'1B50'42F9'5AABull);

        std::printf(
            "[particle-pool] backend=%s chunk=%u n=%u seed=%llu mismatches=%llu missing=%llu "
            "duplicates=%llu histogramRows=%zu drainRounds=%llu slotsReturned=%llu\n",
            backend,
            chunk,
            count,
            static_cast<unsigned long long>(seed),
            static_cast<unsigned long long>(mismatches),
            static_cast<unsigned long long>(missing),
            static_cast<unsigned long long>(duplicates),
            histogram.size(),
            static_cast<unsigned long long>(result.counters.drainRounds),
            static_cast<unsigned long long>(result.counters.slotsReturned));
        std::fflush(stdout);
    }
} // namespace

TEST_CASE("particle feedback pool matches the committed fixture on CPU", "[graph][particle][pool][cpu]")
{
    auto cpu = makeCpuHandles();
    runParticleCase(cpu.device, cpu.executor, "cpu");
}

#if GRAPH_HAS_CUDA
TEST_CASE("particle feedback pool matches the committed fixture on CUDA", "[graph][particle][pool][cuda]")
{
    auto cuda = makeCudaHandles();
    runParticleCase(cuda.device, cuda.executor, "cuda");
}
#endif
