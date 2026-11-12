// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/packet_pool.hpp>

#include <cstdint>
#include <cstring>

/** Packet-pool unit tests.
 *
 * The tests run on the configured accelerator (CPU serial or CUDA). The pool
 * is exercised through alpaka kernels so the same code path is exercised on
 * both backends.
 *
 * `acquire`/`release` are warp-uniform (they broadcast internally), so every
 * lane of the warp must call them on the same line; only the leader performs
 * the locked work. The loops below are therefore warp-uniform, with the
 * leader doing the side effects (writes, counters) and the result broadcast
 * back to the lanes by the pool itself.
 */
namespace
{
    auto makeTestDevice()
    {
#if GRAPH_HAS_CUDA
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
#else
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
#endif
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        REQUIRE(selector.getDeviceCount() > 0u);
        return selector.makeDevice(0);
    }

    void launchOneWarp(auto& device, auto&& kernel, auto&&... args)
    {
#if GRAPH_HAS_CUDA
        // One warp per block: 32 threads.
        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::Vec<std::uint32_t, 1u>{32u},
            alpaka::exec::gpuCuda};
#else
        // The CPU serial executor requires a 1-thread block (it throws if the
        // thread block product != 1). The "one warp" is a single width-1 lane.
        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::exec::cpuSerial};
#endif
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, kernel, args...);
        alpaka::onHost::wait(queue);
    }

    /** Budget formula is a plain integer expression; check it on the host. */
    TEST_CASE("packet pool: budget formula requiredSlots(P,B,W,R,S)", "[graph][packet_pool]")
    {
        // requiredSlots = P + 2*B + W*R + S
        CHECK(graph::requiredSlots(0u, 0u, 0u, 0u, 0u) == 0u);
        CHECK(graph::requiredSlots(10u, 2u, 4u, 3u, 1u) == 10u + 4u + 12u + 1u);
        CHECK(graph::requiredSlots(5u, 0u, 0u, 0u, 0u) == 5u);
        CHECK(graph::requiredSlots(0u, 3u, 0u, 0u, 0u) == 6u);
        CHECK(graph::requiredSlots(0u, 0u, 8u, 2u, 0u) == 16u);
        CHECK(graph::requiredSlots(0u, 0u, 0u, 0u, 7u) == 7u);

        auto budget = graph::PacketPoolBudget{};
        budget.liveParticles = 10u;
        budget.assemblyStreams = 2u;
        budget.workers = 4u;
        budget.replacementsPerWorker = 3u;
        budget.controlAllowance = 1u;
        CHECK(graph::requiredSlots(budget) == 27u);
    }

    /** Kernel: acquire/release lifecycle, capacity/peak accounting, and a
     *  use-after-release hook. All loops are warp-uniform.
     *
     * Results:
     *   [0] = configure succeeded (1) or failed (0)
     *   [1] = violations after everything (must be exactly 1: the hook)
     *   [2] = acquires that succeeded across the whole run (must equal
     *         2 * capacity: two full acquire/release rounds)
     *   [3] = acquires that failed unexpectedly (must be 0)
     *   [4] = releases that failed unexpectedly (must be 0)
     *   [5] = heldCount peak observed (must equal capacity)
     *   [6] = freeCount at the end (must equal capacity)
     */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    struct PoolLifecycleKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto storage) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Pool = graph::PacketPool<Acc, T_SlotCapacity, T_ElementCapacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Pool pool{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Pool::storageBytes);
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            auto budget = graph::PacketPoolBudget{};
            budget.liveParticles = static_cast<std::uint32_t>(T_SlotCapacity);
            bool const ok = pool.configure(acc, warp, budget);
            barrier.barrier(acc, blockThreads, warp);

            std::uint64_t acqOk = 0u, acqFail = 0u, relFail = 0u;
            std::uint64_t peakHeld = 0u;

            for(std::uint32_t round = 0u; round < 2u; ++round)
            {
                // Acquire every slot (warp-uniform).
                for(std::uint32_t i = 0u; i < T_SlotCapacity; ++i)
                {
                    graph::PacketHandle h{};
                    bool const got = pool.acquire(acc, warp, h);
                    if(warp.isLeader() && got)
                    {
                        acqOk++;
                        auto const held = pool.heldCount();
                        if(held > peakHeld)
                            peakHeld = held;
                        auto* data = pool.slotData(h);
                        for(std::size_t b = 0u; b < T_ElementCapacity; ++b)
                            data[b] = static_cast<std::uint8_t>(i + 1u + round);
                        pool.setElementCount(acc, h, static_cast<std::uint32_t>(T_ElementCapacity));
                    }
                    else if(warp.isLeader() && !got)
                    {
                        acqFail++;
                    }
                }
                barrier.barrier(acc, blockThreads, warp);

                // Release every slot (warp-uniform). For a fresh pool the free
                // list is 0,1,2,... so acquire i yields index i; after a full
                // release the free list is 0,1,2,... again. We release by
                // re-reading each slot's record handle, which holds the
                // (index, generation) the acquire set, so the generation always
                // matches.
                for(std::uint32_t i = 0u; i < T_SlotCapacity; ++i)
                {
                    graph::PacketHandle h{};
                    h.index = i;
                    auto const& rec = pool.slotRecord(h);
                    h = rec.handle;
                    bool const freed = pool.release(acc, warp, h);
                    if(warp.isLeader() && !freed)
                        relFail++;
                }
                barrier.barrier(acc, blockThreads, warp);
            }

            // Use-after-release hook: a stale generation must be rejected and
            // record exactly one violation. Warp-uniform (all lanes call).
            {
                graph::PacketHandle stale{};
                stale.index = 0u;
                stale.generation = 999u; // never issued by this pool
                (void) pool.release(acc, warp, stale);
            }
            barrier.barrier(acc, blockThreads, warp);

            if(warp.isLeader())
            {
                results[0] = ok ? 1u : 0u;
                results[1] = pool.ownershipViolations();
                results[2] = acqOk;
                results[3] = acqFail;
                results[4] = relFail;
                results[5] = peakHeld;
                results[6] = pool.freeCount();
            }
        }
    };

    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    void runLifecycle()
    {
        auto device = makeTestDevice();
        using Pool = graph::PacketPool<int, T_SlotCapacity, T_ElementCapacity>;
        auto const totalBytes = Pool::storageBytes + 8 * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchOneWarp(device, PoolLifecycleKernel<T_SlotCapacity, T_ElementCapacity>{}, deviceBuffer);
        auto queue = device.makeQueue();
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const* results = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + Pool::storageBytes);

        INFO(
            "cap=" << T_SlotCapacity << " elemCap=" << T_ElementCapacity << " configure=" << results[0]
                   << " violations=" << results[1] << " acqOk=" << results[2] << " acqFail=" << results[3]
                   << " relFail=" << results[4] << " peakHeld=" << results[5] << " freeCount=" << results[6]);
        CHECK(results[0] == 1u);
        // Exactly one violation: the deliberate use-after-release hook.
        CHECK(results[1] == 1u);
        CHECK(results[2] == 2u * T_SlotCapacity);
        CHECK(results[3] == 0u);
        CHECK(results[4] == 0u);
        CHECK(results[5] == T_SlotCapacity);
        CHECK(results[6] == T_SlotCapacity);
    }

    /** Kernel: configure with an undersized budget must fail loudly and leave
     *  no slot acquirable, recording a violation.
     *
     * Results:
     *   [0] = configure result (must be 0 for an undersized budget)
     *   [1] = violations (must be >= 1)
     *   [2] = number of acquires that succeeded (must be 0)
     *   [3] = freeCount (must be 0: nothing acquirable)
     */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    struct PoolUnderBudgetKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto storage) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Pool = graph::PacketPool<Acc, T_SlotCapacity, T_ElementCapacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Pool pool{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Pool::storageBytes);
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            auto budget = graph::PacketPoolBudget{};
            // Require one more slot than exists.
            budget.liveParticles = static_cast<std::uint32_t>(T_SlotCapacity) + 1u;
            bool const ok = pool.configure(acc, warp, budget);
            barrier.barrier(acc, blockThreads, warp);

            // Warp-uniform acquire attempt; with nothing free it must fail.
            graph::PacketHandle h{};
            bool const got = pool.acquire(acc, warp, h);
            barrier.barrier(acc, blockThreads, warp);

            std::uint64_t acquired = 0u;
            if(warp.isLeader())
            {
                acquired = got ? 1u : 0u;
                results[0] = ok ? 1u : 0u;
                results[1] = pool.ownershipViolations();
                results[2] = acquired;
                results[3] = pool.freeCount();
            }
        }
    };

    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    void runUnderBudget()
    {
        auto device = makeTestDevice();
        using Pool = graph::PacketPool<int, T_SlotCapacity, T_ElementCapacity>;
        auto const totalBytes = Pool::storageBytes + 8 * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchOneWarp(device, PoolUnderBudgetKernel<T_SlotCapacity, T_ElementCapacity>{}, deviceBuffer);
        auto queue = device.makeQueue();
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const* results = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + Pool::storageBytes);

        INFO(
            "cap=" << T_SlotCapacity << " configure=" << results[0] << " violations=" << results[1]
                   << " acquired=" << results[2] << " freeCount=" << results[3]);
        CHECK(results[0] == 0u);
        CHECK(results[1] >= 1u);
        CHECK(results[2] == 0u);
        CHECK(results[3] == 0u);
    }
} // namespace

TEST_CASE("packet pool: lifecycle, capacity accounting and use-after-release, capacity 8", "[graph][packet_pool]")
{
    runLifecycle<8u, 256u>();
}

TEST_CASE("packet pool: lifecycle, capacity 1", "[graph][packet_pool]")
{
    runLifecycle<1u, 64u>();
}

TEST_CASE("packet pool: undersized budget fails startup loudly", "[graph][packet_pool]")
{
    runUnderBudget<8u, 64u>();
}
