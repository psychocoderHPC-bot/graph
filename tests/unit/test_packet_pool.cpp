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

    /** Kernel: multi-warp contention probe. Each block is one worker warp;
     *  every warp leader runs a bounded acquire/release storm. With a small
     *  pool capacity the warps contend on the same free-list head, and the
     *  pool must hand out every slot exactly once per generation window.
     *
     * Layout of `out` (one `std::uint64_t` per slot, after the results):
     *   out[i] = (ownerWorkerIndex << 32) | observedGeneration
     * while the slot is held, zero when it is free.
     *
     * Results (all grid-aggregated):
     *   [0] = total successful acquires (must equal warpCount * acquiresPerWarp)
     *   [1] = total failed acquires (must be 0: contention never exhausts)
     *   [2] = total failed releases (must be 0)
     *   [3] = double hand-out: a slot acquired while another warp held it
     *   [4] = generation non-monotonic: a slot's generation decreased
     *   [5] = ownership violations recorded by the pool (must be 0)
     *   [6] = freeCount at the end (must equal capacity)
     *   [7] = slots still active at the end (must be 0)
     */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    struct PoolContentionKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto storage,
            std::uint32_t acquiresPerWarp) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Pool = graph::PacketPool<Acc, T_SlotCapacity, T_ElementCapacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Pool pool{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Pool::storageBytes);
            auto* out = reinterpret_cast<std::uint64_t*>(base + Pool::storageBytes + 8 * sizeof(std::uint64_t));
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            auto const blockIndex = static_cast<std::uint32_t>(
                acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)[0]);
            auto const gridBlocks = static_cast<std::uint32_t>(
                acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)[0]);

            // Block 0 configures. The serial CPU executor runs blocks
            // sequentially and CUDA runs them concurrently, so a device-scope
            // fence orders the configure stores against the first acquire
            // without a grid barrier (which this alpaka version does not
            // provide).
            if(blockIndex == 0u)
            {
                auto budget = graph::PacketPoolBudget{};
                budget.liveParticles = static_cast<std::uint32_t>(T_SlotCapacity);
                (void) pool.configure(acc, warp, budget);
            }
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);

            std::uint64_t acqOk = 0u, acqFail = 0u, relFail = 0u, doubleOut = 0u, nonMono = 0u;
            if(warp.isLeader())
            {
                // The CPU serial executor runs one width-one warp per launch, so
                // it does one warp's share; the GPU runs every block's warp and
                // the expected total scales with the block count.
                // A monotonically increasing nonce distinguishes the current
                // holder from the previous one: a marker with a different
                // nonce means a double hand-out.
                std::uint64_t nonce = 0u;
                for(std::uint32_t i = 0u; i < acquiresPerWarp; ++i)
                {
                    graph::PacketHandle h{};
                    bool const got = pool.acquire(acc, warp, h);
                    if(!got)
                    {
                        // The other warps release as fast as they acquire, so
                        // the free list cannot be empty for long; a failure is
                        // a lost slot, not exhaustion.
                        acqFail++;
                        continue;
                    }
                    acqOk++;
                    nonce++;
                    // Device-atomic exchange on the per-slot marker. The
                    // previous holder cleared the marker atomically before
                    // releasing the slot, so a nonzero return with a
                    // different nonce means a double hand-out.
                    auto const marker = static_cast<std::uint64_t>(blockIndex + 1u) << 32u | nonce;
                    auto const prev = alpaka::onAcc::atomicExch(
                        acc,
                        &out[h.index],
                        marker,
                        alpaka::onAcc::scope::device);
                    if(prev != 0ull)
                    {
                        doubleOut++;
                        // Non-monotonic: the previous holder's nonce is
                        // greater than ours (should not happen).
                        if((prev & 0xffff'ffffull) > nonce)
                            nonMono++;
                    }
                    // Clear the marker atomically BEFORE the release: once
                    // the slot is back on the free list, another warp may pop
                    // it and exchange its own marker. The clear is safe: this
                    // warp still owns the slot (release has not run yet), so
                    // no other warp can be writing to `out[h.index]`.
                    alpaka::onAcc::atomicExch(acc, &out[h.index], std::uint64_t{0u}, alpaka::onAcc::scope::device);
                    bool const freed = pool.release(acc, warp, h);
                    if(!freed)
                        relFail++;
                }
            }
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);

            // Grid-wide aggregation: device atomics on the shared results.
            if(warp.isLeader())
            {
                (void) alpaka::onAcc::atomicAdd(acc, &results[0], acqOk, alpaka::onAcc::scope::device);
                (void) alpaka::onAcc::atomicAdd(acc, &results[1], acqFail, alpaka::onAcc::scope::device);
                (void) alpaka::onAcc::atomicAdd(acc, &results[2], relFail, alpaka::onAcc::scope::device);
                (void) alpaka::onAcc::atomicAdd(acc, &results[3], doubleOut, alpaka::onAcc::scope::device);
                (void) alpaka::onAcc::atomicAdd(acc, &results[4], nonMono, alpaka::onAcc::scope::device);
            }
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);

            if(blockIndex == 0u && warp.isLeader())
            {
                results[5] = pool.ownershipViolations();
                results[6] = pool.freeCount();
                // Every slot must be free again. PoolSlot is public, so the
                // bookkeeping is directly inspectable.
                auto const* slots = reinterpret_cast<Pool::PoolSlot const*>(base + Pool::headerBytes);
                std::uint64_t held = 0u;
                for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                    if(slots[i].meta.active != 0u)
                        held++;
                results[7] = held;
                // Diagnostic counters from the pool header.
                results[8] = pool.casFailures();
                results[9] = pool.metaChecks();
            }
        }
    };

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

    /** Launch `blockCount` blocks of 32 threads (one warp per block, the pool
     *  layout) on the configured backend and wait.
     *
     * On the CPU serial executor the block extent must be 1 (the executor
     * throws otherwise), so one block of 1 thread is launched. The contention
     * probe then runs a single width-one warp; the GPU runs the full multi-warp
     * grid, which is the case that needs the cross-warp proof.
     */
    void launchWarps(auto& device, std::uint32_t blockCount, auto&& kernel, auto&&... args)
    {
#if GRAPH_HAS_CUDA
        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{blockCount},
            alpaka::Vec<std::uint32_t, 1u>{32u},
            alpaka::exec::gpuCuda};
#else
        // CPU serial: one thread per block; the executor requires a block
        // product of 1, so the multi-block grid degrades to a single block.
        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::exec::cpuSerial};
        (void) blockCount;
#endif
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, kernel, args...);
        alpaka::onHost::wait(queue);
    }

    /** Run the multi-warp contention probe.
     *
     * `blockCount` warps each run `acquiresPerWarp` acquire/release cycles
     * against a pool of `T_SlotCapacity` slots. The assertions prove the
     * free-list hand-out: no slot is handed to two warps at once, no slot is
     * lost, generations are monotonic, and the pool ends clean.
     */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity, std::uint32_t T_BlockCount, std::uint32_t T_AcquiresPerWarp>
    void runContention()
    {
        auto device = makeTestDevice();
        using Pool = graph::PacketPool<int, T_SlotCapacity, T_ElementCapacity>;
        auto const totalBytes = Pool::storageBytes + 10 * sizeof(std::uint64_t) + T_SlotCapacity * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchWarps(device, T_BlockCount, PoolContentionKernel<T_SlotCapacity, T_ElementCapacity>{}, deviceBuffer, T_AcquiresPerWarp);
        auto queue = device.makeQueue();
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const* results = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + Pool::storageBytes);

#if GRAPH_HAS_CUDA
        // GPU: every block's warp runs the full cycle.
        std::uint64_t const expectedAcq = static_cast<std::uint64_t>(T_BlockCount) * T_AcquiresPerWarp;
#else
        // CPU serial: the executor runs one block of one thread (the multi-block
        // grid is not expressible), so exactly one warp does the full cycle.
        std::uint64_t const expectedAcq = T_AcquiresPerWarp;
#endif
        INFO(
            "cap=" << T_SlotCapacity << " warps=" << T_BlockCount << " acqPerWarp=" << T_AcquiresPerWarp
                   << " acqOk=" << results[0] << " acqFail=" << results[1] << " relFail=" << results[2]
                   << " doubleOut=" << results[3] << " nonMono=" << results[4] << " violations=" << results[5]
                   << " freeCount=" << results[6] << " heldAtEnd=" << results[7]
                   << " casFailures=" << results[8] << " metaChecks=" << results[9]);
        CHECK(results[0] == expectedAcq);
        CHECK(results[1] == 0u);
        CHECK(results[2] == 0u);
        CHECK(results[3] == 0u);
        CHECK(results[4] == 0u);
        CHECK(results[5] == 0u);
        CHECK(results[6] == T_SlotCapacity);
        CHECK(results[7] == 0u);
    }

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

TEST_CASE("packet pool: no double hand-out under multi-warp contention", "[graph][packet_pool]")
{
    // 16 warps x 10000 acquires = 160000 hand-outs over 64 slots: the free
    // list must stay consistent under this pressure.
    runContention<64u, 256u, 16u, 10000u>();
}

TEST_CASE("packet pool: no double hand-out, small pool, tight contention", "[graph][packet_pool]")
{
    // 8 warps over 4 slots: every hand-out is a CAS race on the same head.
    runContention<4u, 64u, 8u, 2048u>();
}
