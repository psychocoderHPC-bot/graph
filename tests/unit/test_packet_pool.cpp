// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
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
 * The pool's storage is one device buffer laid out as the pool's own storage
 * (header + slots, sized by the pool type) followed by a fixed results
 * header.
 */
namespace
{
    /** Round up to a multiple of `alignment`. */
    constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
    {
        return (value + alignment - 1u) & ~(alignment - 1u);
    }

    /** One device buffer: pool storage followed by a fixed results header. */
    template<typename T_Acc, std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    struct PoolTestStorage
    {
        using Pool = graph::PacketPool<T_Acc, T_SlotCapacity, T_ElementCapacity>;
        static constexpr std::size_t poolBytes = Pool::storageBytes;

        std::uint8_t pool[poolBytes]{};
        std::uint64_t results[8]{};
    };

    /** Kernel: acquire/release lifecycle with a budget check.
     *
     * A single warp (the leader lane does the work) configures the pool,
     * acquires all slots, writes a pattern into each, releases them in
     * reverse order, re-acquires and verifies the free list is intact, and
     * finally exercises the use-after-release hook.
     *
     * Results:
     *   [0] = configure succeeded (1) or failed (0)
     *   [1] = violations after the normal lifecycle (must be 0)
     *   [2] = violations after the use-after-release attempt (must be 1)
     *   [3] = slots successfully re-acquired after full release (must equal
     *         capacity)
     *   [4] = content errors on re-acquire (must be 0)
     *   [5] = isValid mismatches during the lifecycle (must be 0)
     */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    struct PoolLifecycleKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto storage) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Pool = graph::PacketPool<Acc, T_SlotCapacity, T_ElementCapacity>;
            using Storage = PoolTestStorage<Acc, T_SlotCapacity, T_ElementCapacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Pool pool{base};

            auto* results = reinterpret_cast<std::uint64_t*>(base + Storage::poolBytes);

            graph::WarpGroup<Acc> warp{acc};

            if(warp.isLeader())
            {
                for(int i = 0; i < 6; ++i)
                    results[i] = 0u;
            }
            (void)warp.all(true);

            // Configure with a budget that exactly fits the capacity.
            graph::PacketPoolBudget budget{};
            budget.liveParticles = static_cast<std::uint32_t>(T_SlotCapacity);
            budget.assemblyStreams = 0u;
            budget.workers = 0u;
            budget.replacementsPerWorker = 0u;
            budget.controlAllowance = 0u;
            auto const configured = pool.configure(acc, warp, budget);

            if(warp.isLeader())
            {
                results[0] = configured ? 1u : 0u;
            }

            if(configured)
            {
                // Phase 1: acquire all slots, write a pattern, verify isValid.
                graph::PacketHandle handles[T_SlotCapacity]{};
                bool allAcquired = true;
                for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                {
                    if(!pool.acquire(acc, warp, handles[i]))
                    {
                        allAcquired = false;
                        break;
                    }
                    // Write a pattern: byte j of slot i gets i*256 + j.
                    auto* data = pool.slotData(handles[i]);
                    for(std::size_t j = 0u; j < T_ElementCapacity; ++j)
                    {
                        data[j] = static_cast<std::uint8_t>((i * 256u + j) & 0xffu);
                    }
                    // Verify the handle is valid.
                    if(!pool.isValid(acc, warp, handles[i]))
                    {
                        if(warp.isLeader())
                        {
                            alpaka::onAcc::atomicAdd(acc, &results[5], std::uint64_t{1u}, alpaka::onAcc::scope::device);
                        }
                    }
                }
                if(warp.isLeader() && !allAcquired)
                {
                    results[5] += 1u;
                }

                // Phase 2: release in reverse order.
                for(std::size_t i = T_SlotCapacity; i-- > 0u;)
                {
                    pool.release(acc, warp, handles[i]);
                }

                // The normal lifecycle must produce no violations.
                if(warp.isLeader())
                {
                    results[1] = pool.ownershipViolations(acc);
                }

                // Phase 3: re-acquire all slots and verify the free list is
                // intact (every slot is acquirable again).
                std::uint64_t reAcquired = 0u;
                std::uint64_t contentErrors = 0u;
                for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                {
                    graph::PacketHandle h{};
                    if(pool.acquire(acc, warp, h))
                    {
                        reAcquired += 1u;
                        // The slot was released, so its content is whatever
                        // the previous owner left (the pattern from phase 1).
                        // Verify the pattern is still there (the pool does
                        // not zero the data on release).
                        auto* data = pool.slotData(h);
                        for(std::size_t j = 0u; j < T_ElementCapacity; ++j)
                        {
                            auto const expected = static_cast<std::uint8_t>((i * 256u + j) & 0xffu);
                            if(data[j] != expected)
                            {
                                contentErrors += 1u;
                                break;
                            }
                        }
                        pool.release(acc, warp, h);
                    }
                }
                if(warp.isLeader())
                {
                    results[3] = reAcquired;
                    results[4] = contentErrors;
                }

                // Phase 4: use-after-release hook. Acquire one slot, release
                // it, then attempt to release it again (double-release) and
                // attempt to use the stale handle (isValid must be false).
                graph::PacketHandle stale{};
                if(pool.acquire(acc, warp, stale))
                {
                    pool.release(acc, warp, stale);
                    // Double-release: must be detected.
                    pool.release(acc, warp, stale);
                    // Use-after-release: isValid must be false.
                    if(pool.isValid(acc, warp, stale))
                    {
                        if(warp.isLeader())
                        {
                            alpaka::onAcc::atomicAdd(acc, &results[5], std::uint64_t{1u}, alpaka::onAcc::scope::device);
                        }
                    }
                }
                if(warp.isLeader())
                {
                    // The double-release records exactly one violation; the
                    // normal lifecycle recorded zero.
                    results[2] = pool.ownershipViolations(acc) - results[1];
                }
            }
        }
    };

    /** Host helper: run the lifecycle kernel and check the results. */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    void runLifecycle(auto& device, auto const& executor)
    {
        using PlaceholderPool = graph::PacketPool<int, T_SlotCapacity, T_ElementCapacity>;
        auto const totalBytes = alignUp(sizeof(typename PlaceholderPool::PoolHeader), 64u)
                                + T_SlotCapacity * PlaceholderPool::slotBytes + 6u * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        auto const threadSpec
            = alpaka::onHost::ThreadSpec{
                alpaka::Vec<std::uint32_t, 1u>{1u},
                alpaka::Vec<std::uint32_t, 1u>{32u},
                executor};
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, PoolLifecycleKernel<T_SlotCapacity, T_ElementCapacity>{}, deviceBuffer);
        alpaka::onHost::wait(queue);

        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const poolStorageBytes
            = alignUp(sizeof(typename PlaceholderPool::PoolHeader), 64u)
              + T_SlotCapacity * PlaceholderPool::slotBytes;
        auto const* results = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + poolStorageBytes);

        INFO("slots=" << T_SlotCapacity << " elementCapacity=" << T_ElementCapacity);
        CHECK(results[0] == 1u);
        CHECK(results[1] == 0u);
        CHECK(results[2] == 1u);
        CHECK(results[3] == T_SlotCapacity);
        CHECK(results[4] == 0u);
        CHECK(results[5] == 0u);
    }

    /** Kernel: budget check. A pool configured with a budget that exceeds its
     *  capacity must fail configure() and record a violation. */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    struct PoolBudgetKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto storage) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Pool = graph::PacketPool<Acc, T_SlotCapacity, T_ElementCapacity>;
            using Storage = PoolTestStorage<Acc, T_SlotCapacity, T_ElementCapacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Pool pool{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Storage::poolBytes);
            graph::WarpGroup<Acc> warp{acc};

            if(warp.isLeader())
            {
                for(int i = 0; i < 4; ++i)
                    results[i] = 0u;
            }
            (void)warp.all(true);

            // Budget that exceeds the capacity: P = T_SlotCapacity + 1.
            graph::PacketPoolBudget budget{};
            budget.liveParticles = static_cast<std::uint32_t>(T_SlotCapacity + 1u);
            budget.assemblyStreams = 0u;
            budget.workers = 0u;
            budget.replacementsPerWorker = 0u;
            budget.controlAllowance = 0u;
            auto const configured = pool.configure(acc, warp, budget);

            if(warp.isLeader())
            {
                results[0] = configured ? 1u : 0u;
                results[1] = pool.ownershipViolations(acc);
                // No slot must be acquirable after a failed configure.
                graph::PacketHandle h{};
                results[2] = pool.acquire(acc, warp, h) ? 1u : 0u;
                results[3] = pool.freeCount(acc);
            }
        }
    };

    /** Host helper: run the budget kernel and check the results. */
    template<std::size_t T_SlotCapacity, std::size_t T_ElementCapacity>
    void runBudget(auto& device, auto const& executor)
    {
        using PlaceholderPool = graph::PacketPool<int, T_SlotCapacity, T_ElementCapacity>;
        auto const totalBytes = alignUp(sizeof(typename PlaceholderPool::PoolHeader), 64u)
                                + T_SlotCapacity * PlaceholderPool::slotBytes + 4u * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        auto const threadSpec
            = alpaka::onHost::ThreadSpec{
                alpaka::Vec<std::uint32_t, 1u>{1u},
                alpaka::Vec<std::uint32_t, 1u>{32u},
                executor};
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, PoolBudgetKernel<T_SlotCapacity, T_ElementCapacity>{}, deviceBuffer);
        alpaka::onHost::wait(queue);

        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const poolStorageBytes
            = alignUp(sizeof(typename PlaceholderPool::PoolHeader), 64u)
              + T_SlotCapacity * PlaceholderPool::slotBytes;
        auto const* results = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + poolStorageBytes);

        INFO("slots=" << T_SlotCapacity << " (budget exceeds capacity)");
        CHECK(results[0] == 0u);
        CHECK(results[1] >= 1u);
        CHECK(results[2] == 0u);
        CHECK(results[3] == 0u);
    }
} // namespace

TEST_CASE("packet pool: acquire/release lifecycle with no violations", "[graph][packet_pool]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    runLifecycle<16u, 64u>(device, alpaka::exec::cpuSerial);
}

TEST_CASE("packet pool: lifecycle at tiny capacity (1 slot)", "[graph][packet_pool]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    runLifecycle<1u, 32u>(device, alpaka::exec::cpuSerial);
}

TEST_CASE("packet pool: use-after-release is detected by the generation hook", "[graph][packet_pool]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    // The lifecycle kernel exercises the double-release and isValid paths;
    // the CHECKs in runLifecycle assert the hook fired exactly once.
    runLifecycle<4u, 32u>(device, alpaka::exec::cpuSerial);
}

TEST_CASE("packet pool: budget check fails loudly when capacity is too small", "[graph][packet_pool]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    runBudget<4u, 32u>(device, alpaka::exec::cpuSerial);
}

TEST_CASE("packet pool: requiredSlots formula is runtime-visible and correct", "[graph][packet_pool]")
{
    // P=10, B=3, W=4, R=2, S=5 -> 10 + 6 + 8 + 5 = 29.
    CHECK(graph::requiredSlots(10u, 3u, 4u, 2u, 5u) == 29u);
    // P=0, B=0, W=0, R=0, S=0 -> 0.
    CHECK(graph::requiredSlots(0u, 0u, 0u, 0u, 0u) == 0u);
    // P=1, B=1, W=1, R=1, S=1 -> 1 + 2 + 1 + 1 = 5.
    CHECK(graph::requiredSlots(1u, 1u, 1u, 1u, 1u) == 5u);

    graph::PacketPoolBudget b{};
    b.liveParticles = 10u;
    b.assemblyStreams = 3u;
    b.workers = 4u;
    b.replacementsPerWorker = 2u;
    b.controlAllowance = 5u;
    CHECK(graph::requiredSlots(b) == 29u);
}

TEST_CASE("packet pool: packet lifetime after dequeue (ring reuse is separate)", "[graph][packet_pool]")
{
    // This test verifies that the pool's slot reuse (release + re-acquire)
    // does not affect the content of a slot that has been released: the
    // content stays until the next owner writes it. The lifecycle kernel
    // already checks this (phase 3 verifies the pattern is intact after a
    // full release/re-acquire cycle). This test re-runs it with a different
    // shape to cover the 2-slot case.
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    runLifecycle<2u, 64u>(device, alpaka::exec::cpuSerial);
}

#if GRAPH_HAS_CUDA
TEST_CASE("packet pool: lifecycle on CUDA", "[graph][packet_pool][cuda]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    runLifecycle<16u, 64u>(device, alpaka::exec::gpuCuda);
    runBudget<4u, 32u>(device, alpaka::exec::gpuCuda);
}
#endif
