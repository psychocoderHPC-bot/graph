// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/ready_ring.hpp>

#include <cstdint>
#include <cstring>

/** Ready-ring unit tests.
 *
 * The tests run on the configured accelerator (CPU serial or CUDA). The ring
 * is exercised through alpaka kernels so the same code path is exercised on
 * both backends.
 *
 * Every enqueue/dequeue call is wrapped in a `WarpBarrier` (the real barrier,
 * not a warp vote) so the producer lanes' stores are ordered before the
 * leader publishes and the leader's acquisition is ordered before the lanes
 * read. Every kernel loop carries a finite attempt budget so a deadlock fails
 * the test (via a deadline counter) instead of hanging CI.
 */
namespace
{
    /** Mix a 64-bit value (splitmix64 finalizer). */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t mix64(std::uint64_t x) noexcept
    {
        x ^= x >> 33u;
        x *= 0xff51'afd7'ed55'8ccdull;
        x ^= x >> 33u;
        x *= 0xc4ce'b9fe'1a85'ec53ull;
        x ^= x >> 33u;
        return x;
    }

    /** Build a ring entry for a given push sequence. */
    ALPAKA_FN_HOST_ACC constexpr graph::ReadyEntry makeEntry(std::uint64_t seq) noexcept
    {
        graph::ReadyEntry e{};
        e.operation = graph::OperationId{static_cast<graph::OperationId::ValueType>(seq % 32u)};
        e.kind = graph::JobKind::OPERATION;
        e.packet.index = static_cast<std::uint32_t>(seq % 16u);
        e.packet.generation = static_cast<std::uint32_t>(seq % 7u);
        e.routing.inputPort = 0u;
        e.routing.outputPort = static_cast<std::uint32_t>(seq % 4u);
        e.routing.outputPortCount = 1u;
        e.request.firstId = seq;
        e.request.count = 0u;
        e.request.seed = mix64(seq * 131u);
        e.sequenceNumber = seq;
        return e;
    }

    constexpr std::uint32_t stAccepted = 0u;
    constexpr std::uint32_t stFull = 1u;
    constexpr std::uint32_t stEmpty = 2u;
    constexpr std::uint32_t stContended = 3u;

    /** Launch one block of 32 threads (one warp per block, the bring-up
     *  layout) on the configured backend. */
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
        // thread block product != 1). The "one warp" is a single width-1 lane,
        // which is exactly the host fallback the ring is tested against.
        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::exec::cpuSerial};
#endif
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, kernel, args...);
        alpaka::onHost::wait(queue);
    }

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

    /** Kernel: one warp pushes then pops `itemCount` entries through a ring of
     *  the given compile-time capacity, wrapped in real WarpBarriers.
     *
     * Results:
     *   [0] = enqueues accepted
     *   [1] = enqueues that returned Full
     *   [2] = enqueues that returned Contended
     *   [3] = dequeues accepted
     *   [4] = dequeues that returned Empty
     *   [5] = dequeues that returned Contended
     *   [6] = final in-flight count (must be 0)
     *   [7] = final violations (must be 0)
     */
    template<std::size_t T_Capacity>
    struct RingSingleWarpKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto storage,
            std::uint64_t itemCount) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Ring = graph::ReadyRing<Acc, T_Capacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Ring ring{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Ring::storageBytes);
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            ring.initialize(acc, warp);
            barrier.barrier(acc, blockThreads, warp);

            std::uint64_t enqOk = 0u, enqFull = 0u, enqCont = 0u;
            std::uint64_t deqOk = 0u, deqEmpty = 0u, deqCont = 0u;

            for(std::uint64_t it = 0u; it < itemCount; ++it)
            {
                auto entry = makeEntry(it);
                // Producer: lanes' entry is ready (built per-lane uniformly),
                // barrier, then leader publishes.
                barrier.barrier(acc, blockThreads, warp);
                auto const s1 = ring.enqueue(acc, warp, entry);

                graph::ReadyEntry out{};
                auto const s2 = ring.dequeue(acc, warp, out);
                // Consumer: leader acquired, barrier, then lanes may read `out`.
                barrier.barrier(acc, blockThreads, warp);

                if(warp.isLeader())
                {
                    auto const v1 = static_cast<std::uint32_t>(s1);
                    if(v1 == stAccepted) enqOk++;
                    else if(v1 == stFull) enqFull++;
                    else enqCont++;

                    auto const v2 = static_cast<std::uint32_t>(s2);
                    if(v2 == stAccepted) deqOk++;
                    else if(v2 == stEmpty) deqEmpty++;
                    else deqCont++;
                }
            }

            barrier.barrier(acc, blockThreads, warp);
            if(warp.isLeader())
            {
                results[0] = enqOk;
                results[1] = enqFull;
                results[2] = enqCont;
                results[3] = deqOk;
                results[4] = deqEmpty;
                results[5] = deqCont;
                results[6] = ring.inFlightCount();
                results[7] = ring.ownershipViolations();
                results[8] = barrier.refused() ? 1u : 0u;
            }
        }
    };

    /** Host helper: run the single-warp kernel and check the results. */
    template<std::size_t T_Capacity>
    void runSingleWarp(std::uint64_t itemCount)
    {
        auto device = makeTestDevice();
        using PlaceholderRing = graph::ReadyRing<int, T_Capacity>;
        auto const totalBytes
            = PlaceholderRing::storageBytes + 16 * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchOneWarp(device, RingSingleWarpKernel<T_Capacity>{}, deviceBuffer, itemCount);
        alpaka::onHost::memcpy(
            device.makeQueue(),
            hostBuffer,
            deviceBuffer);
        alpaka::onHost::wait(device.makeQueue());

        auto const* results = reinterpret_cast<std::uint64_t const*>(
            hostBuffer.data() + PlaceholderRing::storageBytes);

        INFO("cap=" << T_Capacity << " items=" << itemCount << " enqOk=" << results[0] << " enqFull=" << results[1]
                    << " enqCont=" << results[2] << " deqOk=" << results[3] << " deqEmpty=" << results[4]
                    << " deqCont=" << results[5] << " inFlight=" << results[6] << " violations=" << results[7]
                    << " barrierRefused=" << results[8]);
        CHECK(results[8] == 0u); // one warp per block: the barrier must not refuse
        CHECK(results[0] == itemCount);
        CHECK(results[3] == itemCount);
        CHECK(results[6] == 0u);
        CHECK(results[7] == 0u);
    }

    /** Kernel: fill the ring to capacity, confirm Full, then drain and confirm
     *  Empty. Also records the empty-ring dequeue and full-ring enqueue.
     *
     * Results:
     *   [0] = enqueues accepted while filling (must equal capacity)
     *   [1] = Full count seen (must be >= 1)
     *   [2] = dequeues accepted while draining (must equal capacity)
     *   [3] = Empty count seen (must be >= 1)
     *   [4] = final in-flight (must be 0)
     */
    template<std::size_t T_Capacity>
    struct RingEmptyFullKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto storage,
            std::uint64_t itemCount) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Ring = graph::ReadyRing<Acc, T_Capacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Ring ring{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Ring::storageBytes);
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            ring.initialize(acc, warp);
            barrier.barrier(acc, blockThreads, warp);

            std::uint64_t enqOk = 0u, enqFull = 0u;
            std::uint64_t deqOk = 0u, deqEmpty = 0u;

            // Empty-ring dequeue must return Empty.
            {
                graph::ReadyEntry out{};
                auto const s = ring.dequeue(acc, warp, out);
                barrier.barrier(acc, blockThreads, warp);
                if(warp.isLeader() && static_cast<std::uint32_t>(s) == stEmpty)
                    deqEmpty++;
            }

            // itemCount = capacity + 1: the first `capacity` enqueues are
            // accepted, the (capacity+1)th must return Full.
            for(std::uint64_t it = 0u; it < itemCount; ++it)
            {
                barrier.barrier(acc, blockThreads, warp);
                auto const s = ring.enqueue(acc, warp, makeEntry(it));
                if(warp.isLeader())
                {
                    auto const v = static_cast<std::uint32_t>(s);
                    if(v == stAccepted) enqOk++;
                    else if(v == stFull) enqFull++;
                }
            }

            // Drain `capacity` entries; the (capacity+1)th must return Empty.
            for(std::uint64_t it = 0u; it < itemCount; ++it)
            {
                graph::ReadyEntry out{};
                auto const s = ring.dequeue(acc, warp, out);
                barrier.barrier(acc, blockThreads, warp);
                if(warp.isLeader())
                {
                    auto const v = static_cast<std::uint32_t>(s);
                    if(v == stAccepted) deqOk++;
                    else if(v == stEmpty) deqEmpty++;
                }
            }

            barrier.barrier(acc, blockThreads, warp);
            if(warp.isLeader())
            {
                results[0] = enqOk;
                results[1] = enqFull;
                results[2] = deqOk;
                results[3] = deqEmpty;
                results[4] = ring.inFlightCount();
            }
        }
    };

    template<std::size_t T_Capacity>
    void runEmptyFull()
    {
        auto device = makeTestDevice();
        using PlaceholderRing = graph::ReadyRing<int, T_Capacity>;
        auto const totalBytes = PlaceholderRing::storageBytes + 8 * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchOneWarp(device, RingEmptyFullKernel<T_Capacity>{}, deviceBuffer, T_Capacity + 1u);
        auto queue = device.makeQueue();
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const* results = reinterpret_cast<std::uint64_t const*>(
            hostBuffer.data() + PlaceholderRing::storageBytes);

        INFO("cap=" << T_Capacity << " enqOk=" << results[0] << " enqFull=" << results[1] << " deqOk=" << results[2]
                    << " deqEmpty=" << results[3] << " inFlight=" << results[4]);
        CHECK(results[0] == T_Capacity);
        CHECK(results[1] >= 1u);
        CHECK(results[2] == T_Capacity);
        CHECK(results[3] >= 1u);
        CHECK(results[4] == 0u);
    }

    /** Kernel: exercise wrap-safe arithmetic by pushing `itemCount` entries,
     *  then reading the counters and the in-flight delta. With a small
     *  capacity and a large itemCount the push/claim counters lap around the
     *  slot mask many times; the ring must stay consistent.
     *
     * Results:
     *   [0] = enqueues accepted
     *   [1] = dequeues accepted
     *   [2] = final pushCounter
     *   [3] = final claimCounter
     *   [4] = final in-flight (must be 0)
     *   [5] = final violations (must be 0)
     */
    template<std::size_t T_Capacity>
    struct RingWrapKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto storage,
            std::uint64_t itemCount) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Ring = graph::ReadyRing<Acc, T_Capacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Ring ring{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Ring::storageBytes);
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            ring.initialize(acc, warp);
            barrier.barrier(acc, blockThreads, warp);

            std::uint64_t enqOk = 0u, deqOk = 0u;
            for(std::uint64_t it = 0u; it < itemCount; ++it)
            {
                barrier.barrier(acc, blockThreads, warp);
                auto const s1 = ring.enqueue(acc, warp, makeEntry(it));
                graph::ReadyEntry out{};
                auto const s2 = ring.dequeue(acc, warp, out);
                barrier.barrier(acc, blockThreads, warp);
                if(warp.isLeader())
                {
                    if(static_cast<std::uint32_t>(s1) == stAccepted) enqOk++;
                    if(static_cast<std::uint32_t>(s2) == stAccepted) deqOk++;
                }
            }

            barrier.barrier(acc, blockThreads, warp);
            if(warp.isLeader())
            {
                results[0] = enqOk;
                results[1] = deqOk;
                results[2] = ring.nextPushSequence();
                results[3] = ring.claimCounter();
                results[4] = ring.inFlightCount();
                results[5] = ring.ownershipViolations();
            }
        }
    };

    template<std::size_t T_Capacity>
    void runWrap(std::uint64_t itemCount)
    {
        auto device = makeTestDevice();
        using PlaceholderRing = graph::ReadyRing<int, T_Capacity>;
        auto const totalBytes = PlaceholderRing::storageBytes + 8 * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchOneWarp(device, RingWrapKernel<T_Capacity>{}, deviceBuffer, itemCount);
        auto queue = device.makeQueue();
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const* results = reinterpret_cast<std::uint64_t const*>(
            hostBuffer.data() + PlaceholderRing::storageBytes);

        INFO("cap=" << T_Capacity << " items=" << itemCount << " enqOk=" << results[0] << " deqOk=" << results[1]
                    << " push=" << results[2] << " claim=" << results[3] << " inFlight=" << results[4]
                    << " violations=" << results[5]);
        CHECK(results[0] == itemCount);
        CHECK(results[1] == itemCount);
        CHECK(results[2] == itemCount);
        CHECK(results[3] == itemCount);
        CHECK(results[4] == 0u);
        CHECK(results[5] == 0u);
    }

    /** Kernel: exact-once claim over many items. One warp pushes all entries,
     *  then claims them all; each entry is claimed exactly once and its
     *  payload is verified. Uses a small capacity so the ring laps many times.
     *
     * Results:
     *   [0] = enqueues accepted (must equal itemCount)
     *   [1] = dequeues accepted (must equal itemCount)
     *   [2] = entries claimed out of order or with a corrupted payload
     *   [3] = duplicate claims (same sequenceNumber seen twice)
     *   [4] = missing claims (an expected sequenceNumber never seen)
     *   [5] = final violations
     */
    template<std::size_t T_Capacity>
    struct RingExactOnceKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto storage,
            std::uint64_t itemCount) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Ring = graph::ReadyRing<Acc, T_Capacity>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Ring ring{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Ring::storageBytes);
            auto* seen = reinterpret_cast<std::uint8_t*>(base + Ring::storageBytes + 8 * sizeof(std::uint64_t));
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;

            ring.initialize(acc, warp);
            barrier.barrier(acc, blockThreads, warp);

            std::uint64_t enqOk = 0u, deqOk = 0u;

            // Fill the ring up to capacity, then claim one at a time as we go,
            // so the ring is always near its capacity boundary (wraps a lot).
            for(std::uint64_t it = 0u; it < itemCount; ++it)
            {
                // Push until accepted (bounded backoff: capacity is small and
                // we drain as we go, so this cannot spin unboundedly).
                std::uint64_t guard = 0u;
                auto s1 = ring.enqueue(acc, warp, makeEntry(it));
                while(static_cast<std::uint32_t>(s1) != stAccepted && guard++ < 1u << 20u)
                {
                    s1 = ring.enqueue(acc, warp, makeEntry(it));
                }
                barrier.barrier(acc, blockThreads, warp);
                if(warp.isLeader() && static_cast<std::uint32_t>(s1) == stAccepted)
                    enqOk++;

                // Claim exactly one (bounded).
                graph::ReadyEntry out{};
                std::uint64_t guard2 = 0u;
                auto s2 = ring.dequeue(acc, warp, out);
                while(static_cast<std::uint32_t>(s2) != stAccepted && guard2++ < 1u << 20u)
                {
                    s2 = ring.dequeue(acc, warp, out);
                }
                barrier.barrier(acc, blockThreads, warp);

                if(warp.isLeader() && static_cast<std::uint32_t>(s2) == stAccepted)
                {
                    deqOk++;
                    auto const seq = out.sequenceNumber;
                    // `seq` must be exactly `it` (claim order == push order),
                    // must not have been seen before, and the payload must be
                    // the entry we built for that sequence.
                    if(seq != it)
                        results[2]++;
                    else if(seen[seq] != 0u)
                        results[3]++;
                    else
                    {
                        seen[seq] = 1u;
                        if(out.request.firstId != it || out.request.seed != mix64(it * 131u))
                            results[2]++;
                    }
                }
            }

            barrier.barrier(acc, blockThreads, warp);
            if(warp.isLeader())
            {
                results[0] = enqOk;
                results[1] = deqOk;
                results[5] = ring.ownershipViolations();
                std::uint64_t missing = 0u;
                for(std::uint64_t i = 0u; i < itemCount; ++i)
                    if(seen[i] != 1u)
                        missing++;
                results[4] = missing;
            }
        }
    };

    template<std::size_t T_Capacity>
    void runExactOnce(std::uint64_t itemCount)
    {
        auto device = makeTestDevice();
        using PlaceholderRing = graph::ReadyRing<int, T_Capacity>;
        auto const totalBytes = PlaceholderRing::storageBytes + 8 * sizeof(std::uint64_t) + itemCount;

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        launchOneWarp(device, RingExactOnceKernel<T_Capacity>{}, deviceBuffer, itemCount);
        auto queue = device.makeQueue();
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const* results = reinterpret_cast<std::uint64_t const*>(
            hostBuffer.data() + PlaceholderRing::storageBytes);

        INFO("cap=" << T_Capacity << " items=" << itemCount << " enqOk=" << results[0] << " deqOk=" << results[1]
                    << " payload/order errors=" << results[2] << " duplicates=" << results[3]
                    << " missing=" << results[4] << " violations=" << results[5]);
        CHECK(results[0] == itemCount);
        CHECK(results[1] == itemCount);
        CHECK(results[2] == 0u);
        CHECK(results[3] == 0u);
        CHECK(results[4] == 0u);
        CHECK(results[5] == 0u);
    }
} // namespace

TEST_CASE("ready ring: single warp enqueue/dequeue, tiny capacity 2", "[graph][ready_ring]")
{
    runSingleWarp<2u>(64u);
}

TEST_CASE("ready ring: single warp enqueue/dequeue, tiny capacity 1", "[graph][ready_ring]")
{
    runSingleWarp<1u>(64u);
}

TEST_CASE("ready ring: empty and full, capacity 2", "[graph][ready_ring]")
{
    runEmptyFull<2u>();
}

TEST_CASE("ready ring: empty and full, capacity 1", "[graph][ready_ring]")
{
    runEmptyFull<1u>();
}

TEST_CASE("ready ring: exact-once claim over 10000 items, capacity 8", "[graph][ready_ring]")
{
    runExactOnce<8u>(10000u);
}

TEST_CASE("ready ring: wrap-safe arithmetic, capacity 4, 4096 items", "[graph][ready_ring]")
{
    runWrap<4u>(4096u);
}

TEST_CASE("ready ring: sequence rollover, capacity 2, 2048 items", "[graph][ready_ring]")
{
    runWrap<2u>(2048u);
}
