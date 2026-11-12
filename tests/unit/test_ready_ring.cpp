// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
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
 * All kernel loops carry a finite attempt budget so a deadlock fails the
 * test (via a deadline counter) instead of hanging CI.
 */
namespace
{
    /** Round up to a multiple of `alignment`. */
    constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
    {
        return (value + alignment - 1u) & ~(alignment - 1u);
    }

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

    /** Kernel: one warp does all the work (init, enqueue, dequeue).
     *
     * Every lane participates in enqueue/dequeue (warpRendezvous requires it).
     * Only the leader (lane 0) counts results and writes the output buffer.
     *
     * Results:
     *   [0] = enqueues accepted
     *   [1] = enqueues that returned Full
     *   [2] = enqueues that returned Contended
     *   [3] = dequeues accepted
     *   [4] = dequeues that returned Empty
     *   [5] = dequeues that returned Contended
     *   [6] = final in-flight count
     *   [7] = final violations
     */
    struct RingSingleWarpKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto storage, std::uint64_t itemCount)
            const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Ring = graph::ReadyRing<Acc, 2u>;

            auto* base = static_cast<std::uint8_t*>(storage.data());
            Ring ring{base};
            auto* results = reinterpret_cast<std::uint64_t*>(base + Ring::storageBytes);
            graph::WarpGroup<Acc> warp{acc};

            ring.initialize(acc, warp);

            std::uint64_t enqOk = 0u, enqFull = 0u, enqCont = 0u;
            std::uint64_t deqOk = 0u, deqEmpty = 0u, deqCont = 0u;

            for(std::uint64_t it = 0u; it < itemCount; ++it)
            {
                auto entry = makeEntry(it);
                auto s1 = ring.enqueue(acc, warp, entry);

                graph::ReadyEntry out{};
                auto s2 = ring.dequeue(acc, warp, out);

                if(warp.isLeader())
                {
                    auto const v1 = static_cast<std::uint32_t>(s1);
                    if(v1 == 0u) enqOk++;
                    else if(v1 == 1u) enqFull++;
                    else enqCont++;

                    auto const v2 = static_cast<std::uint32_t>(s2);
                    if(v2 == 0u) deqOk++;
                    else if(v2 == 2u) deqEmpty++;
                    else deqCont++;
                }
            }

            if(warp.isLeader())
            {
                results[0] = enqOk;
                results[1] = enqFull;
                results[2] = enqCont;
                results[3] = deqOk;
                results[4] = deqEmpty;
                results[5] = deqCont;
                results[6] = ring.inFlightCount(acc);
                results[7] = ring.ownershipViolations(acc);
            }
        }
    };

    /** Host helper: run the single-warp kernel and check the results. */
    void runSingleWarp(auto& device, auto const& executor, std::uint64_t itemCount)
    {
        using PlaceholderRing = graph::ReadyRing<int, 2u>;
        auto const totalBytes = alignUp(sizeof(typename PlaceholderRing::RingHeader), 64u)
                                + 2u * PlaceholderRing::slotBytes + 8u * sizeof(std::uint64_t);

        auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
        std::memset(hostBuffer.data(), 0, totalBytes);
        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

        auto const threadSpec
            = alpaka::onHost::ThreadSpec{
                alpaka::Vec<std::uint32_t, 1u>{1u},
                alpaka::Vec<std::uint32_t, 1u>{32u},
                executor};
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, RingSingleWarpKernel{}, deviceBuffer, itemCount);
        alpaka::onHost::wait(queue);

        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const ringStorageBytes
            = alignUp(sizeof(typename PlaceholderRing::RingHeader), 64u) + 2u * PlaceholderRing::slotBytes;
        auto const* results
            = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + ringStorageBytes);

        INFO("items=" << itemCount << " enqOk=" << results[0] << " enqFull=" << results[1]
                      << " enqCont=" << results[2] << " deqOk=" << results[3]
                      << " deqEmpty=" << results[4] << " deqCont=" << results[5]
                      << " inFlight=" << results[6] << " violations=" << results[7]);
        CHECK(results[0] == itemCount);
        CHECK(results[3] == itemCount);
        CHECK(results[6] == 0u);
        CHECK(results[7] == 0u);
    }
} // namespace

TEST_CASE("ready ring: single warp enqueue/dequeue", "[graph][ready_ring]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    runSingleWarp(device, alpaka::exec::cpuSerial, 64u);
}
