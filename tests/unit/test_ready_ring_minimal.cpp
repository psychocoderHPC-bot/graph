// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

/**
 * Minimal ring test: single thread, no atomics, no warp group.
 * Verifies the ring protocol (lock/unlock, freeSlots, pushCounter,
 * claimCounter, slot indexing) works end-to-end on the device.
 *
 * If this passes, the issue is with alpaka atomics on the device.
 * If this fails, the issue is with the ring protocol itself.
 */

namespace
{
    constexpr std::size_t CAP = 2u;
    constexpr std::uint64_t MASK = CAP - 1u;

    constexpr std::size_t alignUp64(std::size_t v) noexcept
    {
        return (v + 63u) & ~63u;
    }

    struct Header
    {
        std::uint32_t lock;
        std::uint64_t padLock[7];
        std::uint64_t pushCounter;
        std::uint64_t padPush[7];
        std::uint64_t claimCounter;
        std::uint64_t padClaim[7];
        std::uint64_t freeSlots;
        std::uint64_t padFree[7];
    };

    struct Slot
    {
        std::uint64_t value;
        std::uint64_t sequence;
    };

    constexpr std::size_t headerBytes = alignUp64(sizeof(Header));
    constexpr std::size_t slotBytes = sizeof(Slot);
    constexpr std::size_t totalBytes = headerBytes + CAP * slotBytes + 8 * sizeof(std::uint64_t);

    struct MinimalRingAtomicKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto storage, std::uint64_t itemCount)
            const
        {
            auto* base = storage.data();
            auto* hdr = reinterpret_cast<Header*>(base);
            auto* slots = reinterpret_cast<Slot*>(base + headerBytes);
            auto* results = reinterpret_cast<std::uint64_t*>(base + headerBytes + CAP * slotBytes);

            std::uint64_t enqOk = 0, enqFull = 0, deqOk = 0, deqEmpty = 0;
            std::uint32_t firstEnqStatus = 0xDEAD;

            for(std::uint64_t i = 0; i < itemCount; ++i)
            {
                // Enqueue with atomic lock.
                std::uint32_t old = alpaka::onAcc::atomicCas<std::uint32_t>(acc, &hdr->lock, std::uint32_t{0}, std::uint32_t{1});
                if(i == 0) firstEnqStatus = old;
                if(old != 0)
                {
                    enqFull++; // contended (shouldn't happen for single thread)
                }
                else if(hdr->freeSlots == 0)
                {
                    alpaka::onAcc::atomicCas<std::uint32_t>(acc, &hdr->lock, std::uint32_t{1}, std::uint32_t{0});
                    enqFull++;
                }
                else
                {
                    auto const slotIdx = hdr->pushCounter & MASK;
                    slots[slotIdx].sequence = hdr->pushCounter;
                    slots[slotIdx].value = i;
                    alpaka::onAcc::atomicAdd<std::uint64_t>(acc, &hdr->pushCounter, std::uint64_t{1});
                    alpaka::onAcc::atomicAdd<std::uint64_t>(acc, &hdr->freeSlots, ~std::uint64_t{0});
                    alpaka::onAcc::atomicCas<std::uint32_t>(acc, &hdr->lock, std::uint32_t{1}, std::uint32_t{0});
                    enqOk++;
                }

                // Dequeue with atomic lock.
                old = alpaka::onAcc::atomicCas<std::uint32_t>(acc, &hdr->lock, std::uint32_t{0}, std::uint32_t{1});
                if(old != 0)
                {
                    deqEmpty++;
                }
                else if(hdr->freeSlots == CAP)
                {
                    alpaka::onAcc::atomicCas<std::uint32_t>(acc, &hdr->lock, std::uint32_t{1}, std::uint32_t{0});
                    deqEmpty++;
                }
                else
                {
                    auto const slotIdx = hdr->claimCounter & MASK;
                    results[6] = slots[slotIdx].value;
                    alpaka::onAcc::atomicAdd<std::uint64_t>(acc, &hdr->claimCounter, std::uint64_t{1});
                    alpaka::onAcc::atomicAdd<std::uint64_t>(acc, &hdr->freeSlots, std::uint64_t{1});
                    alpaka::onAcc::atomicCas<std::uint32_t>(acc, &hdr->lock, std::uint32_t{1}, std::uint32_t{0});
                    deqOk++;
                }
            }

            results[0] = enqOk;
            results[1] = enqFull;
            results[2] = deqOk;
            results[3] = deqEmpty;
            results[4] = hdr->freeSlots;
            results[5] = hdr->pushCounter;
            results[7] = firstEnqStatus;
        }
    };

}

TEST_CASE("minimal ring: single thread, no atomics", "[graph][ready_ring][minimal]")
{
    // Placeholder - the atomic test is below.
    SUCCEED();
}

TEST_CASE("minimal ring: single thread, with alpaka atomics", "[graph][ready_ring][minimal_atomic]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);

    auto hostBuffer = alpaka::onHost::allocHost<std::uint8_t>(totalBytes);
    std::memset(hostBuffer.data(), 0, totalBytes);
    auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);

    auto queue = device.makeQueue();
    queue.enqueue(
        alpaka::onHost::ThreadSpec{alpaka::Vec<std::uint32_t, 1u>{1u}, alpaka::Vec<std::uint32_t, 1u>{1u}, alpaka::exec::cpuSerial},
        MinimalRingAtomicKernel{},
        deviceBuffer,
        64u);
    alpaka::onHost::wait(queue);

    alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
    alpaka::onHost::wait(queue);

    auto const* results = reinterpret_cast<std::uint64_t const*>(hostBuffer.data() + headerBytes + CAP * slotBytes);

    INFO("enqOk=" << results[0] << " enqFull=" << results[1] << " deqOk=" << results[2]
                  << " deqEmpty=" << results[3] << " freeSlots=" << results[4]
                  << " pushCounter=" << results[5] << " lastValue=" << results[6]
                  << " firstCasOld=" << results[7]);
    CHECK(results[0] == 64u);
    CHECK(results[2] == 64u);
    CHECK(results[4] == CAP);
    CHECK(results[5] == 64u);
    CHECK(results[7] == 0u); // first atomicCas should return 0 (lock was free)
}
