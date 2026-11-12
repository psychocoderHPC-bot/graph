// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/ingress_ring.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace
{
    using Descriptor = graph::JobDescriptor;

    template<std::uint32_t T_capacity>
    using Ring = graph::IngressRing<Descriptor, T_capacity>;

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

    std::uint64_t mix64(std::uint64_t value) noexcept
    {
        value ^= value >> 30u;
        value *= 0xbf58'476d'1ce4'e5b9ull;
        value ^= value >> 27u;
        value *= 0x94d0'49bb'1331'11ebull;
        value ^= value >> 31u;
        return value;
    }

    Descriptor makeDescriptor(std::uint64_t sequence)
    {
        Descriptor descriptor;
        descriptor.operation = graph::OperationId{static_cast<std::uint32_t>(sequence % 7u + 1u)};
        descriptor.kind = graph::JobKind::OPERATION;
        descriptor.packet = graph::PacketHandle{
            static_cast<std::uint32_t>(sequence % 17u),
            static_cast<std::uint32_t>(sequence / 17u)};
        descriptor.routing = graph::RoutingInfo{
            static_cast<std::uint32_t>(sequence % 3u),
            static_cast<std::uint32_t>(sequence % 5u),
            static_cast<std::uint32_t>(sequence % 4u + 1u)};
        descriptor.request = graph::RequestSpec{sequence * 3u, sequence + 1u, mix64(sequence)};
        descriptor.sequenceNumber = sequence;
        return descriptor;
    }

    void requireDescriptorEqual(Descriptor const& actual, Descriptor const& expected)
    {
        REQUIRE(actual.operation == expected.operation);
        REQUIRE(actual.kind == expected.kind);
        REQUIRE(actual.packet == expected.packet);
        REQUIRE(actual.routing == expected.routing);
        REQUIRE(actual.request == expected.request);
        REQUIRE(actual.sequenceNumber == expected.sequenceNumber);
    }

    template<std::size_t T_capacity>
    std::span<Descriptor const> asSpan(std::array<Descriptor, T_capacity>& descriptors, std::size_t count)
    {
        return std::span<Descriptor const>{descriptors.data(), count};
    }

    /** Resolver that rejects one sequence number and accepts everything else. */
    struct RejectSequence
    {
        std::uint64_t rejected{0u};

        ALPAKA_FN_ACC bool operator()(Descriptor const& descriptor) const noexcept
        {
            return descriptor.sequenceNumber != rejected;
        }
    };
} // namespace

TEST_CASE("ingress empty ring reports no work", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    CHECK(ring.occupancy() == 0u);
    CHECK(ring.published() == 0u);
    CHECK(ring.consumed() == 0u);
    CHECK_FALSE(ring.isClosed());

    std::array<Descriptor, 4u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 4u, 4u);
    CHECK(batch.status == graph::ImportStatus::nothing_new);
    CHECK(batch.imported == 0u);

    auto const counters = ring.debugCounters();
    CHECK(counters.totalEnqueued == 0u);
    CHECK(counters.totalImported == 0u);
    CHECK(counters.peakOccupancy == 0u);
}

TEST_CASE("ingress accepts exactly capacity then reports ring_full", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    std::array<Descriptor, 2u> jobs{makeDescriptor(0u), makeDescriptor(1u)};
    auto const first = ring.tryEnqueue(asSpan(jobs, jobs.size()));
    CHECK(first.status == graph::EnqueueStatus::accepted);
    CHECK(first.accepted == 2u);
    CHECK(ring.occupancy() == 2u);

    auto const extra = makeDescriptor(2u);
    auto const second = ring.tryEnqueue(std::span<Descriptor const>{&extra, 1u});
    CHECK(second.status == graph::EnqueueStatus::ring_full);
    CHECK(second.accepted == 0u);
    CHECK(ring.published() == 2u);

    std::array<Descriptor, 2u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 2u, 2u);
    REQUIRE(batch.status == graph::ImportStatus::imported);
    REQUIRE(batch.imported == 2u);
    requireDescriptorEqual(staging[0], jobs[0]);
    requireDescriptorEqual(staging[1], jobs[1]);

    CHECK(ring.occupancy() == 0u);
    CHECK(ring.debugCounters().peakOccupancy == 2u);
}

TEST_CASE("ingress capacity one round trips", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<1u>::create(device);

    auto job = makeDescriptor(42u);
    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&job, 1u}).status == graph::EnqueueStatus::accepted);
    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&job, 1u}).status == graph::EnqueueStatus::ring_full);

    std::array<Descriptor, 1u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 1u, 1u);
    REQUIRE(batch.imported == 1u);
    requireDescriptorEqual(staging[0], job);

    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&job, 1u}).status == graph::EnqueueStatus::accepted);
    CHECK(ring.occupancy() == 1u);
}

TEST_CASE("ingress partial batch larger than free space", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    std::array<Descriptor, 3u> initial{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u)};
    CHECK(ring.tryEnqueue(asSpan(initial, initial.size())).accepted == 3u);

    std::array<Descriptor, 4u> large{makeDescriptor(3u), makeDescriptor(4u), makeDescriptor(5u), makeDescriptor(6u)};
    auto const partial = ring.tryEnqueue(asSpan(large, large.size()));
    CHECK(partial.status == graph::EnqueueStatus::ring_full);
    CHECK(partial.accepted == 1u);
    CHECK(ring.occupancy() == 4u);

    std::array<Descriptor, 4u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 4u, 4u);
    REQUIRE(batch.imported == 4u);
    requireDescriptorEqual(staging[3], large[0]);

    auto const refill = ring.tryEnqueue(asSpan(large, large.size()));
    CHECK(refill.status == graph::EnqueueStatus::accepted);
    CHECK(refill.accepted == 4u);
}

TEST_CASE("ingress rejects batches larger than the capacity without a hang", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    std::array<Descriptor, 5u>
        jobs{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u), makeDescriptor(3u), makeDescriptor(4u)};
    auto const first = ring.tryEnqueue(asSpan(jobs, jobs.size()));
    CHECK(first.status == graph::EnqueueStatus::ring_full);
    CHECK(first.accepted == 2u);

    std::array<Descriptor, 2u> staging{};
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 2u, 2u).imported == 2u);

    auto const second = ring.tryEnqueue(asSpan(jobs, jobs.size()));
    CHECK(second.status == graph::EnqueueStatus::ring_full);
    CHECK(second.accepted == 2u);
}

TEST_CASE("ingress backpressure then drain then accept", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    for(std::uint64_t round = 0u; round < 5u; ++round)
    {
        auto first = makeDescriptor(round * 2u);
        auto second = makeDescriptor(round * 2u + 1u);
        std::array<Descriptor, 2u> pair{first, second};
        CHECK(ring.tryEnqueue(asSpan(pair, pair.size())).status == graph::EnqueueStatus::accepted);

        auto overflow = makeDescriptor(100u + round);
        CHECK(ring.tryEnqueue(std::span<Descriptor const>{&overflow, 1u}).status == graph::EnqueueStatus::ring_full);

        std::array<Descriptor, 2u> staging{};
        auto const batch = ring.view().pollAndImportHost(staging.data(), 2u, 2u);
        REQUIRE(batch.imported == 2u);
        requireDescriptorEqual(staging[0], first);
        requireDescriptorEqual(staging[1], second);
    }
}

TEST_CASE("ingress late submission works while the pool is idle", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    std::array<Descriptor, 2u> firstBatch{makeDescriptor(0u), makeDescriptor(1u)};
    CHECK(ring.tryEnqueue(asSpan(firstBatch, firstBatch.size())).accepted == 2u);

    std::array<Descriptor, 2u> staging{};
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 2u, 2u).imported == 2u);
    CHECK(ring.occupancy() == 0u);
    CHECK_FALSE(ring.isClosed());

    auto late = makeDescriptor(99u);
    auto const lateResult = ring.tryEnqueue(std::span<Descriptor const>{&late, 1u});
    CHECK(lateResult.status == graph::EnqueueStatus::accepted);
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 2u, 2u).imported == 1u);
    requireDescriptorEqual(staging[0], late);
}

TEST_CASE("ingress wraparound preserves order across many batches", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<3u>::create(device);

    constexpr std::uint64_t count = 1000u;
    for(std::uint64_t sequence = 0u; sequence < count; ++sequence)
    {
        auto job = makeDescriptor(sequence);
        auto const result = ring.tryEnqueue(std::span<Descriptor const>{&job, 1u});
        REQUIRE(result.status == graph::EnqueueStatus::accepted);

        std::array<Descriptor, 3u> staging{};
        auto const batch = ring.view().pollAndImportHost(staging.data(), 3u, 3u);
        REQUIRE(batch.imported == 1u);
        requireDescriptorEqual(staging[0], job);
    }

    CHECK(ring.published() == count);
    CHECK(ring.consumed() == count);
    CHECK(ring.debugCounters().totalImported == count);
}

TEST_CASE("ingress counters initialised near the counter maximum roll over", "[ingress]")
{
    auto device = makeTestDevice();
    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();

    auto ring = Ring<4u>::create(device, maximum - 2u, maximum - 2u);
    CHECK(ring.published() == maximum - 2u);
    CHECK(ring.consumed() == maximum - 2u);
    CHECK(ring.occupancy() == 0u);

    std::array<Descriptor, 4u> jobs{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u), makeDescriptor(3u)};
    auto const enqueued = ring.tryEnqueue(asSpan(jobs, jobs.size()));
    CHECK(enqueued.status == graph::EnqueueStatus::accepted);
    CHECK(enqueued.accepted == 4u);
    CHECK(ring.published() == 1u);
    CHECK(ring.occupancy() == 4u);

    std::array<Descriptor, 4u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 4u, 4u);
    REQUIRE(batch.imported == 4u);
    for(std::size_t i = 0u; i < jobs.size(); ++i)
        requireDescriptorEqual(staging[i], jobs[i]);
    CHECK(ring.consumed() == 1u);
    CHECK(ring.occupancy() == 0u);

    auto extra = makeDescriptor(4u);
    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&extra, 1u}).status == graph::EnqueueStatus::accepted);
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 4u, 4u).imported == 1u);
    requireDescriptorEqual(staging[0], extra);
}

TEST_CASE("ingress does not overwrite unacknowledged slots", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    auto first = makeDescriptor(100u);
    auto second = makeDescriptor(101u);
    std::array<Descriptor, 2u> initial{first, second};
    REQUIRE(ring.tryEnqueue(asSpan(initial, initial.size())).accepted == 2u);

    std::uint64_t const head = ring.published();
    Descriptor const slot0Before = *ring.descriptorSlot(head - 2u);
    Descriptor const slot1Before = *ring.descriptorSlot(head - 1u);
    requireDescriptorEqual(slot0Before, first);
    requireDescriptorEqual(slot1Before, second);

    auto blocked = makeDescriptor(102u);
    auto const rejected = ring.tryEnqueue(std::span<Descriptor const>{&blocked, 1u});
    CHECK(rejected.status == graph::EnqueueStatus::ring_full);
    requireDescriptorEqual(*ring.descriptorSlot(head - 2u), first);
    requireDescriptorEqual(*ring.descriptorSlot(head - 1u), second);

    std::array<Descriptor, 2u> staging{};
    // Acknowledge only the first slot; the second must stay untouched.
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 2u, 1u).imported == 1u);
    requireDescriptorEqual(staging[0], first);

    auto third = makeDescriptor(102u);
    auto const accepted = ring.tryEnqueue(std::span<Descriptor const>{&third, 1u});
    REQUIRE(accepted.status == graph::EnqueueStatus::accepted);
    // The acknowledged slot may be reused, the still-unacknowledged one must not.
    requireDescriptorEqual(*ring.descriptorSlot(head - 1u), second);

    auto const secondBatch = ring.view().pollAndImportHost(staging.data(), 2u, 2u);
    REQUIRE(secondBatch.imported == 2u);
    requireDescriptorEqual(staging[0], second);
    requireDescriptorEqual(staging[1], third);
}

TEST_CASE("ingress close marker is ordered and blocks later submissions", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    auto first = makeDescriptor(0u);
    auto second = makeDescriptor(1u);
    std::array<Descriptor, 2u> jobs{first, second};
    REQUIRE(ring.tryEnqueue(asSpan(jobs, jobs.size())).accepted == 2u);

    std::uint64_t const beforeClose = ring.published();
    auto const closedResult = ring.closeInput();
    CHECK(closedResult.status == graph::EnqueueStatus::accepted);
    CHECK(closedResult.accepted == 1u);
    CHECK(ring.isClosed());
    CHECK(ring.published() == beforeClose + 1u);

    auto rejected = makeDescriptor(3u);
    auto const rejectedResult = ring.tryEnqueue(std::span<Descriptor const>{&rejected, 1u});
    CHECK(rejectedResult.status == graph::EnqueueStatus::closed);
    CHECK(rejectedResult.accepted == 0u);
    CHECK(ring.published() == beforeClose + 1u);
    CHECK(ring.closeInput().status == graph::EnqueueStatus::closed);

    std::array<Descriptor, 4u> staging{};
    auto const firstBatch = ring.view().pollAndImportHost(staging.data(), 4u, 2u);
    REQUIRE(firstBatch.status == graph::ImportStatus::imported);
    REQUIRE(firstBatch.imported == 2u);
    requireDescriptorEqual(staging[0], first);
    requireDescriptorEqual(staging[1], second);
    CHECK_FALSE(firstBatch.endMarkerSeen);

    auto const markerBatch = ring.view().pollAndImportHost(staging.data(), 4u, 2u);
    REQUIRE(markerBatch.status == graph::ImportStatus::end_marker_seen);
    REQUIRE(markerBatch.imported == 1u);
    CHECK(markerBatch.jobCount == 0u);
    CHECK(staging[0].kind == graph::JobKind::END_OF_INPUT);
    CHECK(ring.occupancy() == 0u);

    CHECK(ring.view().pollAndImportHost(staging.data(), 4u, 2u).status == graph::ImportStatus::nothing_new);
}

TEST_CASE("ingress close marker is only accepted when it fits", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    auto first = makeDescriptor(0u);
    auto second = makeDescriptor(1u);
    std::array<Descriptor, 2u> jobs{first, second};
    REQUIRE(ring.tryEnqueue(asSpan(jobs, jobs.size())).accepted == 2u);

    std::uint64_t const publishedBefore = ring.published();
    CHECK(ring.closeInput().status == graph::EnqueueStatus::ring_full);
    CHECK_FALSE(ring.isClosed());
    CHECK(ring.published() == publishedBefore);

    std::array<Descriptor, 2u> staging{};
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 2u, 1u).imported == 1u);
    CHECK(ring.closeInput().status == graph::EnqueueStatus::accepted);
    CHECK(ring.isClosed());

    REQUIRE(ring.view().pollAndImportHost(staging.data(), 2u, 1u).status == graph::ImportStatus::imported);
    auto const marker = ring.view().pollAndImportHost(staging.data(), 2u, 2u);
    CHECK(marker.status == graph::ImportStatus::end_marker_seen);
    CHECK(staging[0].kind == graph::JobKind::END_OF_INPUT);
}

TEST_CASE("ingress peak occupancy and debug counters", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    std::array<Descriptor, 3u> first{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u)};
    CHECK(ring.tryEnqueue(asSpan(first, first.size())).accepted == 3u);
    CHECK(ring.debugCounters().peakOccupancy == 3u);

    std::array<Descriptor, 4u> staging{};
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 4u, 1u).imported == 1u);

    std::array<Descriptor, 2u> second{makeDescriptor(3u), makeDescriptor(4u)};
    CHECK(ring.tryEnqueue(asSpan(second, second.size())).accepted == 2u);
    CHECK(ring.occupancy() == 4u);
    CHECK(ring.debugCounters().peakOccupancy == 4u);

    REQUIRE(ring.view().pollAndImportHost(staging.data(), 4u, 4u).imported == 4u);
    auto tail = makeDescriptor(5u);
    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&tail, 1u}).accepted == 1u);
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 4u, 4u).imported == 1u);

    auto const counters = ring.debugCounters();
    CHECK(counters.totalEnqueued == 6u);
    CHECK(counters.totalImported == 6u);
    CHECK(counters.peakOccupancy == 4u);
    CHECK(counters.occupancy == 0u);
    CHECK_FALSE(counters.closed);
}

TEST_CASE("ingress reports invalid input as an error and mutates nothing", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    Descriptor marker;
    marker.kind = graph::JobKind::END_OF_INPUT;
    auto const rejected = ring.tryEnqueue(std::span<Descriptor const>{&marker, 1u});
    CHECK(rejected.status == graph::EnqueueStatus::error);
    CHECK(rejected.error == graph::IngressError::invalid_descriptor);
    CHECK(rejected.accepted == 0u);
    CHECK(ring.published() == 0u);
    CHECK_FALSE(ring.isClosed());

    std::array<Descriptor, 4u> staging{};
    auto const badStaging = ring.view().pollAndImportHost(staging.data(), 0u, 4u);
    CHECK(badStaging.status == graph::ImportStatus::error);
    CHECK(badStaging.error == graph::IngressError::staging_too_small);
}

TEST_CASE("ingress setCountersForTest rejects an out-of-range occupancy", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<2u>::create(device);

    CHECK_THROWS_AS(ring.setCountersForTest(5u, 1u), std::invalid_argument);
    CHECK(ring.occupancy() == 0u);

    ring.setCountersForTest(10u, 8u);
    CHECK(ring.published() == 10u);
    CHECK(ring.consumed() == 8u);
    CHECK(ring.occupancy() == 2u);
}

TEST_CASE("ingress enqueueBlocking waits for the consumer and then accepts", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    constexpr std::uint64_t total = 8u;
    std::array<Descriptor, 4u> first{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u), makeDescriptor(3u)};
    REQUIRE(ring.tryEnqueue(asSpan(first, first.size())).accepted == 4u);

    std::array<Descriptor, 4u> second{makeDescriptor(4u), makeDescriptor(5u), makeDescriptor(6u), makeDescriptor(7u)};

    std::vector<Descriptor> received(total);
    std::atomic<std::uint32_t> consumerFailed{0u};

    std::thread consumer(
        [&]
        {
            auto view = ring.view();
            std::uint64_t imported = 0u;
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while(imported < total && std::chrono::steady_clock::now() < deadline)
            {
                std::array<Descriptor, 4u> staging{};
                auto const batch = view.pollAndImportHost(staging.data(), 4u, 2u);
                if(batch.status == graph::ImportStatus::error)
                {
                    consumerFailed.store(1u);
                    return;
                }
                for(std::uint32_t i = 0u; i < batch.imported; ++i)
                {
                    if(imported >= total || staging[i].sequenceNumber != imported)
                    {
                        consumerFailed.store(1u);
                        return;
                    }
                    received[imported] = staging[i];
                    ++imported;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            if(imported != total)
                consumerFailed.store(1u);
        });

    auto const before = std::chrono::steady_clock::now();
    auto const result = ring.enqueueBlocking(asSpan(second, second.size()), std::chrono::seconds(2));
    auto const elapsed = std::chrono::steady_clock::now() - before;
    consumer.join();

    CHECK(result.status == graph::EnqueueStatus::accepted);
    CHECK(result.accepted == 4u);
    CHECK(result.requested == 4u);
    CHECK(elapsed < std::chrono::seconds(5));
    CHECK(consumerFailed.load() == 0u);
    for(std::uint64_t i = 0u; i < total; ++i)
        requireDescriptorEqual(received[i], makeDescriptor(i));
    CHECK(ring.occupancy() == 0u);
    CHECK(ring.consumed() == total);
}

TEST_CASE("ingress enqueueBlocking times out on a full ring without mutating it", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    std::array<Descriptor, 4u> fill{
        makeDescriptor(10u),
        makeDescriptor(11u),
        makeDescriptor(12u),
        makeDescriptor(13u)};
    REQUIRE(ring.tryEnqueue(asSpan(fill, fill.size())).accepted == 4u);

    std::array<Descriptor, 2u> extra{makeDescriptor(20u), makeDescriptor(21u)};
    auto const before = std::chrono::steady_clock::now();
    auto const result = ring.enqueueBlocking(asSpan(extra, extra.size()), std::chrono::milliseconds(25));
    auto const elapsed = std::chrono::steady_clock::now() - before;

    CHECK(result.status == graph::EnqueueStatus::ring_full);
    CHECK(result.accepted == 0u);
    CHECK(result.requested == 2u);
    CHECK(ring.published() == 4u);
    CHECK(ring.occupancy() == 4u);
    // The timeout is honoured: the call waited but terminated on its own.
    CHECK(elapsed >= std::chrono::milliseconds(20));
    CHECK(elapsed < std::chrono::seconds(10));
    for(std::size_t i = 0u; i < fill.size(); ++i)
        requireDescriptorEqual(*ring.descriptorSlot(static_cast<std::uint64_t>(i)), fill[i]);
}

TEST_CASE("ingress enqueueBlocking reports a truthful partial accept on timeout", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    std::array<Descriptor, 2u> fill{makeDescriptor(0u), makeDescriptor(1u)};
    REQUIRE(ring.tryEnqueue(asSpan(fill, fill.size())).accepted == 2u);

    std::array<Descriptor, 4u> offered{makeDescriptor(2u), makeDescriptor(3u), makeDescriptor(4u), makeDescriptor(5u)};
    auto const result = ring.enqueueBlocking(asSpan(offered, offered.size()), std::chrono::milliseconds(25));

    CHECK(result.status == graph::EnqueueStatus::ring_full);
    CHECK(result.accepted == 2u);
    CHECK(result.requested == 4u);
    CHECK(ring.published() == 4u);
    CHECK(ring.occupancy() == 4u);

    std::array<Descriptor, 4u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 4u, 4u);
    REQUIRE(batch.imported == 4u);
    for(std::size_t i = 0u; i < fill.size(); ++i)
        requireDescriptorEqual(staging[i], fill[i]);
    requireDescriptorEqual(staging[2], offered[0]);
    requireDescriptorEqual(staging[3], offered[1]);
    CHECK(ring.occupancy() == 0u);
}

TEST_CASE("ingress requestStop terminates a blocked enqueueBlocking as closed", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    std::array<Descriptor, 4u> fill{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u), makeDescriptor(3u)};
    REQUIRE(ring.tryEnqueue(asSpan(fill, fill.size())).accepted == 4u);

    std::array<Descriptor, 2u> blocked{makeDescriptor(4u), makeDescriptor(5u)};
    graph::EnqueueResult blockedResult;
    std::thread producer(
        [&] { blockedResult = ring.enqueueBlocking(asSpan(blocked, blocked.size()), std::chrono::seconds(1)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Documented exception to the single-mutator rule: requestStop only stores
    // the close flag, which the producer never writes.
    ring.requestStop();
    producer.join();

    CHECK(blockedResult.status == graph::EnqueueStatus::closed);
    CHECK(blockedResult.accepted == 0u);
    CHECK(blockedResult.requested == 2u);
    CHECK(ring.isClosed());
    CHECK(ring.published() == 4u);

    // Already-published descriptors stay importable: the ring is not wedged.
    std::array<Descriptor, 4u> staging{};
    auto const batch = ring.view().pollAndImportHost(staging.data(), 4u, 4u);
    REQUIRE(batch.imported == 4u);
    for(std::size_t i = 0u; i < fill.size(); ++i)
        requireDescriptorEqual(staging[i], fill[i]);
    CHECK(ring.view().pollAndImportHost(staging.data(), 4u, 4u).status == graph::ImportStatus::nothing_new);
    CHECK(
        ring.enqueueBlocking(asSpan(blocked, blocked.size()), std::chrono::milliseconds(0)).status
        == graph::EnqueueStatus::closed);
}

TEST_CASE("ingress enqueueBlocking on a closed ring returns closed without waiting", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    auto job = makeDescriptor(0u);
    REQUIRE(ring.tryEnqueue(std::span<Descriptor const>{&job, 1u}).accepted == 1u);
    REQUIRE(ring.closeInput().status == graph::EnqueueStatus::accepted);

    std::array<Descriptor, 2u> offered{makeDescriptor(1u), makeDescriptor(2u)};
    auto const publishedBefore = ring.published();

    std::span<Descriptor const> const empty{};
    auto const emptyResult = ring.enqueueBlocking(empty, std::chrono::seconds(1));
    CHECK(emptyResult.status == graph::EnqueueStatus::closed);
    CHECK(emptyResult.accepted == 0u);

    auto const before = std::chrono::steady_clock::now();
    auto const result = ring.enqueueBlocking(asSpan(offered, offered.size()), std::chrono::seconds(1));
    auto const elapsed = std::chrono::steady_clock::now() - before;

    CHECK(result.status == graph::EnqueueStatus::closed);
    CHECK(result.accepted == 0u);
    CHECK(result.requested == 2u);
    CHECK(ring.published() == publishedBefore);
    CHECK(elapsed < std::chrono::milliseconds(500));
}

TEST_CASE("ingress concurrent host producer and host importer keep order and exactly-once", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<8u>::create(device);

    constexpr std::uint64_t messageCount = 4096u;
    std::vector<Descriptor> received(messageCount + 1u);
    std::atomic<std::uint32_t> producerFailed{0u};
    std::atomic<std::uint32_t> consumerFailed{0u};
    // Capacity is far below the message count, so the producer must observe a
    // full ring and wait for the importer: that is the concurrency evidence.
    std::atomic<std::uint32_t> backpressure{0u};

    std::thread producer(
        [&]
        {
            for(std::uint64_t sequence = 0u; sequence < messageCount; ++sequence)
            {
                Descriptor const descriptor = makeDescriptor(sequence);
                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                for(;;)
                {
                    auto const result = ring.tryEnqueue(std::span<Descriptor const>{&descriptor, 1u});
                    if(result.status == graph::EnqueueStatus::accepted)
                        break;
                    if(result.status == graph::EnqueueStatus::ring_full)
                        backpressure.fetch_add(1u);
                    if(result.status == graph::EnqueueStatus::closed || result.status == graph::EnqueueStatus::error
                       || std::chrono::steady_clock::now() >= deadline)
                    {
                        producerFailed.store(1u);
                        return;
                    }
                    std::this_thread::yield();
                }
            }
            auto const closeDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            for(;;)
            {
                auto const result = ring.closeInput();
                if(result.status == graph::EnqueueStatus::accepted)
                    return;
                if(result.status == graph::EnqueueStatus::error || std::chrono::steady_clock::now() >= closeDeadline)
                {
                    producerFailed.store(1u);
                    return;
                }
                std::this_thread::yield();
            }
        });

    std::thread consumer(
        [&]
        {
            auto view = ring.view();
            std::uint64_t expected = 0u;
            std::uint64_t lastConsumed = 0u;
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
            for(;;)
            {
                std::array<Descriptor, 8u> staging{};
                auto const batch = view.pollAndImportHost(staging.data(), 8u, 4u);
                if(batch.status == graph::ImportStatus::error)
                {
                    consumerFailed.store(1u);
                    return;
                }
                for(std::uint32_t i = 0u; i < batch.imported; ++i)
                {
                    if(staging[i].kind == graph::JobKind::END_OF_INPUT)
                    {
                        // The marker carries no user sequence number; it must be
                        // the last descriptor of the batch and of the stream.
                        if(expected != messageCount || i + 1u != batch.imported)
                        {
                            consumerFailed.store(1u);
                            return;
                        }
                        received[messageCount] = staging[i];
                        ++expected;
                        continue;
                    }
                    if(expected >= messageCount || staging[i].sequenceNumber != expected)
                    {
                        consumerFailed.store(1u);
                        return;
                    }
                    received[expected] = staging[i];
                    ++expected;
                }

                // Single consumer: consumed must advance monotonically. Compare
                // modular deltas, never absolute magnitudes.
                std::uint64_t const consumed = ring.consumed();
                if(consumed - lastConsumed > 8u)
                {
                    consumerFailed.store(1u);
                    return;
                }
                lastConsumed = consumed;

                if(batch.endMarkerSeen)
                    break;
                if(std::chrono::steady_clock::now() >= deadline)
                {
                    consumerFailed.store(1u);
                    return;
                }
                std::this_thread::yield();
            }
            if(expected != messageCount + 1u)
                consumerFailed.store(1u);
        });

    producer.join();
    consumer.join();

    CHECK(producerFailed.load() == 0u);
    CHECK(consumerFailed.load() == 0u);
    CHECK(backpressure.load() > 0u);
    for(std::uint64_t sequence = 0u; sequence < messageCount; ++sequence)
        requireDescriptorEqual(received[sequence], makeDescriptor(sequence));
    CHECK(received[messageCount].kind == graph::JobKind::END_OF_INPUT);

    auto const counters = ring.debugCounters();
    CHECK(counters.published == messageCount + 1u);
    CHECK(counters.consumed == messageCount + 1u);
    CHECK(counters.totalEnqueued == messageCount + 1u);
    CHECK(counters.totalImported == messageCount + 1u);
    CHECK(counters.occupancy == 0u);
    CHECK(counters.closed);
    CHECK(counters.importError == graph::IngressError::none);
}

TEST_CASE("ingress import error acknowledges nothing and does not poison the ring", "[ingress]")
{
    auto device = makeTestDevice();
    auto ring = Ring<4u>::create(device);

    std::array<Descriptor, 3u> jobs{makeDescriptor(0u), makeDescriptor(1u), makeDescriptor(2u)};
    REQUIRE(ring.tryEnqueue(asSpan(jobs, jobs.size())).accepted == 3u);

    std::array<Descriptor, 4u> staging{};
    auto const rejected = ring.view().pollAndImportHost(staging.data(), 4u, 4u, RejectSequence{1u});
    CHECK(rejected.status == graph::ImportStatus::error);
    CHECK(rejected.error == graph::IngressError::invalid_descriptor);
    CHECK(rejected.imported == 0u);
    CHECK(rejected.jobCount == 0u);
    CHECK_FALSE(rejected.endMarkerSeen);

    // Nothing acknowledged: the rejected descriptor and its predecessor stay in
    // the ring and `consumed` does not pass them.
    CHECK(ring.consumed() == 0u);
    CHECK(ring.published() == 3u);
    CHECK(ring.occupancy() == 3u);
    CHECK(ring.debugCounters().totalImported == 0u);
    CHECK(ring.debugCounters().importError == graph::IngressError::invalid_descriptor);
    CHECK(ring.descriptorSlot(1u)->sequenceNumber == 1u);

    // Not poisoned: the same descriptors import once the resolver accepts them.
    auto const accepted = ring.view().pollAndImportHost(staging.data(), 4u, 4u);
    REQUIRE(accepted.status == graph::ImportStatus::imported);
    REQUIRE(accepted.imported == 3u);
    for(std::size_t i = 0u; i < jobs.size(); ++i)
        requireDescriptorEqual(staging[i], jobs[i]);
    CHECK(ring.consumed() == 3u);
    CHECK(ring.debugCounters().totalImported == 3u);
    // The error code is sticky, so the host can still observe it afterwards.
    CHECK(ring.debugCounters().importError == graph::IngressError::invalid_descriptor);

    // And the ring is still fully usable.
    auto const extra = makeDescriptor(3u);
    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&extra, 1u}).status == graph::EnqueueStatus::accepted);
    REQUIRE(ring.view().pollAndImportHost(staging.data(), 4u, 4u).imported == 1u);
    requireDescriptorEqual(staging[0], extra);
    CHECK(ring.occupancy() == 0u);
}
