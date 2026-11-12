// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/ingress_ring.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>

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
