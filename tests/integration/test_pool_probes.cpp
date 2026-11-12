// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Diagnostics probes for the pool-core CUDA correctness work (T8b).
//
// Probe 1: broadcast a multi-field POD (the real JobDescriptor shape) from the
// warp leader through graph::WarpGroup::broadcast, field by field, and verify
// every lane of the warp receives byte-identical values.
//
// Probe 2: the ready-ring claim path. The leader claims from a ready ring,
// broadcastEntry runs, and every lane reports the full handle, generation,
// operation id, and request fields it observed after the broadcast.
//
// Probe 3: the packet-pool handle path, the same exercise through
// PacketPool::acquire (whose broadcastHandle covers index and generation).
//
// The probes exercise warp-collective semantics (shuffle, the leader-only
// claim, the width-32 shfl path of broadcast). They are meaningful only where
// a real warp exists, so they run on the GPU build (warp width 32) via the
// gpuCuda executor and are excluded from the CPU-only build, where the host
// accelerator has no warp and the shfl path is degenerate.

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/packet_pool.hpp>
#include <graph/ready_ring.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
#if GRAPH_HAS_CUDA
    auto makeHandles()
    {
        auto const spec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
        auto selector = alpaka::onHost::makeDeviceSelector(spec);
        REQUIRE(selector.isAvailable());
        REQUIRE(selector.getDeviceCount() > 0u);
        auto device = selector.makeDevice(0);
        auto executor = alpaka::exec::gpuCuda;
        return std::pair(std::move(device), executor);
    }
#else
    auto makeHandles()
    {
        auto const spec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
        auto selector = alpaka::onHost::makeDeviceSelector(spec);
        REQUIRE(selector.getDeviceCount() > 0u);
        auto device = selector.makeDevice(0);
        auto executor = alpaka::onHost::defaultExecutor(device);
        return std::pair(std::move(device), executor);
    }
#endif

    struct DescriptorProbe
    {
        std::uint32_t op{0u};
        std::uint32_t kind{0u};
        std::uint32_t packetIndex{0u};
        std::uint32_t packetGeneration{0u};
        std::uint32_t inputPort{0u};
        std::uint32_t outputPort{0u};
        std::uint32_t outputPortCount{0u};
        std::uint64_t firstId{0u};
        std::uint64_t count{0u};
        std::uint64_t seed{0u};
        std::uint64_t sequenceNumber{0u};
    };

    static_assert(sizeof(DescriptorProbe) == sizeof(graph::JobDescriptor));
    static_assert(alignof(DescriptorProbe) == alignof(graph::JobDescriptor));

    template<typename T_Acc>
    ALPAKA_FN_ACC void broadcastDescriptorProbeBody(
        T_Acc const& acc,
        std::uint32_t laneCount,
        DescriptorProbe const& source,
        DescriptorProbe* received)
    {
        graph::WarpGroup<T_Acc> const warp{acc};
        if(warp.laneIndex() >= laneCount)
            return;
        auto const index = static_cast<std::size_t>(warp.laneIndex());

        DescriptorProbe value{};
        if(warp.isLeader())
            value = source;

        // The ready-ring broadcast order, field by field.
        value.op = warp.template broadcast<std::uint32_t>(value.op, 0u);
        value.kind = warp.template broadcast<std::uint32_t>(value.kind, 0u);
        value.packetIndex = warp.template broadcast<std::uint32_t>(value.packetIndex, 0u);
        value.packetGeneration = warp.template broadcast<std::uint32_t>(value.packetGeneration, 0u);
        value.inputPort = warp.template broadcast<std::uint32_t>(value.inputPort, 0u);
        value.outputPort = warp.template broadcast<std::uint32_t>(value.outputPort, 0u);
        value.outputPortCount = warp.template broadcast<std::uint32_t>(value.outputPortCount, 0u);
        value.firstId = warp.template broadcast<std::uint64_t>(value.firstId, 0u);
        value.count = warp.template broadcast<std::uint64_t>(value.count, 0u);
        value.seed = warp.template broadcast<std::uint64_t>(value.seed, 0u);
        value.sequenceNumber = warp.template broadcast<std::uint64_t>(value.sequenceNumber, 0u);

        // The additive multi-field overload: one call exchanges the whole POD.
        DescriptorProbe whole = value;
        whole = warp.broadcast(whole, 0u, laneCount);
        value.op = whole.op;
        value.kind = whole.kind;
        value.packetIndex = whole.packetIndex;
        value.packetGeneration = whole.packetGeneration;
        value.inputPort = whole.inputPort;
        value.outputPort = whole.outputPort;
        value.outputPortCount = whole.outputPortCount;
        value.firstId = whole.firstId;
        value.count = whole.count;
        value.seed = whole.seed;
        value.sequenceNumber = whole.sequenceNumber;

        received[index] = value;
    }

    struct BroadcastDescriptorKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, std::uint32_t laneCount, DescriptorProbe const& source,
                                      DescriptorProbe* received) const
        {
            broadcastDescriptorProbeBody(acc, laneCount, source, received);
        }
    };

    template<typename T_Acc>
    ALPAKA_FN_ACC void ringClaimProbeBody(
        T_Acc const& acc,
        std::uint32_t laneCount,
        void* ringStorage,
        std::uint32_t entryIndex,
        DescriptorProbe const& expected,
        DescriptorProbe* reported)
    {
        using Ring = graph::ReadyRing<T_Acc, 16u>;
        Ring ring{ringStorage};
        graph::WarpGroup<T_Acc> const warp{acc};
        if(warp.laneIndex() >= laneCount)
            return;
        auto const index = static_cast<std::size_t>(warp.laneIndex());

        // Fill one ring entry directly: the leader writes the full payload
        // plus the stamp, mirroring what enqueue() does under the lock.
        if(warp.isLeader())
        {
            auto* slot = reinterpret_cast<Ring::RingSlot*>(
                static_cast<char*>(ringStorage) + Ring::headerBytes + static_cast<std::size_t>(entryIndex)
                    * Ring::slotBytes);
            std::uint64_t const stamp = static_cast<std::uint64_t>(entryIndex);
            std::memcpy(
                &slot->entry,
                reinterpret_cast<DescriptorProbe const*>(&expected),
                sizeof(graph::JobDescriptor));
            slot->sequence = stamp;
            auto* hdr = reinterpret_cast<Ring::RingHeader*>(ringStorage);
            hdr->pushCounter = static_cast<std::uint64_t>(entryIndex + 1u);
            hdr->claimCounter = stamp;
        }

        graph::JobDescriptor entry{};
        auto const status = ring.dequeue(acc, warp, entry);
        (void) status;
        // The scheduler runs a warp barrier after a claim; replicate it.
        alpaka::onAcc::syncBlockThreads(acc);

        auto& out = reported[index];
        std::memcpy(&out, &entry, sizeof(graph::JobDescriptor));
    }

    struct RingClaimKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void
        operator()(T_Acc const& acc, std::uint32_t laneCount, void* ringStorage, std::uint32_t entryIndex,
                   DescriptorProbe const& expected, DescriptorProbe* reported) const
        {
            ringClaimProbeBody(acc, laneCount, ringStorage, entryIndex, expected, reported);
        }
    };

    template<typename T_Acc>
    ALPAKA_FN_ACC void poolAcquireProbeBody(
        T_Acc const& acc,
        std::uint32_t laneCount,
        void* poolStorage,
        graph::PacketHandle* reported)
    {
        using Pool = graph::PacketPool<T_Acc, 8u, 128u>;
        Pool pool{poolStorage};
        graph::WarpGroup<T_Acc> const warp{acc};
        if(warp.laneIndex() >= laneCount)
            return;
        auto const index = static_cast<std::size_t>(warp.laneIndex());

        graph::PacketPoolBudget budget{};
        budget.liveParticles = 2u;
        budget.controlAllowance = 2u;
        pool.configure(acc, warp, budget);
        alpaka::onAcc::syncBlockThreads(acc);

        graph::PacketHandle handle{};
        auto const ok = pool.acquire(acc, warp, handle);
        (void) ok;
        reported[index] = handle;
    }

    struct PoolAcquireKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void
        operator()(T_Acc const& acc, std::uint32_t laneCount, void* poolStorage, graph::PacketHandle* reported) const
        {
            poolAcquireProbeBody(acc, laneCount, poolStorage, reported);
        }
    };

    struct WidthKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, std::uint32_t* out) const
        {
            graph::WarpGroup<T_Acc> const warp{acc};
            if(warp.isLeader())
                *out = warp.workerWidth();
        }
    };

    template<typename T_Device, typename T_Executor>
    std::uint32_t warpWidthOf(T_Device const& device, T_Executor const& executor)
    {
        auto buffer = alpaka::onHost::alloc<std::uint32_t>(device, 1u);
        auto queue = const_cast<std::remove_cvref_t<T_Device>&>(device).makeQueue();
        queue.enqueue(
            alpaka::onHost::ThreadSpec{std::uint32_t{1u}, std::uint32_t{32u}, executor}, WidthKernel{},
            alpaka::onHost::data(buffer));
        alpaka::onHost::wait(queue);
        auto host = alpaka::onHost::allocHost<std::uint32_t>(1u);
        alpaka::onHost::memcpy(queue, host, buffer);
        alpaka::onHost::wait(queue);
        return *alpaka::onHost::data(host);
    }

    std::uint32_t reportMismatches(
        char const* probeName,
        DescriptorProbe const* received,
        std::size_t laneCount,
        DescriptorProbe const& expected)
    {
        std::uint32_t mismatches = 0u;
        for(std::size_t lane = 0u; lane < laneCount; ++lane)
        {
            auto const& value = received[lane];
            bool const ok
                = value.op == expected.op && value.kind == expected.kind
                  && value.packetIndex == expected.packetIndex
                  && value.packetGeneration == expected.packetGeneration
                  && value.inputPort == expected.inputPort && value.outputPort == expected.outputPort
                  && value.outputPortCount == expected.outputPortCount && value.firstId == expected.firstId
                  && value.count == expected.count && value.seed == expected.seed
                  && value.sequenceNumber == expected.sequenceNumber;
            if(!ok)
            {
                ++mismatches;
                std::printf(
                    "[probe] %s lane=%zu: op=0x%x kind=%u pkt=%u/0x%x ports=%u/%u/%u req=(%llu,%llu,0x%llx) seq=%llu "
                    "expected op=0x%x kind=%u pkt=%u/0x%x ports=%u/%u/%u req=(%llu,%llu,0x%llx) seq=%llu\n",
                    probeName,
                    lane,
                    value.op,
                    value.kind,
                    value.packetIndex,
                    value.packetGeneration,
                    value.inputPort,
                    value.outputPort,
                    value.outputPortCount,
                    static_cast<unsigned long long>(value.firstId),
                    static_cast<unsigned long long>(value.count),
                    static_cast<unsigned long long>(value.seed),
                    static_cast<unsigned long long>(value.sequenceNumber),
                    expected.op,
                    expected.kind,
                    expected.packetIndex,
                    expected.packetGeneration,
                    expected.inputPort,
                    expected.outputPort,
                    expected.outputPortCount,
                    static_cast<unsigned long long>(expected.firstId),
                    static_cast<unsigned long long>(expected.count),
                    static_cast<unsigned long long>(expected.seed),
                    static_cast<unsigned long long>(expected.sequenceNumber));
                std::fflush(stdout);
            }
        }
        return mismatches;
    }

    DescriptorProbe makeExpected()
    {
        DescriptorProbe expected{};
        expected.op = 0x00c0ffeeu;
        expected.kind = 0u;
        expected.packetIndex = 0x0badf00du;
        expected.packetGeneration = 7u;
        expected.inputPort = 3u;
        expected.outputPort = 1u;
        expected.outputPortCount = 2u;
        expected.firstId = 0x1122'3344'5566'7788ull;
        expected.count = 987654321ull;
        expected.seed = 0xfedc'ba98'7654'3210ull;
        expected.sequenceNumber = 4242424242ull;
        return expected;
    }
} // namespace

#if GRAPH_HAS_CUDA

TEST_CASE("probe 1: warp broadcast of the JobDescriptor field set", "[graph][pool][probe][gpu]")
{
    auto [device, executor] = makeHandles();

    auto const width = warpWidthOf(device, executor);
    std::uint32_t const laneCount = width > 0u && width <= 32u ? width : 1u;

    auto const expected = makeExpected();

    auto receivedDevice = alpaka::onHost::alloc<DescriptorProbe>(device, laneCount);
    auto hostReceived = alpaka::onHost::allocHost<DescriptorProbe>(laneCount);
    std::memset(alpaka::onHost::data(hostReceived), 0xff, laneCount * sizeof(DescriptorProbe));
    auto queue = device.makeQueue();
    alpaka::onHost::memcpy(queue, receivedDevice, hostReceived);

    queue.enqueue(
        alpaka::onHost::ThreadSpec{std::uint32_t{1u}, std::uint32_t{32u}, executor}, BroadcastDescriptorKernel{},
        laneCount, expected, alpaka::onHost::data(receivedDevice));
    alpaka::onHost::wait(queue);

    alpaka::onHost::memcpy(queue, hostReceived, receivedDevice);
    alpaka::onHost::wait(queue);

    auto const mismatches = reportMismatches("broadcast", alpaka::onHost::data(hostReceived), laneCount, expected);
    std::printf("[probe] broadcast: lanes=%u mismatches=%u\n", laneCount, mismatches);
    std::fflush(stdout);
    CHECK(mismatches == 0u);
}

TEST_CASE("probe 2: ready-ring claim and broadcast of the entry", "[graph][pool][probe][gpu]")
{
    auto [device, executor] = makeHandles();

    auto const width = warpWidthOf(device, executor);
    std::uint32_t const laneCount = width > 0u && width <= 32u ? width : 1u;

    auto const expected = makeExpected();

    using Ring = graph::ReadyRing<void*, 16u>;
    auto ringStorage = alpaka::onHost::alloc<std::uint8_t>(device, Ring::storageBytes);
    auto reportedDevice = alpaka::onHost::alloc<DescriptorProbe>(device, laneCount);
    auto hostReported = alpaka::onHost::allocHost<DescriptorProbe>(laneCount);
    std::memset(alpaka::onHost::data(hostReported), 0xff, laneCount * sizeof(DescriptorProbe));
    auto queue = device.makeQueue();
    alpaka::onHost::memcpy(queue, reportedDevice, hostReported);

    queue.enqueue(
        alpaka::onHost::ThreadSpec{std::uint32_t{1u}, std::uint32_t{32u}, executor}, RingClaimKernel{}, laneCount,
        alpaka::onHost::data(ringStorage), 5u, expected, alpaka::onHost::data(reportedDevice));
    alpaka::onHost::wait(queue);

    alpaka::onHost::memcpy(queue, hostReported, reportedDevice);
    alpaka::onHost::wait(queue);

    auto const mismatches = reportMismatches("ring-claim", alpaka::onHost::data(hostReported), laneCount, expected);
    std::printf("[probe] ring-claim: lanes=%u mismatches=%u\n", laneCount, mismatches);
    std::fflush(stdout);
    CHECK(mismatches == 0u);
}

TEST_CASE("probe 3: packet-pool acquire broadcasts the handle", "[graph][pool][probe][gpu]")
{
    auto [device, executor] = makeHandles();

    auto const width = warpWidthOf(device, executor);
    std::uint32_t const laneCount = width > 0u && width <= 32u ? width : 1u;

    using Pool = graph::PacketPool<void*, 8u, 128u>;
    auto poolStorage = alpaka::onHost::alloc<std::uint8_t>(device, Pool::storageBytes);
    auto reportedDevice = alpaka::onHost::alloc<graph::PacketHandle>(device, laneCount);
    auto hostReported = alpaka::onHost::allocHost<graph::PacketHandle>(laneCount);
    auto queue = device.makeQueue();
    alpaka::onHost::memcpy(queue, reportedDevice, hostReported);

    queue.enqueue(
        alpaka::onHost::ThreadSpec{std::uint32_t{1u}, std::uint32_t{32u}, executor}, PoolAcquireKernel{}, laneCount,
        alpaka::onHost::data(poolStorage), alpaka::onHost::data(reportedDevice));
    alpaka::onHost::wait(queue);

    alpaka::onHost::memcpy(queue, hostReported, reportedDevice);
    alpaka::onHost::wait(queue);

    auto const handles = alpaka::onHost::data(hostReported);
    graph::PacketHandle const first = handles[0u];
    std::uint32_t mismatches = 0u;
    for(std::uint32_t lane = 0u; lane < laneCount; ++lane)
    {
        if(handles[lane].index != first.index || handles[lane].generation != first.generation)
        {
            ++mismatches;
            std::printf(
                "[probe] pool-acquire lane=%u: %u/0x%x expected %u/0x%x\n",
                lane,
                handles[lane].index,
                handles[lane].generation,
                first.index,
                first.generation);
        }
    }
    std::printf(
        "[probe] pool-acquire: lanes=%u mismatches=%u handle=%u/0x%x\n",
        laneCount,
        mismatches,
        first.index,
        first.generation);
    std::fflush(stdout);
    CHECK(mismatches == 0u);
    CHECK(first.generation == 1u);
}

#endif // GRAPH_HAS_CUDA
