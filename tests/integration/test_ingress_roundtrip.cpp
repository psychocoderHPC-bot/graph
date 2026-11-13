// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/ingress_ring.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <span>
#include <thread>

#if GRAPH_HAS_CUDA
#    include <cuda/atomic>
#endif

/** End-to-end ingress protocol test.
 *
 * On a CUDA build the host producer thread publishes descriptors while a real
 * device warp imports them. Lane 0 polls the host-facing ring; all lanes take
 * part in the device-scope block synchronisation that copies and validates the
 * imported batch. On a CPU-only build the identical protocol runs against a
 * host importer thread through the CPU accelerator and the same
 * `pollAndImport` code path. That CPU run is protocol evidence, not GPU
 * evidence.
 */
namespace
{
    using Descriptor = graph::JobDescriptor;

    constexpr std::uint32_t ringCapacity = 32u;
    constexpr std::uint32_t stagingCapacity = 16u;
    constexpr std::uint32_t maxBatch = 8u;
    constexpr std::uint64_t userJobs = 2048u;

    using Ring = graph::IngressRing<Descriptor, ringCapacity>;

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

    auto makeLaunchCfg()
    {
#if GRAPH_HAS_CUDA
        return alpaka::onHost::ThreadSpec{1u, 32u, alpaka::exec::gpuCuda};
#else
        // One host importer thread; "warp" width is one on the serial CPU path.
        return alpaka::onHost::FrameSpec{1u, 1u, alpaka::exec::cpuSerial};
#endif
    }

    ALPAKA_FN_ACC inline std::uint64_t mix64(std::uint64_t value) noexcept
    {
        value ^= value >> 30u;
        value *= 0xbf58'476d'1ce4'e5b9ull;
        value ^= value >> 27u;
        value *= 0x94d0'49bb'1331'11ebull;
        value ^= value >> 31u;
        return value;
    }

    /** Deterministic descriptor used by producer, device validation, and host
     *  verification. */
    ALPAKA_FN_ACC inline Descriptor makeDescriptorValue(std::uint64_t sequence) noexcept
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

    ALPAKA_FN_ACC inline bool descriptorEquals(Descriptor const& left, Descriptor const& right) noexcept
    {
        return left.operation == right.operation && left.kind == right.kind && left.packet == right.packet
               && left.routing.inputPort == right.routing.inputPort
               && left.routing.outputPort == right.routing.outputPort
               && left.routing.outputPortCount == right.routing.outputPortCount
               && left.request.firstId == right.request.firstId && left.request.count == right.request.count
               && left.request.seed == right.request.seed && left.sequenceNumber == right.sequenceNumber;
    }

    std::uint32_t hostFlagLoad(std::uint32_t* ptr)
    {
#if GRAPH_HAS_CUDA
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system> ref(*ptr);
        return graph::host::hostLoadAcquire(ref);
#else
        std::atomic_ref<std::uint32_t> ref(*ptr);
        return graph::host::hostLoadAcquire(ref);
#endif
    }

    void hostFlagStore(std::uint32_t* ptr, std::uint32_t value)
    {
#if GRAPH_HAS_CUDA
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system> ref(*ptr);
        graph::host::hostStoreRelease(ref, value);
#else
        std::atomic_ref<std::uint32_t> ref(*ptr);
        graph::host::hostStoreRelease(ref, value);
#endif
    }

    /** Importer kernel, one warp lane owns the ring.
     *
     * flags layout: [0] stop, [1] done, [2] error, [3] end marker seen,
     * [4] imported low word, [5] imported high word.
     */
    struct ImporterKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            typename Ring::View view,
            Descriptor* staging,
            Descriptor* received,
            std::uint32_t* flags,
            std::uint64_t expectedJobs) const
        {
            auto const laneVector = acc.getIdxWithin(alpaka::onAcc::origin::warp, alpaka::onAcc::unit::threads);
            std::uint32_t const lane = laneVector[0];
            std::uint32_t const warpSize = alpaka::onAcc::warp::getSize(acc);

            auto& sharedCount = alpaka::onAcc::declareSharedVar<std::uint32_t, alpaka::uniqueId()>(acc);
            auto& sharedEnd = alpaka::onAcc::declareSharedVar<std::uint32_t, alpaka::uniqueId()>(acc);
            auto& sharedStop = alpaka::onAcc::declareSharedVar<std::uint32_t, alpaka::uniqueId()>(acc);
            auto& sharedBase = alpaka::onAcc::declareSharedVar<std::uint64_t, alpaka::uniqueId()>(acc);
            auto& sharedError = alpaka::onAcc::declareSharedVar<std::uint32_t, alpaka::uniqueId()>(acc);

            std::uint64_t total = 0u;
            std::uint32_t error = 0u;
            bool endSeen = false;

            for(;;)
            {
                if(lane == 0u)
                {
                    graph::ImportBatch const batch = view.pollAndImport(acc, staging, stagingCapacity, maxBatch);
                    sharedCount = batch.imported;
                    sharedEnd = batch.endMarkerSeen ? 1u : 0u;
                    sharedBase = total;
                    sharedError = batch.status == graph::ImportStatus::error ? 1u : 0u;
                    sharedStop = graph::systemLoadAcquire(acc, flags + 0u);
                }
                alpaka::onAcc::syncBlockThreads(acc);

                std::uint32_t const count = sharedCount;
                std::uint64_t const base = sharedBase;
                for(std::uint32_t i = lane; i < count; i += warpSize)
                    received[base + i] = staging[i];
                alpaka::onAcc::syncBlockThreads(acc);

                if(lane == 0u)
                {
                    for(std::uint32_t i = 0u; i < count; ++i)
                    {
                        std::uint64_t const globalIndex = base + i;
                        if(globalIndex < expectedJobs)
                        {
                            if(!descriptorEquals(staging[i], makeDescriptorValue(globalIndex)))
                                sharedError = 1u;
                        }
                        else if(staging[i].kind != graph::JobKind::END_OF_INPUT)
                        {
                            sharedError = 1u;
                        }
                    }

                    if(sharedError != 0u)
                        error = 1u;
                    total += count;
                    if(sharedEnd != 0u)
                        endSeen = true;
                }

                if(sharedError != 0u || sharedEnd != 0u || sharedStop != 0u)
                    break;
            }

            if(lane == 0u)
            {
                graph::systemStoreRelease(acc, flags + 2u, error);
                graph::systemStoreRelease(acc, flags + 3u, endSeen ? 1u : 0u);
                graph::systemStoreRelease(acc, flags + 4u, static_cast<std::uint32_t>(total & 0xffff'ffffull));
                graph::systemStoreRelease(acc, flags + 5u, static_cast<std::uint32_t>(total >> 32u));
                graph::systemStoreRelease(acc, flags + 1u, 1u);
            }
        }
    };

    /** Resolver that rejects the descriptor with sequence number one. */
    struct RejectSequenceOne
    {
        ALPAKA_FN_ACC bool operator()(Descriptor const& descriptor) const noexcept
        {
            return descriptor.sequenceNumber != 1u;
        }
    };

    /** One-shot importer: the first poll rejects, the second poll accepts all.
     *
     * flags layout: [0] first status, [1] first error, [2] first imported,
     * [3] second status, [4] second imported, [5] second error.
     */
    struct ErrorPathKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            typename Ring::View view,
            Descriptor* staging,
            std::uint32_t* flags) const
        {
            if(alpaka::onAcc::warp::getLaneIdx(acc) != 0u)
                return;

            graph::ImportBatch const rejected
                = view.pollAndImport(acc, staging, stagingCapacity, maxBatch, RejectSequenceOne{});
            graph::systemStoreRelease(acc, flags + 0u, static_cast<std::uint32_t>(rejected.status));
            graph::systemStoreRelease(acc, flags + 1u, static_cast<std::uint32_t>(rejected.error));
            graph::systemStoreRelease(acc, flags + 2u, rejected.imported);

            graph::ImportBatch const accepted = view.pollAndImport(acc, staging, stagingCapacity, maxBatch);
            graph::systemStoreRelease(acc, flags + 3u, static_cast<std::uint32_t>(accepted.status));
            graph::systemStoreRelease(acc, flags + 4u, accepted.imported);
            graph::systemStoreRelease(acc, flags + 5u, static_cast<std::uint32_t>(accepted.error));
        }
    };
} // namespace

TEST_CASE("ingress round trip preserves order, exactly once, and close last", "[ingress][roundtrip]")
{
    auto device = makeTestDevice();
    auto ring = Ring::create(device);

    auto staging = alpaka::onHost::alloc<Descriptor>(device, alpaka::Vec<std::uint32_t, 1u>{stagingCapacity});
    auto received = alpaka::onHost::alloc<Descriptor>(
        device,
        alpaka::Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(userJobs + 1u)});
    auto flagBuffer = alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec<std::uint32_t, 1u>{8u});
    auto* flags = alpaka::onHost::data(flagBuffer);
    for(std::uint32_t i = 0u; i < 8u; ++i)
        hostFlagStore(flags + i, 0u);

    auto importerDevice = makeTestDevice();
    auto const launchCfg = makeLaunchCfg();
    std::thread importer(
        [&, importerDevice = std::move(importerDevice)]() mutable
        {
            auto queue = importerDevice.makeQueue();
            queue.enqueue(
                launchCfg,
                ImporterKernel{},
                ring.view(),
                alpaka::onHost::data(staging),
                alpaka::onHost::data(received),
                flags,
                userJobs);
            alpaka::onHost::wait(queue);
        });

    bool observedConsumedProgress = false;
    bool producerFinished = true;
    std::uint64_t lastConsumed = ring.consumed();

    auto const producerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    for(std::uint64_t sequence = 0u; sequence < userJobs; ++sequence)
    {
        Descriptor const descriptor = makeDescriptorValue(sequence);
        for(;;)
        {
            auto const result = ring.tryEnqueue(std::span<Descriptor const>{&descriptor, 1u});
            if(result.status == graph::EnqueueStatus::accepted)
                break;
            if(result.status == graph::EnqueueStatus::closed || result.status == graph::EnqueueStatus::error)
            {
                producerFinished = false;
                hostFlagStore(flags + 0u, 1u);
                break;
            }

            std::uint64_t const consumed = ring.consumed();
            if(consumed != lastConsumed)
            {
                lastConsumed = consumed;
                observedConsumedProgress = true;
            }
            if(std::chrono::steady_clock::now() >= producerDeadline)
            {
                producerFinished = false;
                hostFlagStore(flags + 0u, 1u);
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        if(!producerFinished)
            break;

        std::uint64_t const consumed = ring.consumed();
        if(consumed != lastConsumed)
        {
            lastConsumed = consumed;
            observedConsumedProgress = true;
        }
    }

    // Explicitly observe host progress with an acquire load before closing.
    auto const progressDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while(producerFinished && ring.consumed() == 0u && std::chrono::steady_clock::now() < progressDeadline)
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    if(ring.consumed() > 0u)
        observedConsumedProgress = true;

    bool closeAccepted = false;
    if(producerFinished)
    {
        for(;;)
        {
            auto const result = ring.closeInput();
            if(result.status == graph::EnqueueStatus::accepted)
            {
                closeAccepted = true;
                break;
            }
            if(result.status == graph::EnqueueStatus::error)
                break;
            if(std::chrono::steady_clock::now() >= producerDeadline)
            {
                hostFlagStore(flags + 0u, 1u);
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }

    bool importerFinished = false;
    auto const importerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while(std::chrono::steady_clock::now() < importerDeadline)
    {
        if(hostFlagLoad(flags + 1u) != 0u)
        {
            importerFinished = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if(!importerFinished)
    {
        // Finite-deadline escape: ask the importer to stop, then give it time
        // to observe the flag. The kernel checks it on every poll iteration.
        hostFlagStore(flags + 0u, 1u);
        auto const stopDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while(std::chrono::steady_clock::now() < stopDeadline)
        {
            if(hostFlagLoad(flags + 1u) != 0u)
            {
                importerFinished = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    importer.join();

    REQUIRE(producerFinished);
    REQUIRE(closeAccepted);
    REQUIRE(importerFinished);
    REQUIRE(observedConsumedProgress);
    REQUIRE(hostFlagLoad(flags + 2u) == 0u);
    REQUIRE(hostFlagLoad(flags + 3u) == 1u);

    std::uint64_t const importedTotal = static_cast<std::uint64_t>(hostFlagLoad(flags + 4u))
                                        | (static_cast<std::uint64_t>(hostFlagLoad(flags + 5u)) << 32u);
    REQUIRE(importedTotal == userJobs + 1u);

    auto hostReceived = alpaka::onHost::allocHost<Descriptor>(
        alpaka::Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(userJobs + 1u)});
    auto copyQueue = device.makeQueue();
    alpaka::onHost::memcpy(copyQueue, hostReceived, received);
    alpaka::onHost::wait(copyQueue);

    for(std::uint64_t sequence = 0u; sequence < userJobs; ++sequence)
    {
        REQUIRE(hostReceived[sequence].sequenceNumber == sequence);
        REQUIRE(descriptorEquals(hostReceived[sequence], makeDescriptorValue(sequence)));
    }
    REQUIRE(hostReceived[userJobs].kind == graph::JobKind::END_OF_INPUT);

    auto const counters = ring.debugCounters();
    REQUIRE(counters.totalEnqueued == userJobs + 1u);
    REQUIRE(counters.totalImported == userJobs + 1u);
    REQUIRE(counters.published == userJobs + 1u);
    REQUIRE(counters.consumed == userJobs + 1u);
    REQUIRE(counters.occupancy == 0u);
    REQUIRE(counters.closed);
    REQUIRE(counters.importError == graph::IngressError::none);

    std::uint32_t const total32 = static_cast<std::uint32_t>(importedTotal);
    std::cout << (GRAPH_HAS_CUDA ? "CUDA device warp import evidence" : "CPU host-importer path (not GPU evidence)")
              << ": imported=" << total32 << " expected=" << (userJobs + 1u)
              << " ordered_exactly_once=yes close_marker_last=yes consumed_progress_observed=yes" << std::endl;
}

TEST_CASE("ingress import error is reported and leaves the ring usable", "[ingress][roundtrip]")
{
    auto device = makeTestDevice();
    auto ring = Ring::create(device);

    std::array<Descriptor, 3u> jobs{makeDescriptorValue(0u), makeDescriptorValue(1u), makeDescriptorValue(2u)};
    auto const enqueued = ring.tryEnqueue(std::span<Descriptor const>{jobs.data(), jobs.size()});
    REQUIRE(enqueued.status == graph::EnqueueStatus::accepted);
    REQUIRE(enqueued.accepted == 3u);

    // Mapped staging so the same buffer serves the device poll and the host
    // importer check below.
    auto staging = alpaka::onHost::allocMapped<Descriptor>(device, alpaka::Vec<std::uint32_t, 1u>{stagingCapacity});
    auto flagBuffer = alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec<std::uint32_t, 1u>{8u});
    auto* flags = alpaka::onHost::data(flagBuffer);
    for(std::uint32_t i = 0u; i < 8u; ++i)
        hostFlagStore(flags + i, 0u);

    auto queue = device.makeQueue();
    queue.enqueue(makeLaunchCfg(), ErrorPathKernel{}, ring.view(), alpaka::onHost::data(staging), flags);
    alpaka::onHost::wait(queue);

    auto const firstStatus = hostFlagLoad(flags + 0u);
    auto const firstError = hostFlagLoad(flags + 1u);
    auto const firstImported = hostFlagLoad(flags + 2u);
    auto const secondStatus = hostFlagLoad(flags + 3u);
    auto const secondImported = hostFlagLoad(flags + 4u);
    auto const secondError = hostFlagLoad(flags + 5u);

    CHECK(firstStatus == static_cast<std::uint32_t>(graph::ImportStatus::error));
    CHECK(firstError == static_cast<std::uint32_t>(graph::IngressError::invalid_descriptor));
    CHECK(firstImported == 0u);
    // The second poll re-imports the same three descriptors, so the first poll
    // acknowledged nothing.
    CHECK(secondStatus == static_cast<std::uint32_t>(graph::ImportStatus::imported));
    CHECK(secondImported == 3u);
    CHECK(secondError == static_cast<std::uint32_t>(graph::IngressError::none));

    CHECK(ring.consumed() == 3u);
    CHECK(ring.published() == 3u);
    CHECK(ring.occupancy() == 0u);

    // The sticky error code is still observable from the host after the fact,
    // and the ring keeps accepting and importing new work.
    auto const counters = ring.debugCounters();
    CHECK(counters.importError == graph::IngressError::invalid_descriptor);
    CHECK(counters.totalImported == 3u);

    auto const extra = makeDescriptorValue(3u);
    CHECK(ring.tryEnqueue(std::span<Descriptor const>{&extra, 1u}).status == graph::EnqueueStatus::accepted);
    REQUIRE(ring.view().pollAndImportHost(alpaka::onHost::data(staging), stagingCapacity, maxBatch).imported == 1u);
    CHECK(descriptorEquals(*ring.descriptorSlot(3u), extra));
    CHECK(ring.occupancy() == 0u);
}
