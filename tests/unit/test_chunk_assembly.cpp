// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/chunk_assembly.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace
{
    using Elem = std::uint32_t;

    constexpr std::uint32_t invalidStatus = 0xffff'ffffu;

    template<typename T>
    graph::ChunkSlot<T> makeSlot(std::uint32_t slot, std::uint32_t capacity)
    {
        return graph::ChunkSlot<T>{slot, capacity, 0u, graph::PacketHandle{slot, 1u}};
    }

    template<typename T>
    graph::AssemblyStream<T> makeStream(
        std::uint32_t id,
        std::uint32_t capacity,
        std::uint32_t firstSlot,
        std::uint32_t secondSlot,
        std::uint32_t count = 0u)
    {
        graph::AssemblyStream<T> state{};
        state.id = graph::StreamId{id};
        state.capacity = capacity;
        state.count = count;
        state.first = graph::ChunkSlot<T>{firstSlot, capacity, count, graph::PacketHandle{firstSlot, 1u}};
        state.second = graph::ChunkSlot<T>{secondSlot, capacity, 0u, graph::PacketHandle{secondSlot, 1u}};
        // Match a stream that `StreamAssembler::initialize()` has initialized.
        state.version = 1u;
        return state;
    }

    template<typename T>
    std::vector<T> makeStorage(std::uint32_t slotCount, std::uint32_t capacity, T value = T{})
    {
        return std::vector<T>(static_cast<std::size_t>(slotCount) * capacity, value);
    }

    template<typename T, typename T_Device, typename T_Queue>
    auto makeDeviceCopy(T_Device& device, T_Queue& queue, std::vector<T> const& values)
    {
        REQUIRE_FALSE(values.empty());
        auto host = alpaka::onHost::allocHost<T>(values.size());
        for(std::size_t i = 0u; i < values.size(); ++i)
            host[i] = values[i];
        auto deviceBuffer = alpaka::onHost::allocLike(device, host);
        alpaka::onHost::memcpy(queue, deviceBuffer, host);
        alpaka::onHost::wait(queue);
        return deviceBuffer;
    }

    template<typename T, typename T_Queue, typename T_DeviceBuffer>
    std::vector<T> copyToHost(T_Queue& queue, T_DeviceBuffer const& deviceBuffer, std::size_t count)
    {
        auto host = alpaka::onHost::allocHost<T>(count);
        alpaka::onHost::memcpy(queue, host, deviceBuffer);
        alpaka::onHost::wait(queue);
        std::vector<T> result(count);
        for(std::size_t i = 0u; i < count; ++i)
            result[i] = host[i];
        return result;
    }

    /** Write one append for a single destination from every active lane.
     *
     * The loops over rounds make the helper independent of the runtime warp
     * width: a lane with rank `round * width + group.rank` writes one value per
     * round. `destinationGroup()` supplies the ballot-based rank within the
     * destination for the current round.
     */
    template<typename T, typename T_Acc>
    ALPAKA_FN_ACC graph::AppendStatus appendOnce(
        T_Acc const& acc,
        graph::WarpGroup<T_Acc> const& warp,
        graph::StreamAssembler<T> const& assembler,
        graph::StreamId destination,
        std::uint32_t itemCount,
        std::uint32_t valueBase,
        graph::ChunkSlot<T> const* replacements,
        std::uint32_t replacementCount,
        graph::ChunkSlot<T>* publications,
        std::uint32_t publicationCount,
        std::uint32_t ownerToken = 1u)
    {
        graph::StreamReservation<T> reservation;
        graph::AppendRequest request{destination, itemCount};
        auto const status = reservation.tryReserve(
            acc,
            warp,
            ownerToken,
            assembler,
            &request,
            1u,
            replacements,
            replacementCount,
            publications,
            publicationCount);
        if(status != graph::AppendStatus::Ok)
            return status;

        auto const width = warp.workerWidth();
        for(std::uint32_t base = 0u; base < itemCount; base += width)
        {
            bool const active = (base + warp.laneIndex()) < itemCount;
            auto const group = assembler.destinationGroup(warp, destination, active);
            auto const rank = base + group.rank;
            if(active)
            {
                T* pointer = reservation.writePointer(0u, rank);
                if(pointer != nullptr)
                    *pointer = static_cast<T>(valueBase + rank);
            }
        }

        if(!reservation.commit(acc, warp))
            return graph::AppendStatus::NotPending;
        return graph::AppendStatus::Ok;
    }

    template<typename T>
    struct SingleAppendKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            graph::StreamId destination,
            std::uint32_t itemCount,
            std::uint32_t valueBase,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            std::uint32_t* outStatus) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            std::uint32_t status = invalidStatus;
            if(graph::isOneWarpPerBlockLayout(acc))
            {
                status = static_cast<std::uint32_t>(appendOnce<T>(
                    acc,
                    warp,
                    assembler,
                    destination,
                    itemCount,
                    valueBase,
                    replacements,
                    replacementCount,
                    publications,
                    publicationCount));
            }
            if(warp.isLeader())
                outStatus[0] = status;
        }
    };

    template<typename T>
    struct ScenarioResult
    {
        std::vector<graph::AssemblyStream<T>> streams;
        std::vector<T> storage;
        std::vector<graph::ChunkSlot<T>> publications;
        std::uint32_t status{invalidStatus};
    };

    template<typename T, typename T_Device, typename T_Queue, typename T_Executor>
    ScenarioResult<T> runSingleAppend(
        T_Device& device,
        T_Queue& queue,
        T_Executor executor,
        std::uint32_t threads,
        std::uint32_t capacity,
        std::vector<graph::AssemblyStream<T>> const& streams,
        std::vector<T> const& storage,
        std::uint32_t destinationStream,
        std::uint32_t itemCount,
        std::uint32_t valueBase,
        std::vector<graph::ChunkSlot<T>> replacements,
        std::uint32_t publicationCapacity)
    {
        REQUIRE(storage.size() % capacity == 0u);
        auto streamDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, streams);
        auto storageDevice = makeDeviceCopy<T>(device, queue, storage);
        if(replacements.empty())
            replacements.push_back(graph::ChunkSlot<T>{});
        auto replacementDevice = makeDeviceCopy<graph::ChunkSlot<T>>(device, queue, replacements);
        std::vector<graph::ChunkSlot<T>> publicationHost(
            std::max<std::size_t>(1u, publicationCapacity),
            graph::ChunkSlot<T>{});
        auto publicationDevice = makeDeviceCopy<graph::ChunkSlot<T>>(device, queue, publicationHost);
        std::vector<std::uint32_t> statusHost(1u, invalidStatus);
        auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);

        graph::ChunkArena<T> const arena{
            alpaka::onHost::data(storageDevice),
            static_cast<std::uint32_t>(storage.size() / capacity),
            capacity};
        graph::StreamAssembler<T> const assembler{
            alpaka::onHost::data(streamDevice),
            static_cast<std::uint32_t>(streams.size()),
            arena};

        auto const spec = alpaka::onHost::ThreadSpec{1u, threads, executor};
        queue.enqueue(
            spec,
            SingleAppendKernel<T>{},
            assembler,
            streams[destinationStream].id,
            itemCount,
            valueBase,
            alpaka::onHost::data(replacementDevice),
            static_cast<std::uint32_t>(replacements.size()),
            alpaka::onHost::data(publicationDevice),
            publicationCapacity,
            alpaka::onHost::data(statusDevice));
        alpaka::onHost::wait(queue);

        ScenarioResult<T> result;
        result.streams = copyToHost<graph::AssemblyStream<T>>(queue, streamDevice, streams.size());
        result.storage = copyToHost<T>(queue, storageDevice, storage.size());
        result.publications = copyToHost<graph::ChunkSlot<T>>(queue, publicationDevice, publicationHost.size());
        result.status = copyToHost<std::uint32_t>(queue, statusDevice, 1u)[0];
        return result;
    }

    template<typename T>
    struct RepeatedAppendKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            graph::StreamId destination,
            std::uint32_t iterations,
            std::uint32_t itemCount,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            std::uint32_t* outStatus,
            std::uint32_t* outPublicationCount,
            std::uint32_t* outMaxOccupancy) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            std::uint32_t status = invalidStatus;
            std::uint32_t publicationCursor = 0u;
            std::uint32_t maxOccupancy = 0u;
            if(graph::isOneWarpPerBlockLayout(acc))
            {
                auto const streamIndex = assembler.find(destination);
                auto const capacity = assembler.stream(streamIndex).capacity;
                std::uint32_t replacementCursor = 0u;
                std::uint32_t valueBase = 0u;
                status = 0u;
                for(std::uint32_t iteration = 0u; iteration < iterations; ++iteration)
                {
                    graph::StreamReservation<T> reservation;
                    graph::AppendRequest request{destination, itemCount};
                    auto const reserveStatus = reservation.tryReserve(
                        acc,
                        warp,
                        1u,
                        assembler,
                        &request,
                        1u,
                        replacements + replacementCursor,
                        replacementCount - replacementCursor,
                        publications + publicationCursor,
                        publicationCount - publicationCursor);
                    if(reserveStatus != graph::AppendStatus::Ok)
                    {
                        status = static_cast<std::uint32_t>(reserveStatus);
                        break;
                    }
                    auto const occupancy = reservation.baseCount(0u) + reservation.groupSize(0u);
                    maxOccupancy = std::max(maxOccupancy, occupancy);
                    if(occupancy > graph::StreamAssembler<T>::maxAppendOccupancy(capacity))
                    {
                        status = 0x0cc0'0001u;
                        break;
                    }
                    auto const width = warp.workerWidth();
                    for(std::uint32_t base = 0u; base < itemCount; base += width)
                    {
                        bool const active = (base + warp.laneIndex()) < itemCount;
                        auto const group = assembler.destinationGroup(warp, destination, active);
                        auto const rank = base + group.rank;
                        if(active)
                        {
                            T* pointer = reservation.writePointer(0u, rank);
                            if(pointer != nullptr)
                                *pointer = static_cast<T>(valueBase + rank);
                        }
                    }
                    if(!reservation.commit(acc, warp))
                    {
                        status = 0x0cc0'0002u;
                        break;
                    }
                    valueBase += itemCount;
                    if(reservation.rotates(0u))
                    {
                        replacementCursor += 1u;
                        publicationCursor += 1u;
                    }
                }
            }
            if(warp.isLeader())
            {
                outStatus[0] = status;
                outPublicationCount[0] = publicationCursor;
                outMaxOccupancy[0] = maxOccupancy;
            }
        }
    };

    template<typename T>
    struct RandomizedAppendKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            graph::StreamId destination,
            std::uint32_t iterations,
            std::uint32_t seed,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            std::uint32_t* outStatus,
            std::uint32_t* outPublicationCount,
            std::uint32_t* outMaxOccupancy,
            std::uint32_t* outItems) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            std::uint32_t status = invalidStatus;
            std::uint32_t publicationCursor = 0u;
            std::uint32_t maxOccupancy = 0u;
            std::uint32_t items = 0u;
            if(graph::isOneWarpPerBlockLayout(acc))
            {
                auto const streamIndex = assembler.find(destination);
                auto const capacity = assembler.stream(streamIndex).capacity;
                std::uint32_t replacementCursor = 0u;
                std::uint32_t valueBase = 0u;
                std::uint32_t random = seed;
                status = 0u;
                for(std::uint32_t iteration = 0u; iteration < iterations; ++iteration)
                {
                    random = random * 1'664'525u + 1'013'904'223u;
                    std::uint32_t const itemCount = random % (capacity + 1u);
                    graph::StreamReservation<T> reservation;
                    graph::AppendRequest request{destination, itemCount};
                    auto const reserveStatus = reservation.tryReserve(
                        acc,
                        warp,
                        1u,
                        assembler,
                        &request,
                        1u,
                        replacements + replacementCursor,
                        replacementCount - replacementCursor,
                        publications + publicationCursor,
                        publicationCount - publicationCursor);
                    if(reserveStatus != graph::AppendStatus::Ok)
                    {
                        status = static_cast<std::uint32_t>(reserveStatus);
                        break;
                    }
                    auto const occupancy = reservation.baseCount(0u) + reservation.groupSize(0u);
                    maxOccupancy = std::max(maxOccupancy, occupancy);
                    if(occupancy > graph::StreamAssembler<T>::maxAppendOccupancy(capacity))
                    {
                        status = 0x0cc0'0003u;
                        break;
                    }
                    auto const width = warp.workerWidth();
                    for(std::uint32_t base = 0u; base < itemCount; base += width)
                    {
                        bool const active = (base + warp.laneIndex()) < itemCount;
                        auto const group = assembler.destinationGroup(warp, destination, active);
                        auto const rank = base + group.rank;
                        if(active)
                        {
                            T* pointer = reservation.writePointer(0u, rank);
                            if(pointer != nullptr)
                                *pointer = static_cast<T>(valueBase + rank);
                        }
                    }
                    if(!reservation.commit(acc, warp))
                    {
                        status = 0x0cc0'0004u;
                        break;
                    }
                    valueBase += itemCount;
                    items += itemCount;
                    if(reservation.rotates(0u))
                    {
                        replacementCursor += 1u;
                        publicationCursor += 1u;
                    }
                }
            }
            if(warp.isLeader())
            {
                outStatus[0] = status;
                outPublicationCount[0] = publicationCursor;
                outMaxOccupancy[0] = maxOccupancy;
                outItems[0] = items;
            }
        }
    };

    template<typename T, typename T_Device, typename T_Queue, typename T_Executor>
    ScenarioResult<T> runRepeatedAppend(
        T_Device& device,
        T_Queue& queue,
        T_Executor executor,
        std::uint32_t threads,
        std::uint32_t capacity,
        std::uint32_t iterations,
        std::uint32_t itemCount,
        std::uint32_t seed,
        bool randomized,
        std::uint32_t& publicationCount,
        std::uint32_t& maxOccupancy,
        std::uint32_t& itemTotal)
    {
        auto streams = std::vector<graph::AssemblyStream<T>>{makeStream<T>(1u, capacity, 0u, 1u)};
        auto storage = makeStorage<T>(iterations + 8u, capacity);
        auto replacements = std::vector<graph::ChunkSlot<T>>{};
        for(std::uint32_t i = 0u; i < iterations + 2u; ++i)
            replacements.push_back(makeSlot<T>(2u + i, capacity));
        std::vector<graph::ChunkSlot<T>> publicationHost(iterations + 2u, graph::ChunkSlot<T>{});

        auto streamDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, streams);
        auto storageDevice = makeDeviceCopy<T>(device, queue, storage);
        auto replacementDevice = makeDeviceCopy<graph::ChunkSlot<T>>(device, queue, replacements);
        auto publicationDevice = makeDeviceCopy<graph::ChunkSlot<T>>(device, queue, publicationHost);
        std::vector<std::uint32_t> scalarHost(4u, invalidStatus);
        auto scalarDevice = makeDeviceCopy<std::uint32_t>(device, queue, scalarHost);

        graph::ChunkArena<T> const arena{
            alpaka::onHost::data(storageDevice),
            static_cast<std::uint32_t>(storage.size() / capacity),
            capacity};
        graph::StreamAssembler<T> const assembler{
            alpaka::onHost::data(streamDevice),
            static_cast<std::uint32_t>(streams.size()),
            arena};

        auto const spec = alpaka::onHost::ThreadSpec{1u, threads, executor};
        if(randomized)
        {
            queue.enqueue(
                spec,
                RandomizedAppendKernel<T>{},
                assembler,
                streams[0].id,
                iterations,
                seed,
                alpaka::onHost::data(replacementDevice),
                static_cast<std::uint32_t>(replacements.size()),
                alpaka::onHost::data(publicationDevice),
                static_cast<std::uint32_t>(publicationHost.size()),
                alpaka::onHost::data(scalarDevice),
                alpaka::onHost::data(scalarDevice) + 1u,
                alpaka::onHost::data(scalarDevice) + 2u,
                alpaka::onHost::data(scalarDevice) + 3u);
        }
        else
        {
            queue.enqueue(
                spec,
                RepeatedAppendKernel<T>{},
                assembler,
                streams[0].id,
                iterations,
                itemCount,
                alpaka::onHost::data(replacementDevice),
                static_cast<std::uint32_t>(replacements.size()),
                alpaka::onHost::data(publicationDevice),
                static_cast<std::uint32_t>(publicationHost.size()),
                alpaka::onHost::data(scalarDevice),
                alpaka::onHost::data(scalarDevice) + 1u,
                alpaka::onHost::data(scalarDevice) + 2u);
        }
        alpaka::onHost::wait(queue);

        ScenarioResult<T> result;
        result.streams = copyToHost<graph::AssemblyStream<T>>(queue, streamDevice, streams.size());
        result.storage = copyToHost<T>(queue, storageDevice, storage.size());
        result.publications = copyToHost<graph::ChunkSlot<T>>(queue, publicationDevice, publicationHost.size());
        auto const scalars = copyToHost<std::uint32_t>(queue, scalarDevice, 4u);
        result.status = scalars[0];
        publicationCount = scalars[1];
        maxOccupancy = scalars[2];
        itemTotal = randomized ? scalars[3] : 0u;
        return result;
    }

    /** Verify that published chunks and the final partial chunk contain the
     *  written values in order.
     */
    void checkFlattened(
        std::vector<graph::ChunkSlot<Elem>> const& publications,
        std::uint32_t publicationCount,
        std::vector<Elem> const& storage,
        graph::AssemblyStream<Elem> const& stream,
        std::uint32_t capacity,
        std::uint32_t expectedItems)
    {
        std::vector<Elem> flattened;
        for(std::uint32_t i = 0u; i < publicationCount; ++i)
        {
            auto const& chunk = publications[i];
            REQUIRE(chunk.slot != graph::invalidChunkSlot);
            REQUIRE(chunk.count == capacity);
            for(std::uint32_t j = 0u; j < capacity; ++j)
                flattened.push_back(storage[static_cast<std::size_t>(chunk.slot) * capacity + j]);
        }
        REQUIRE(stream.count == expectedItems % capacity);
        if(stream.count != 0u)
        {
            REQUIRE(stream.first.slot != graph::invalidChunkSlot);
            for(std::uint32_t j = 0u; j < stream.count; ++j)
                flattened.push_back(storage[static_cast<std::size_t>(stream.first.slot) * capacity + j]);
        }
        REQUIRE(flattened.size() == expectedItems);
        for(std::uint32_t i = 0u; i < expectedItems; ++i)
            CHECK(flattened[i] == i);
    }

    template<typename T>
    struct FlushKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            std::uint32_t streamIndex,
            graph::ChunkSlot<T> replacement,
            graph::ChunkSlot<T>* outPublished,
            std::uint32_t* outStatus) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            graph::ChunkSlot<T> published{};
            auto const status = assembler.flushPartial(acc, warp, streamIndex, 1u, replacement, published);
            if(warp.isLeader())
            {
                outPublished[0] = published;
                outStatus[0] = static_cast<std::uint32_t>(status);
            }
        }
    };

    template<typename T>
    struct ReservationProbeKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            graph::StreamId destination,
            std::uint32_t itemCount,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            T const* unused,
            graph::AssemblyStream<T>* outBefore,
            graph::AssemblyStream<T>* outAfterFailures,
            graph::AssemblyStream<T>* outAfterCancel,
            std::uint32_t* outStatuses,
            std::uint32_t* outFlags) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            (void) unused;
            if(!graph::isOneWarpPerBlockLayout(acc))
            {
                if(warp.isLeader())
                    outStatuses[0] = invalidStatus;
                return;
            }
            auto const streamIndex = assembler.find(destination);
            if(warp.isLeader())
                outBefore[0] = assembler.stream(streamIndex);

            // 1. Rotation is required but no replacement is offered.
            {
                graph::StreamReservation<T> reservation;
                graph::AppendRequest request{destination, itemCount};
                auto const status = reservation.tryReserve(
                    acc,
                    warp,
                    1u,
                    assembler,
                    &request,
                    1u,
                    nullptr,
                    0u,
                    publications,
                    publicationCount);
                if(warp.isLeader())
                    outStatuses[0] = static_cast<std::uint32_t>(status);
            }
            if(warp.isLeader())
                outAfterFailures[0] = assembler.stream(streamIndex);

            // 2. Replacement exists but the publication slot is missing.
            {
                graph::StreamReservation<T> reservation;
                graph::AppendRequest request{destination, itemCount};
                auto const status = reservation.tryReserve(
                    acc,
                    warp,
                    1u,
                    assembler,
                    &request,
                    1u,
                    replacements,
                    replacementCount,
                    nullptr,
                    0u);
                if(warp.isLeader())
                    outStatuses[1] = static_cast<std::uint32_t>(status);
            }
            if(warp.isLeader())
                outAfterFailures[1] = assembler.stream(streamIndex);

            // 3. A successful reservation is postponed, then committed.
            {
                graph::StreamReservation<T> reservation;
                graph::AppendRequest request{destination, itemCount};
                auto const status = reservation.tryReserve(
                    acc,
                    warp,
                    1u,
                    assembler,
                    &request,
                    1u,
                    replacements,
                    replacementCount,
                    publications,
                    publicationCount);
                if(warp.isLeader())
                {
                    outStatuses[2] = static_cast<std::uint32_t>(status);
                    outFlags[0] = reservation.pending() ? 1u : 0u;
                }
                if(status == graph::AppendStatus::Ok)
                {
                    T* pointer = reservation.writePointer(0u, 0u);
                    if(pointer != nullptr)
                        *pointer = static_cast<T>(777u);
                    auto const committed = reservation.commit(acc, warp);
                    if(warp.isLeader())
                    {
                        outFlags[1] = committed ? 1u : 0u;
                        outFlags[2] = reservation.retireInput(warp) ? 1u : 0u;
                        outFlags[3] = reservation.retireInput(warp) ? 1u : 0u;
                        outFlags[4] = reservation.doubleRetireDetected() ? 1u : 0u;
                    }
                }
            }

            // 4. A successful reservation that is cancelled leaves state unchanged.
            if(warp.isLeader())
                outAfterCancel[0] = assembler.stream(streamIndex);
            {
                graph::StreamReservation<T> reservation;
                graph::AppendRequest request{destination, itemCount};
                auto const status = reservation.tryReserve(
                    acc,
                    warp,
                    1u,
                    assembler,
                    &request,
                    1u,
                    replacements,
                    replacementCount,
                    publications,
                    publicationCount);
                if(status == graph::AppendStatus::Ok)
                    reservation.cancel(acc, warp);
                if(warp.isLeader())
                    outStatuses[3] = static_cast<std::uint32_t>(status);
            }
            if(warp.isLeader())
                outAfterCancel[1] = assembler.stream(streamIndex);
        }
    };

    template<typename T>
    struct CompetingWriterKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            graph::StreamId destination,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            std::uint32_t* outIndices,
            std::uint32_t* outStatuses) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            auto const blockIdx = static_cast<std::uint32_t>(
                acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)[0]);
            std::uint32_t status = invalidStatus;
            std::uint32_t physicalIndex = graph::invalidChunkSlot;
            if(graph::isOneWarpPerBlockLayout(acc))
            {
                for(std::uint32_t attempt = 0u; attempt < 200000u; ++attempt)
                {
                    graph::StreamReservation<T> reservation;
                    graph::AppendRequest request{destination, 1u};
                    auto const reserveStatus = reservation.tryReserve(
                        acc,
                        warp,
                        blockIdx + 1u,
                        assembler,
                        &request,
                        1u,
                        replacements,
                        replacementCount,
                        publications,
                        publicationCount);
                    if(reserveStatus == graph::AppendStatus::StreamBusy)
                        continue;
                    if(reserveStatus != graph::AppendStatus::Ok)
                    {
                        status = static_cast<std::uint32_t>(reserveStatus);
                        break;
                    }
                    T* pointer = reservation.writePointer(0u, 0u, &physicalIndex);
                    if(pointer != nullptr)
                        *pointer = static_cast<T>(blockIdx);
                    if(!reservation.commit(acc, warp))
                    {
                        status = 0x0cc0'0005u;
                        break;
                    }
                    status = static_cast<std::uint32_t>(graph::AppendStatus::Ok);
                    break;
                }
            }
            if(warp.isLeader())
            {
                outIndices[blockIdx] = physicalIndex;
                outStatuses[blockIdx] = status;
            }
        }
    };

    struct MultiDestinationKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<Elem> assembler,
            graph::StreamId first,
            graph::StreamId second,
            std::uint32_t* outRank,
            std::uint32_t* outSize,
            std::uint32_t* outStatus) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            bool const isFirst = (lane % 2u) == 0u;
            auto const destination = isFirst ? first : second;
            auto const group = assembler.destinationGroup(warp, destination, true);
            outRank[lane] = group.rank;
            outSize[lane] = group.size;

            std::uint32_t const firstCount = (width + 1u) / 2u;
            std::uint32_t const secondCount = width / 2u;
            graph::StreamReservation<Elem> reservation;
            graph::AppendRequest requests[2]{{first, firstCount}, {second, secondCount}};
            auto const status
                = reservation.tryReserve(acc, warp, 1u, assembler, requests, 2u, nullptr, 0u, nullptr, 0u);
            if(status != graph::AppendStatus::Ok)
            {
                if(warp.isLeader())
                    outStatus[0] = static_cast<std::uint32_t>(status);
                return;
            }
            std::uint32_t const target = isFirst ? 0u : 1u;
            Elem* pointer = reservation.writePointer(target, group.rank);
            if(pointer != nullptr)
                *pointer = isFirst ? (100u + group.rank) : (200u + group.rank);
            reservation.commit(acc, warp);
            if(warp.isLeader())
                outStatus[0] = static_cast<std::uint32_t>(graph::AppendStatus::Ok);
        }
    };

} // namespace

TEST_CASE("chunk_assembly all particles target one bin", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 128u;
    auto const streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto const storage = makeStorage<Elem>(4u, capacity);

    auto const result = runSingleAppend<
        Elem>(device, queue, alpaka::exec::cpuSerial, 1u, capacity, streams, storage, 0u, 5u, 0u, {}, 0u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(result.streams[0].count == 5u);
    REQUIRE(result.streams[0].version == 2u);
    for(std::uint32_t i = 0u; i < 5u; ++i)
        CHECK(result.storage[i] == i);
}

TEST_CASE("chunk_assembly both bins receive data", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 16u;
    auto streams = std::vector<graph::AssemblyStream<Elem>>{
        makeStream<Elem>(11u, capacity, 0u, 1u),
        makeStream<Elem>(22u, capacity, 2u, 3u)};
    auto storage = makeStorage<Elem>(6u, capacity);

    // Stream 11 is assembled first.
    {
        auto const result = runSingleAppend<
            Elem>(device, queue, alpaka::exec::cpuSerial, 1u, capacity, streams, storage, 0u, 5u, 100u, {}, 0u);
        REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
        REQUIRE(result.streams[0].count == 5u);
        REQUIRE(result.streams[1].count == 0u);
        streams = result.streams;
        storage = result.storage;
    }

    // Stream 22 is assembled next. Its state is independent even though the
    // numeric bin id is shared with nothing else.
    {
        auto const result = runSingleAppend<
            Elem>(device, queue, alpaka::exec::cpuSerial, 1u, capacity, streams, storage, 1u, 7u, 200u, {}, 0u);
        REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
        REQUIRE(result.streams[0].count == 5u);
        REQUIRE(result.streams[1].count == 7u);
        for(std::uint32_t i = 0u; i < 5u; ++i)
            CHECK(result.storage[i] == 100u + i);
        for(std::uint32_t i = 0u; i < 7u; ++i)
            CHECK(result.storage[2u * capacity + i] == 200u + i);
    }
}

TEST_CASE("chunk_assembly counter four plus input 128 yields full packet and four carried", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 128u;
    auto streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u, 4u)};
    auto storage = makeStorage<Elem>(6u, capacity);
    for(std::uint32_t i = 0u; i < 4u; ++i)
        storage[i] = 1000u + i;

    auto const result = runSingleAppend<Elem>(
        device,
        queue,
        alpaka::exec::cpuSerial,
        1u,
        capacity,
        streams,
        storage,
        0u,
        128u,
        0u,
        {makeSlot<Elem>(2u, capacity)},
        1u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(result.publications[0].slot == 0u);
    REQUIRE(result.publications[0].count == 128u);
    for(std::uint32_t i = 0u; i < 4u; ++i)
        CHECK(result.storage[i] == 1000u + i);
    for(std::uint32_t i = 4u; i < 128u; ++i)
        CHECK(result.storage[i] == i - 4u);

    // The four carried entries were written into B and B was promoted to A.
    REQUIRE(result.streams[0].count == 4u);
    REQUIRE(result.streams[0].first.slot == 1u);
    for(std::uint32_t i = 0u; i < 4u; ++i)
        CHECK(result.storage[capacity + i] == 124u + i);
}

TEST_CASE("chunk_assembly exactly full rotates including the full chunk", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    auto const streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto const storage = makeStorage<Elem>(4u, capacity);

    auto const result = runSingleAppend<Elem>(
        device,
        queue,
        alpaka::exec::cpuSerial,
        1u,
        capacity,
        streams,
        storage,
        0u,
        8u,
        0u,
        {makeSlot<Elem>(2u, capacity)},
        1u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(result.publications[0].count == 8u);
    REQUIRE(result.streams[0].count == 0u);
    REQUIRE(result.streams[0].first.slot == 1u);
    for(std::uint32_t i = 0u; i < 8u; ++i)
        CHECK(result.storage[i] == i);
}

TEST_CASE("chunk_assembly partial input stays in A", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    auto const streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto const storage = makeStorage<Elem>(4u, capacity);

    auto const result = runSingleAppend<
        Elem>(device, queue, alpaka::exec::cpuSerial, 1u, capacity, streams, storage, 0u, 3u, 0u, {}, 0u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(result.streams[0].count == 3u);
    REQUIRE(result.streams[0].version == 2u);
    REQUIRE(result.storage[0] == 0u);
    REQUIRE(result.storage[1] == 1u);
    REQUIRE(result.storage[2] == 2u);
}

TEST_CASE("chunk_assembly capacity one rotates every append", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 1u;
    auto const streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto const storage = makeStorage<Elem>(4u, capacity);

    auto const result = runSingleAppend<Elem>(
        device,
        queue,
        alpaka::exec::cpuSerial,
        1u,
        capacity,
        streams,
        storage,
        0u,
        1u,
        77u,
        {makeSlot<Elem>(2u, capacity)},
        1u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(result.publications[0].count == 1u);
    REQUIRE(result.storage[result.publications[0].slot] == 77u);
    REQUIRE(result.streams[0].count == 0u);
}

TEST_CASE("chunk_assembly non-warp-multiple capacity", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 100u;
    auto streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u, 4u)};
    auto storage = makeStorage<Elem>(6u, capacity);
    for(std::uint32_t i = 0u; i < 4u; ++i)
        storage[i] = 500u + i;

    auto const result = runSingleAppend<Elem>(
        device,
        queue,
        alpaka::exec::cpuSerial,
        1u,
        capacity,
        streams,
        storage,
        0u,
        100u,
        0u,
        {makeSlot<Elem>(2u, capacity)},
        1u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(result.publications[0].count == 100u);
    REQUIRE(result.streams[0].count == 4u);
    REQUIRE(result.streams[0].first.slot == 1u);
    for(std::uint32_t i = 0u; i < 4u; ++i)
        CHECK(result.storage[capacity + i] == 96u + i);
}

TEST_CASE("chunk_assembly safe rotation under repeated appends", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    constexpr std::uint32_t iterations = 37u;
    std::uint32_t publicationCount = 0u;
    std::uint32_t maxOccupancy = 0u;
    std::uint32_t itemTotal = 0u;
    auto const result = runRepeatedAppend<Elem>(
        device,
        queue,
        alpaka::exec::cpuSerial,
        1u,
        capacity,
        iterations,
        1u,
        0u,
        false,
        publicationCount,
        maxOccupancy,
        itemTotal);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    REQUIRE(publicationCount == iterations / capacity);
    REQUIRE(maxOccupancy <= 2u * capacity - 1u);
    checkFlattened(result.publications, publicationCount, result.storage, result.streams[0], capacity, iterations);
}

TEST_CASE("chunk_assembly reservation failure leaves state unchanged", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    std::vector<graph::AssemblyStream<Elem>> streams{
        makeStream<Elem>(1u, capacity, 0u, 1u, capacity - 1u),
        makeStream<Elem>(2u, capacity, 2u, 3u, capacity - 1u)};
    auto storage = makeStorage<Elem>(8u, capacity);
    std::vector<graph::ChunkSlot<Elem>> replacements{makeSlot<Elem>(4u, capacity)};
    std::vector<graph::ChunkSlot<Elem>> publications(2u, graph::ChunkSlot<Elem>{});
    std::vector<Elem> unused(1u, 0u);

    auto streamDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, streams);
    auto storageDevice = makeDeviceCopy<Elem>(device, queue, storage);
    auto replacementDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, replacements);
    auto publicationDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, publications);
    auto unusedDevice = makeDeviceCopy<Elem>(device, queue, unused);
    std::vector<graph::AssemblyStream<Elem>> beforeHost(1u);
    auto beforeDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, beforeHost);
    std::vector<graph::AssemblyStream<Elem>> afterHost(2u);
    auto afterDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, afterHost);
    std::vector<graph::AssemblyStream<Elem>> cancelHost(2u);
    auto cancelDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, cancelHost);
    std::vector<std::uint32_t> statusHost(4u, invalidStatus);
    auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);
    std::vector<std::uint32_t> flagHost(5u, 0u);
    auto flagDevice = makeDeviceCopy<std::uint32_t>(device, queue, flagHost);

    graph::ChunkArena<Elem> const arena{
        alpaka::onHost::data(storageDevice),
        static_cast<std::uint32_t>(storage.size() / capacity),
        capacity};
    graph::StreamAssembler<Elem> const assembler{
        alpaka::onHost::data(streamDevice),
        static_cast<std::uint32_t>(streams.size()),
        arena};

    auto const spec = alpaka::onHost::ThreadSpec{1u, 1u, alpaka::exec::cpuSerial};
    queue.enqueue(
        spec,
        ReservationProbeKernel<Elem>{},
        assembler,
        streams[0].id,
        1u,
        alpaka::onHost::data(replacementDevice),
        static_cast<std::uint32_t>(replacements.size()),
        alpaka::onHost::data(publicationDevice),
        static_cast<std::uint32_t>(publications.size()),
        alpaka::onHost::data(unusedDevice),
        alpaka::onHost::data(beforeDevice),
        alpaka::onHost::data(afterDevice),
        alpaka::onHost::data(cancelDevice),
        alpaka::onHost::data(statusDevice),
        alpaka::onHost::data(flagDevice));
    alpaka::onHost::wait(queue);

    auto const before = copyToHost<graph::AssemblyStream<Elem>>(queue, beforeDevice, 1u)[0];
    auto const after = copyToHost<graph::AssemblyStream<Elem>>(queue, afterDevice, 2u);
    auto const afterCancel = copyToHost<graph::AssemblyStream<Elem>>(queue, cancelDevice, 2u);
    auto const statuses = copyToHost<std::uint32_t>(queue, statusDevice, 4u);
    auto const flags = copyToHost<std::uint32_t>(queue, flagDevice, 5u);

    CHECK(statuses[0] == static_cast<std::uint32_t>(graph::AppendStatus::NotEnoughReplacements));
    CHECK(statuses[1] == static_cast<std::uint32_t>(graph::AppendStatus::NotEnoughPublications));
    CHECK(statuses[2] == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    CHECK(statuses[3] == static_cast<std::uint32_t>(graph::AppendStatus::Ok));

    CHECK(after[0].count == before.count);
    CHECK(after[0].version == before.version);
    CHECK(after[0].first == before.first);
    CHECK(after[0].second == before.second);
    CHECK(after[0].claim == 0u);
    CHECK(after[1].count == before.count);
    CHECK(after[1].version == before.version);
    CHECK(after[1].claim == 0u);

    // The explicitly postponed reservation was committed later.
    CHECK(flags[0] == 1u);
    CHECK(flags[1] == 1u);
    CHECK(flags[2] == 1u);
    CHECK(flags[3] == 0u);
    CHECK(flags[4] == 1u);

    // The cancelled reservation left its stream untouched.
    CHECK(afterCancel[1].count == afterCancel[0].count);
    CHECK(afterCancel[1].version == afterCancel[0].version);
    CHECK(afterCancel[1].first == afterCancel[0].first);
    CHECK(afterCancel[1].second == afterCancel[0].second);
    CHECK(afterCancel[1].claim == 0u);
}

TEST_CASE("chunk_assembly flush partial publishes the real count", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;

    auto runFlush
        = [&](graph::AssemblyStream<Elem> const& stream, graph::ChunkSlot<Elem> replacement, std::vector<Elem> storage)
    {
        std::vector<graph::AssemblyStream<Elem>> streams{stream};
        auto streamDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, streams);
        auto storageDevice = makeDeviceCopy<Elem>(device, queue, storage);
        std::vector<graph::ChunkSlot<Elem>> publishedHost(1u, graph::ChunkSlot<Elem>{});
        auto publishedDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, publishedHost);
        std::vector<std::uint32_t> statusHost(1u, invalidStatus);
        auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);

        graph::ChunkArena<Elem> const arena{
            alpaka::onHost::data(storageDevice),
            static_cast<std::uint32_t>(storage.size() / capacity),
            capacity};
        graph::StreamAssembler<Elem> const assembler{alpaka::onHost::data(streamDevice), 1u, arena};
        auto const spec = alpaka::onHost::ThreadSpec{1u, 1u, alpaka::exec::cpuSerial};
        queue.enqueue(
            spec,
            FlushKernel<Elem>{},
            assembler,
            0u,
            replacement,
            alpaka::onHost::data(publishedDevice),
            alpaka::onHost::data(statusDevice));
        alpaka::onHost::wait(queue);

        struct FlushResult
        {
            graph::AssemblyStream<Elem> stream;
            graph::ChunkSlot<Elem> published;
            std::uint32_t status;
            std::vector<Elem> storage;
        };
        return FlushResult{
            copyToHost<graph::AssemblyStream<Elem>>(queue, streamDevice, 1u)[0],
            copyToHost<graph::ChunkSlot<Elem>>(queue, publishedDevice, 1u)[0],
            copyToHost<std::uint32_t>(queue, statusDevice, 1u)[0],
            copyToHost<Elem>(queue, storageDevice, storage.size())};
    };

    // c == 0 publishes nothing and does not consume the replacement.
    {
        auto result = runFlush(
            makeStream<Elem>(1u, capacity, 0u, 1u),
            makeSlot<Elem>(2u, capacity),
            makeStorage<Elem>(4u, capacity));
        CHECK(result.status == static_cast<std::uint32_t>(graph::FlushStatus::NothingToPublish));
        CHECK(result.published.slot == graph::invalidChunkSlot);
        CHECK(result.stream.count == 0u);
        CHECK(result.stream.first.slot == 0u);
        CHECK(result.stream.version == 1u);
    }

    // c == 3 publishes A with the real count and attaches a fresh A.
    {
        auto storage = makeStorage<Elem>(4u, capacity);
        for(std::uint32_t i = 0u; i < 3u; ++i)
            storage[i] = 40u + i;
        auto result = runFlush(makeStream<Elem>(1u, capacity, 0u, 1u, 3u), makeSlot<Elem>(2u, capacity), storage);
        CHECK(result.status == static_cast<std::uint32_t>(graph::FlushStatus::Published));
        CHECK(result.published.slot == 0u);
        CHECK(result.published.count == 3u);
        CHECK(result.stream.count == 0u);
        CHECK(result.stream.first.slot == 2u);
        CHECK(result.stream.first.count == 0u);
    }

    // A terminal flush without a replacement drains the stream.
    {
        auto storage = makeStorage<Elem>(4u, capacity);
        for(std::uint32_t i = 0u; i < 3u; ++i)
            storage[i] = 50u + i;
        auto result = runFlush(makeStream<Elem>(1u, capacity, 0u, 1u, 3u), graph::ChunkSlot<Elem>{}, storage);
        CHECK(result.status == static_cast<std::uint32_t>(graph::FlushStatus::Published));
        CHECK(result.published.count == 3u);
        CHECK(result.stream.drained);
        CHECK(result.stream.first.slot == graph::invalidChunkSlot);
    }
}

TEST_CASE("chunk_assembly grouped rank uses ballot with width fallback", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    auto const streams = std::vector<graph::AssemblyStream<Elem>>{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto const storage = makeStorage<Elem>(4u, capacity);

    auto const result = runSingleAppend<
        Elem>(device, queue, alpaka::exec::cpuSerial, 1u, capacity, streams, storage, 0u, 3u, 0u, {}, 0u);

    REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    // Width one: each round has a single active lane, so rank 0 in every round.
    REQUIRE(result.streams[0].count == 3u);
    CHECK(result.storage[0] == 0u);
    CHECK(result.storage[1] == 1u);
    CHECK(result.storage[2] == 2u);
}

TEST_CASE("chunk_assembly competing writers get disjoint ranges", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    constexpr std::uint32_t blocks = capacity;
    std::vector<graph::AssemblyStream<Elem>> streams{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto storage = makeStorage<Elem>(4u, capacity);
    std::vector<graph::ChunkSlot<Elem>> replacements{makeSlot<Elem>(2u, capacity)};
    std::vector<graph::ChunkSlot<Elem>> publications(1u, graph::ChunkSlot<Elem>{});

    auto streamDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, streams);
    auto storageDevice = makeDeviceCopy<Elem>(device, queue, storage);
    auto replacementDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, replacements);
    auto publicationDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, publications);
    std::vector<std::uint32_t> indexHost(blocks, graph::invalidChunkSlot);
    auto indexDevice = makeDeviceCopy<std::uint32_t>(device, queue, indexHost);
    std::vector<std::uint32_t> statusHost(blocks, invalidStatus);
    auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);

    graph::ChunkArena<Elem> const arena{
        alpaka::onHost::data(storageDevice),
        static_cast<std::uint32_t>(storage.size() / capacity),
        capacity};
    graph::StreamAssembler<Elem> const assembler{
        alpaka::onHost::data(streamDevice),
        static_cast<std::uint32_t>(streams.size()),
        arena};

    auto const spec = alpaka::onHost::ThreadSpec{blocks, 1u, alpaka::exec::cpuSerial};
    queue.enqueue(
        spec,
        CompetingWriterKernel<Elem>{},
        assembler,
        streams[0].id,
        alpaka::onHost::data(replacementDevice),
        static_cast<std::uint32_t>(replacements.size()),
        alpaka::onHost::data(publicationDevice),
        static_cast<std::uint32_t>(publications.size()),
        alpaka::onHost::data(indexDevice),
        alpaka::onHost::data(statusDevice));
    alpaka::onHost::wait(queue);

    auto const indices = copyToHost<std::uint32_t>(queue, indexDevice, blocks);
    auto const statuses = copyToHost<std::uint32_t>(queue, statusDevice, blocks);
    auto const published = copyToHost<graph::ChunkSlot<Elem>>(queue, publicationDevice, publications.size());
    auto const downloaded = copyToHost<Elem>(queue, storageDevice, storage.size());

    for(std::uint32_t i = 0u; i < blocks; ++i)
        CHECK(statuses[i] == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    std::vector<std::uint32_t> sorted = indices;
    std::sort(sorted.begin(), sorted.end());
    for(std::uint32_t i = 0u; i < blocks; ++i)
        REQUIRE(sorted[i] == i);

    REQUIRE(published[0].slot == 0u);
    REQUIRE(published[0].count == capacity);
    std::vector<Elem> publishedValues(capacity);
    for(std::uint32_t i = 0u; i < capacity; ++i)
        publishedValues[i] = downloaded[static_cast<std::size_t>(published[0].slot) * capacity + i];
    std::sort(publishedValues.begin(), publishedValues.end());
    for(std::uint32_t i = 0u; i < capacity; ++i)
        REQUIRE(publishedValues[i] == i);
}

TEST_CASE("chunk_assembly two Y minus one bound over randomized appends", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    for(std::uint32_t capacity : {1u, 2u, 100u, 128u})
    {
        std::uint32_t publicationCount = 0u;
        std::uint32_t maxOccupancy = 0u;
        std::uint32_t itemTotal = 0u;
        auto const result = runRepeatedAppend<Elem>(
            device,
            queue,
            alpaka::exec::cpuSerial,
            1u,
            capacity,
            40u,
            1u,
            12345u,
            true,
            publicationCount,
            maxOccupancy,
            itemTotal);

        INFO("capacity=" << capacity);
        REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
        CHECK(maxOccupancy <= 2u * capacity - 1u);
        checkFlattened(result.publications, publicationCount, result.storage, result.streams[0], capacity, itemTotal);
    }
}

#if GRAPH_HAS_CUDA
TEST_CASE("chunk_assembly competing writers on CUDA warps get disjoint ranges", "[chunk_assembly][cuda]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto const executor = alpaka::exec::gpuCuda;

    constexpr std::uint32_t capacity = 8u;
    constexpr std::uint32_t blocks = capacity;
    constexpr std::uint32_t threads = 32u;
    std::vector<graph::AssemblyStream<Elem>> streams{makeStream<Elem>(1u, capacity, 0u, 1u)};
    auto storage = makeStorage<Elem>(4u, capacity);
    std::vector<graph::ChunkSlot<Elem>> replacements{makeSlot<Elem>(2u, capacity)};
    std::vector<graph::ChunkSlot<Elem>> publications(1u, graph::ChunkSlot<Elem>{});

    auto streamDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, streams);
    auto storageDevice = makeDeviceCopy<Elem>(device, queue, storage);
    auto replacementDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, replacements);
    auto publicationDevice = makeDeviceCopy<graph::ChunkSlot<Elem>>(device, queue, publications);
    std::vector<std::uint32_t> indexHost(blocks, graph::invalidChunkSlot);
    auto indexDevice = makeDeviceCopy<std::uint32_t>(device, queue, indexHost);
    std::vector<std::uint32_t> statusHost(blocks, invalidStatus);
    auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);

    graph::ChunkArena<Elem> const arena{
        alpaka::onHost::data(storageDevice),
        static_cast<std::uint32_t>(storage.size() / capacity),
        capacity};
    graph::StreamAssembler<Elem> const assembler{
        alpaka::onHost::data(streamDevice),
        static_cast<std::uint32_t>(streams.size()),
        arena};

    auto const spec = alpaka::onHost::ThreadSpec{blocks, threads, executor};
    queue.enqueue(
        spec,
        CompetingWriterKernel<Elem>{},
        assembler,
        streams[0].id,
        alpaka::onHost::data(replacementDevice),
        static_cast<std::uint32_t>(replacements.size()),
        alpaka::onHost::data(publicationDevice),
        static_cast<std::uint32_t>(publications.size()),
        alpaka::onHost::data(indexDevice),
        alpaka::onHost::data(statusDevice));
    alpaka::onHost::wait(queue);

    auto const indices = copyToHost<std::uint32_t>(queue, indexDevice, blocks);
    auto const statuses = copyToHost<std::uint32_t>(queue, statusDevice, blocks);
    auto const published = copyToHost<graph::ChunkSlot<Elem>>(queue, publicationDevice, publications.size());
    auto const downloaded = copyToHost<Elem>(queue, storageDevice, storage.size());

    for(std::uint32_t i = 0u; i < blocks; ++i)
        CHECK(statuses[i] == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    std::vector<std::uint32_t> sorted = indices;
    std::sort(sorted.begin(), sorted.end());
    for(std::uint32_t i = 0u; i < blocks; ++i)
        REQUIRE(sorted[i] == i);

    REQUIRE(published[0].slot == 0u);
    REQUIRE(published[0].count == capacity);
    std::vector<Elem> publishedValues(capacity);
    for(std::uint32_t i = 0u; i < capacity; ++i)
        publishedValues[i] = downloaded[static_cast<std::size_t>(published[0].slot) * capacity + i];
    std::sort(publishedValues.begin(), publishedValues.end());
    for(std::uint32_t i = 0u; i < capacity; ++i)
        REQUIRE(publishedValues[i] == i);
}

TEST_CASE("chunk_assembly multi-destination ranks split one CUDA warp", "[chunk_assembly][cuda]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 64u;
    constexpr std::uint32_t threads = 32u;
    std::vector<graph::AssemblyStream<Elem>> streams{
        makeStream<Elem>(1u, capacity, 0u, 1u),
        makeStream<Elem>(2u, capacity, 2u, 3u)};
    auto storage = makeStorage<Elem>(6u, capacity);

    auto streamDevice = makeDeviceCopy<graph::AssemblyStream<Elem>>(device, queue, streams);
    auto storageDevice = makeDeviceCopy<Elem>(device, queue, storage);
    std::vector<std::uint32_t> rankHost(threads, invalidStatus);
    auto rankDevice = makeDeviceCopy<std::uint32_t>(device, queue, rankHost);
    std::vector<std::uint32_t> sizeHost(threads, invalidStatus);
    auto sizeDevice = makeDeviceCopy<std::uint32_t>(device, queue, sizeHost);
    std::vector<std::uint32_t> statusHost(1u, invalidStatus);
    auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);

    graph::ChunkArena<Elem> const arena{
        alpaka::onHost::data(storageDevice),
        static_cast<std::uint32_t>(storage.size() / capacity),
        capacity};
    graph::StreamAssembler<Elem> const assembler{
        alpaka::onHost::data(streamDevice),
        static_cast<std::uint32_t>(streams.size()),
        arena};

    auto const spec = alpaka::onHost::ThreadSpec{1u, threads, alpaka::exec::gpuCuda};
    queue.enqueue(
        spec,
        MultiDestinationKernel{},
        assembler,
        streams[0].id,
        streams[1].id,
        alpaka::onHost::data(rankDevice),
        alpaka::onHost::data(sizeDevice),
        alpaka::onHost::data(statusDevice));
    alpaka::onHost::wait(queue);

    auto const ranks = copyToHost<std::uint32_t>(queue, rankDevice, threads);
    auto const sizes = copyToHost<std::uint32_t>(queue, sizeDevice, threads);
    auto const status = copyToHost<std::uint32_t>(queue, statusDevice, 1u)[0];
    auto const downloadedStreams = copyToHost<graph::AssemblyStream<Elem>>(queue, streamDevice, streams.size());
    auto const downloadedStorage = copyToHost<Elem>(queue, storageDevice, storage.size());

    REQUIRE(status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    for(std::uint32_t lane = 0u; lane < threads; ++lane)
    {
        CHECK(sizes[lane] == threads / 2u);
        CHECK(ranks[lane] == lane / 2u);
    }
    REQUIRE(downloadedStreams[0].count == threads / 2u);
    REQUIRE(downloadedStreams[1].count == threads / 2u);
    for(std::uint32_t i = 0u; i < threads / 2u; ++i)
    {
        CHECK(downloadedStorage[i] == 100u + i);
        CHECK(downloadedStorage[2u * capacity + i] == 200u + i);
    }
}
#endif
