// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/chunk_assembly.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

/** CUDA-only publication regression guard for `StreamAssembler` /
 *  `StreamReservation`.
 *
 * One warp per block (the bring-up layout). The writer warp of block 0 has
 * all 32 lanes write each published chunk's elements through the
 * reservation, commits, and publishes the filled chunk. The reader warp of
 * block 1 rendezvous on a device-scope arrival word and then reads every
 * element of every published chunk. If the publication chain in `commit()`
 * (lane stores, `WarpBarrier`, leader release fence, consumer acquire)
 * regressed to a vote or a bare leader fence, the reader can observe stale
 * elements or a descriptor published ahead of the payload; the repeated
 * launches make that observable.
 *
 * CUDA only: the CPU serial executor is strictly width one, so no second
 * warp exists to read from.
 */
#if GRAPH_HAS_CUDA
namespace
{
    using Elem = std::uint32_t;

    constexpr std::uint32_t invalidStatus = 0xffff'ffffu;

    /** Mix a 64-bit value (splitmix64 finalizer). */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t mix64(std::uint64_t x) noexcept
    {
        x = (x + 0x9e37'79b9'7f4a'7c15ull) ^ (x >> 30u);
        x *= 0xbf58'476d'1ce4'e5b9ull;
        x = (x ^ (x >> 27u)) * 0x94d0'49bb'1331'11ebull;
        return x ^ (x >> 31u);
    }

    /** One published chunk carries `chunkCount` elements; element `i` of
     *  round `it` carries `static_cast<Elem>(mix64(it * 131u + i))`. */
    constexpr std::uint32_t chunkCount = 32u;

    template<typename T>
    struct PublicationKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            T* storage,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t* arrival,
            std::uint32_t iterations,
            std::uint32_t* outErrors,
            std::uint32_t* outStatus) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            auto const blockIdx = static_cast<std::uint32_t>(
                acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)[0]);
            if(!graph::isOneWarpPerBlockLayout(acc))
            {
                if(warp.isLeader())
                    *outStatus = invalidStatus;
                return;
            }
            constexpr std::uint32_t blockThreads = 32u;
            auto const width = warp.workerWidth();
            auto const lane = warp.laneIndex();
            auto const streamIndex = assembler.find(graph::StreamId{1u});
            auto const capacity = assembler.stream(streamIndex).capacity;
            std::uint32_t localErrors = 0u;

            if(blockIdx == 0u)
            {
                // Writer warp: every round fills one full chunk. All lanes
                // write through the reservation, then the leader publishes
                // it via commit().
                for(std::uint32_t it = 0u; it < iterations; ++it)
                {
                    graph::StreamReservation<T> reservation;
                    graph::AppendRequest request{graph::StreamId{1u}, chunkCount};
                    // ownerToken is uniform across the warp (a shuffle from
                    // lane 0): the claim word must be identical on every
                    // lane for the leader's CAS to succeed.
                    auto const ownerToken = warp.broadcast(std::uint32_t{1u} + it);
                    // Each round needs a fresh replacement slot: after a
                    // rotation the old `second` becomes `first`, and the
                    // replacement becomes the new `second`, so the next round
                    // must not reuse it.
                    auto const reserveStatus = reservation.tryReserve(
                        acc,
                        warp,
                        ownerToken,
                        assembler,
                        &request,
                        1u,
                        replacements + it,
                        replacementCount - it,
                        publications + it,
                        1u);
                    if(reserveStatus != graph::AppendStatus::Ok)
                    {
                        if(warp.isLeader())
                        {
                            *outStatus = static_cast<std::uint32_t>(reserveStatus);
                            outErrors[1] = it;
                        }
                        return;
                    }
                    for(std::uint32_t base = 0u; base < chunkCount; base += width)
                    {
                        bool const active = (base + lane) < chunkCount;
                        auto const group = assembler.destinationGroup(warp, graph::StreamId{1u}, active);
                        auto const rank = base + group.rank;
                        if(active)
                        {
                            T* pointer = reservation.writePointer(0u, rank);
                            if(pointer == nullptr)
                            {
                                if(warp.isLeader())
                                    *outStatus = 0x0cc0'1001u;
                                return;
                            }
                            *pointer = static_cast<T>(mix64(static_cast<std::uint64_t>(it) * 131u + rank));
                        }
                    }
                    if(!reservation.commit(acc, warp))
                    {
                        if(warp.isLeader())
                            *outStatus = 0x0cc0'1002u;
                        return;
                    }
                }
                // The commit() chain already released the lane stores and the
                // publication descriptors. This second release fence orders
                // the publication stores against the arrival store, so the
                // reader's acquire on the arrival word sees everything.
                alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
                if(warp.isLeader())
                {
                    alpaka::onAcc::atomicAdd(acc, arrival, std::uint32_t{1u}, alpaka::onAcc::scope::device);
                    *outStatus = static_cast<std::uint32_t>(graph::AppendStatus::Ok);
                }
            }
            else
            {
                // Reader warp: wait for the writer to finish, then verify
                // every element of every published chunk. Each rotation uses
                // a unique slot, so the published chunks are not overwritten.
                if(warp.isLeader())
                {
                    std::uint32_t guard = 0u;
                    while(alpaka::onAcc::atomicAdd(acc, arrival, std::uint32_t{0u}, alpaka::onAcc::scope::device) == 0u
                          && guard++ < (1u << 26u))
                    {
                        // bounded poll
                    }
                }
                alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);
                graph::WarpBarrier<Acc> readBarrier{};
                readBarrier.barrier(acc, blockThreads, warp);

                // Every lane reads its own element from each published chunk.
                // A different warp reads the payload written by the writer.
                for(std::uint32_t it = 0u; it < iterations; ++it)
                {
                    auto const& published = publications[it];
                    if(published.slot == graph::invalidChunkSlot || published.count != capacity)
                    {
                        ++localErrors;
                        break;
                    }
                    if(lane < chunkCount)
                    {
                        auto const expected = static_cast<T>(mix64(static_cast<std::uint64_t>(it) * 131u + lane));
                        auto const actual = storage[static_cast<std::size_t>(published.slot) * capacity + lane];
                        if(actual != expected)
                        {
                            ++localErrors;
                            if(localErrors <= 4u && warp.isLeader())
                            {
                                outErrors[1] = it;
                                outErrors[2] = lane;
                                outErrors[3] = actual;
                                outErrors[4] = expected;
                                outErrors[5] = published.slot;
                            }
                        }
                    }
                }
            }
            if(warp.isLeader())
                *outErrors = localErrors;
        }
    };

    template<typename T>
    struct PublicationResult
    {
        std::vector<T> storage;
        std::vector<graph::AssemblyStream<T>> streams;
        std::vector<graph::ChunkSlot<T>> publications;
        std::uint32_t errors{invalidStatus};
        std::uint32_t status{invalidStatus};
    };

    template<typename T, typename T_Device, typename T_Queue>
    PublicationResult<T> runPublication(
        T_Device& device,
        T_Queue& queue,
        std::uint32_t capacity,
        std::uint32_t iterations)
    {
        // One fresh arena slot per rotation, plus the two initial chunks.
        auto const slotCount = iterations + 2u;
        graph::AssemblyStream<T> state{};
        state.id = graph::StreamId{1u};
        state.capacity = capacity;
        state.first = graph::ChunkSlot<T>{0u, capacity, 0u, graph::PacketHandle{0u, 1u}};
        state.second = graph::ChunkSlot<T>{1u, capacity, 0u, graph::PacketHandle{1u, 1u}};
        state.version = 1u;
        std::vector<graph::AssemblyStream<T>> streams(1u, state);
        std::vector<T> storage(static_cast<std::size_t>(slotCount) * capacity);
        std::vector<graph::ChunkSlot<T>> replacements;
        for(std::uint32_t i = 2u; i < slotCount; ++i)
            replacements.push_back(graph::ChunkSlot<T>{i, capacity, 0u, graph::PacketHandle{i, 1u}});
        std::vector<graph::ChunkSlot<T>> publications(iterations, graph::ChunkSlot<T>{});

        auto upload = [&](auto const& host, std::size_t size)
        {
            auto d = alpaka::onHost::allocLike(device, host);
            alpaka::onHost::memcpy(queue, d, host);
            alpaka::onHost::wait(queue);
            return d;
        };
        auto streamHost = alpaka::onHost::allocHost<graph::AssemblyStream<T>>(1u);
        streamHost[0] = streams[0];
        auto storageHost = alpaka::onHost::allocHost<T>(storage.size());
        for(std::size_t i = 0u; i < storage.size(); ++i)
            storageHost[i] = storage[i];
        auto replacementHost = alpaka::onHost::allocHost<graph::ChunkSlot<T>>(replacements.size());
        for(std::size_t i = 0u; i < replacements.size(); ++i)
            replacementHost[i] = replacements[i];
        auto publicationHost = alpaka::onHost::allocHost<graph::ChunkSlot<T>>(publications.size());
        for(std::size_t i = 0u; i < publications.size(); ++i)
            publicationHost[i] = publications[i];
        auto arrivalHost = alpaka::onHost::allocHost<std::uint32_t>(1u);
        arrivalHost[0] = 0u;
        auto outHost = alpaka::onHost::allocHost<std::uint32_t>(4u);
        outHost[0] = invalidStatus;
        outHost[1] = invalidStatus;
        outHost[2] = 0u;
        outHost[3] = 0u;

        auto streamDevice = upload(streamHost, 1u);
        auto storageDevice = upload(storageHost, storage.size());
        auto replacementDevice = upload(replacementHost, replacements.size());
        auto publicationDevice = upload(publicationHost, publications.size());
        auto arrivalDevice = upload(arrivalHost, 1u);
        auto outDevice = upload(outHost, 2u);

        graph::ChunkArena<T> const arena{alpaka::onHost::data(storageDevice), slotCount, capacity};
        graph::StreamAssembler<T> const assembler{alpaka::onHost::data(streamDevice), 1u, arena};

        auto const spec = alpaka::onHost::ThreadSpec{2u, 32u, alpaka::exec::gpuCuda};
        queue.enqueue(
            spec,
            PublicationKernel<T>{},
            assembler,
            alpaka::onHost::data(storageDevice),
            alpaka::onHost::data(replacementDevice),
            static_cast<std::uint32_t>(replacements.size()),
            alpaka::onHost::data(publicationDevice),
            alpaka::onHost::data(arrivalDevice),
            iterations,
            alpaka::onHost::data(outDevice),
            alpaka::onHost::data(outDevice) + 1u);
        alpaka::onHost::wait(queue);

        auto download = [&](auto host, auto dev)
        {
            alpaka::onHost::memcpy(queue, host, dev);
            alpaka::onHost::wait(queue);
        };
        download(outHost, outDevice);
        download(storageHost, storageDevice);
        download(streamHost, streamDevice);
        download(publicationHost, publicationDevice);

        PublicationResult<T> result;
        result.errors = outHost[0];
        result.status = outHost[1];
        for(std::size_t i = 0u; i < storage.size(); ++i)
            result.storage.push_back(storageHost[i]);
        result.streams.push_back(streamHost[0]);
        for(std::size_t i = 0u; i < publications.size(); ++i)
            result.publications.push_back(publicationHost[i]);
        return result;
    }
} // namespace

TEST_CASE("chunk_assembly publication guard: different warp reads every element", "[chunk_assembly][gpu]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 32u; // one chunk per arena slot
    constexpr std::uint32_t iterations = 1000u;

    // Run the guard in repeated launches of its own: the publication race is
    // probabilistic, so a single launch can pass by luck. Each launch writes
    // `iterations` published chunks and reads every element from the other
    // warp.
    for(int launch = 0; launch < 20; ++launch)
    {
        auto const result = runPublication<Elem>(device, queue, capacity, iterations);

        INFO(
            "launch=" << launch << " iterations=" << iterations << " status=" << result.status
                      << " errors=" << result.errors);
        REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
        CHECK(result.errors == 0u);

        // The writer must have published exactly one full chunk per
        // iteration and left the stream empty.
        auto const& stream = result.streams[0];
        CHECK(stream.count == 0u);
        std::uint32_t publishedChunks = 0u;
        for(std::uint32_t i = 0u; i < iterations; ++i)
        {
            auto const& chunk = result.publications[i];
            if(chunk.slot != graph::invalidChunkSlot)
            {
                CHECK(chunk.count == capacity);
                ++publishedChunks;
            }
        }
        CHECK(publishedChunks == iterations);
    }
}

#endif // GRAPH_HAS_CUDA
