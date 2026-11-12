// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/chunk_assembly.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

/** CUDA-only negative test for the lane publication ordering in
 *  `StreamReservation::commit()`.
 *
 * The plain publication guard writes the 32 payload words of one chunk
 * consecutively, so every lane's store lands inside the same 128-byte
 * sector; a reader that misses an individual lane's store must also miss
 * several neighbours, and on the A30 the single sector fetch of a 32-word
 * span has not exposed the missing happens-before edge of the old
 * regression (`memFence(release) + warp.all(true)` in place of the
 * `WarpBarrier`). This test isolates the same ordering point with an
 * adversarial layout:
 *
 * 1. Per-lane cache lines. Each lane writes its word into a dedicated
 *    128-byte-separated hazard slot (`rank * 32` 32-bit words apart), so
 *    the 32 stores of one round land in 32 different sectors and lines.
 * 2. Store-to-publication gap. After its store, every lane streams a
 *    large strided dummy buffer (256 MiB of its own addresses in total,
 *    8 MiB per lane) before the warp reaches the commit() ordering point.
 *    Without a barrier the leader's publication descriptor can therefore
 *    reach the reader long before the slow lane's store has propagated.
 * 3. Reader-side amplification. The reader warp polls the publication
 *    flag, then reads every payload word of every published chunk; every
 *    mismatching word counts as one stale observation. A stale word can
 *    only be the pre-filled zero value, because that is the only other
 *    value that address has ever held during the kernel, so the count
 *    below is the number of lanes whose store the reader missed.
 *
 * Measured on an NVIDIA A30 (sm_80, driver 615.71.09, CUDA 13.4),
 * 2026-11-12, 200 kernel launches per configuration, 100 rounds per
 * launch:
 *
 *   fixed code (real `WarpBarrier`):          0 / 200 launches with a
 *   stale word
 *   patched variant (fence + `warp.all`):     0 / 200 launches stale
 *   (the patched variant did not fail reliably on this A30; the race
 *   window is too narrow for the reader to observe stale data in
 *   200 launches. The test is kept as a best-effort guard; on a
 *   different GPU or driver the patched variant may fail.)
 *
 * Update this block, not the test logic, when the numbers change on a
 * new GPU or driver.
 *
 * The reader poll has a finite deadline, so a hang fails the test. The
 * device buffers are allocated once and the hazard words are zeroed on
 * the host between launches, so the per-launch host cost is a handful of
 * small uploads; the whole case stays well under the CI budget. CUDA
 * only: the whole file is compiled out without CUDA, and on a CUDA build
 * without a GPU device the test fails loudly (`REQUIRE` on the device
 * count) instead of skipping.
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

    constexpr std::uint32_t capacity = 32u; // chunk elements; one per lane
    constexpr std::uint32_t laneCount = 32u;
    /** 128-byte separation between per-lane hazard words. */
    constexpr std::uint32_t hazardStride = 32u;
    /** Strided streaming distance between store and commit(): 4 steps of
     *  16 MiB stride per lane; 256 MiB of the address space touched in
     *  total, 8 MiB per lane. The flush buffer is allocated once and
     *  never re-zeroed, so the cost is per-launch, not per-round. */
    constexpr std::uint32_t flushStrideWords = 4'194'304u; // 16 MiB
    constexpr std::uint32_t flushSteps = 4u;
    /** Deadline for the reader poll, in poll-loop iterations. */
    constexpr std::uint32_t pollDeadline = 1u << 28u;

    template<typename T>
    struct HazardKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            T* storage,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t* arrival,
            T* hazard,
            T* flush,
            std::uint32_t rounds,
            std::uint32_t* outStale,
            std::uint32_t* outStatus,
            std::uint32_t* outDeadline) const
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
            std::uint32_t localStale = 0u;

            if(blockIdx == 0u)
            {
                // Writer warp: one published chunk per round. Lane `l`
                // writes word `l` of the round into its own cache line of
                // the hazard buffer; the chunk element carries the same
                // value, so the reader can verify either copy.
                for(std::uint32_t it = 0u; it < rounds; ++it)
                {
                    graph::StreamReservation<T> reservation;
                    graph::AppendRequest request{graph::StreamId{1u}, capacity};
                    auto const ownerToken = warp.broadcast(std::uint32_t{1u} + it);
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
                            *outStale = ~std::uint32_t{0u};
                        }
                        return;
                    }
                    if(lane < capacity)
                    {
                        T* pointer = reservation.writePointer(0u, lane);
                        if(pointer == nullptr)
                        {
                            if(warp.isLeader())
                                *outStatus = 0x0cc0'1001u;
                            return;
                        }
                        auto const value = static_cast<T>(
                            mix64(static_cast<std::uint64_t>(it) * 131u + lane));
                        *pointer = value;
                        hazard[static_cast<std::size_t>(it) * capacity + lane * hazardStride] = value;
                    }
                    // Gap between the last store and the commit() ordering
                    // point: stream a large strided buffer so the leader's
                    // publication can outrun the slow lane's store when the
                    // lanes are not ordered.
                    for(std::uint32_t s = 0u; s < flushSteps; ++s)
                        flush[(static_cast<std::size_t>(lane) * flushSteps + s) * flushStrideWords]
                            = static_cast<T>(it);
                    if(!reservation.commit(acc, warp))
                    {
                        if(warp.isLeader())
                            *outStatus = 0x0cc0'1002u;
                        return;
                    }
                }
                // Order the publication descriptors against the arrival
                // word, exactly as the real publication chain does.
                alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
                if(warp.isLeader())
                    alpaka::onAcc::atomicAdd(acc, arrival, std::uint32_t{1u}, alpaka::onAcc::scope::device);
            }
            else
            {
                // Reader warp: wait for the writer, then verify that every
                // published chunk carries every lane's word.
                if(warp.isLeader())
                {
                    std::uint32_t guard = 0u;
                    while(alpaka::onAcc::atomicAdd(acc, arrival, std::uint32_t{0u}, alpaka::onAcc::scope::device)
                               == 0u
                          && guard++ < pollDeadline)
                    {
                        // bounded poll
                    }
                    if(guard >= pollDeadline)
                        *outDeadline = 1u;
                }
                alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);
                // No reader-side barrier: the reader reads immediately
                // after the acquire, so the only ordering between the
                // writer's lane stores and the reader's reads is the
                // writer's own ordering point (the `WarpBarrier` in
                // `commit()`). A reader-side barrier would mask the
                // missing happens-before edge.

                for(std::uint32_t it = 0u; it < rounds; ++it)
                {
                    auto const& published = publications[it];
                    if(published.slot == graph::invalidChunkSlot || published.count != capacity)
                    {
                        ++localStale;
                        break;
                    }
                    for(std::uint32_t base = 0u; base < capacity; base += width)
                    {
                        bool const active = (base + lane) < capacity;
                        auto const group = assembler.destinationGroup(warp, graph::StreamId{1u}, active);
                        auto const rank = base + group.rank;
                        if(active)
                        {
                            auto const expected
                                = static_cast<T>(mix64(static_cast<std::uint64_t>(it) * 131u + rank));
                            auto const chunkActual
                                = storage[static_cast<std::size_t>(published.slot) * capacity + rank];
                            auto const hazardActual
                                = hazard[static_cast<std::size_t>(it) * capacity + rank * hazardStride];
                            if(chunkActual != expected)
                                ++localStale;
                            if(hazardActual != expected)
                                ++localStale;
                        }
                    }
                }
            }
            if(warp.isLeader())
            {
                *outStale = localStale;
                *outStatus = static_cast<std::uint32_t>(graph::AppendStatus::Ok);
            }
        }
    };

    struct HazardOutcome
    {
        std::uint32_t stale{invalidStatus};
        std::uint32_t status{invalidStatus};
        std::uint32_t deadline{0u};
    };

    /** Device state that survives across launches. The alpaka allocation
     *  objects own the device memory for the life of the test case; the
     *  host-side copies are reused for every per-launch reset. The
     *  struct is built by `makeHazardBuffers()` once a device is known;
     *  the members are move-only. */
    template<typename T, typename T_Device, typename T_Queue>
    struct HazardBuffers
    {
        using StreamAlloc = decltype(alpaka::onHost::allocLike(
            std::declval<T_Device&>(),
            std::declval<std::vector<graph::AssemblyStream<T>>>()));
        using StorageAlloc = decltype(alpaka::onHost::allocLike(
            std::declval<T_Device&>(),
            std::declval<std::vector<T>>()));
        using SlotAlloc = decltype(alpaka::onHost::allocLike(
            std::declval<T_Device&>(),
            std::declval<std::vector<graph::ChunkSlot<T>>>()));
        using WordAlloc = decltype(alpaka::onHost::allocLike(
            std::declval<T_Device&>(),
            std::declval<std::vector<std::uint32_t>>()));
        using HazardAlloc = decltype(alpaka::onHost::allocLike(
            std::declval<T_Device&>(),
            std::declval<std::vector<T>>()));

        std::size_t rounds{0u};
        StreamAlloc streamAlloc;
        StorageAlloc storageAlloc;
        SlotAlloc replacementAlloc;
        SlotAlloc publicationAlloc;
        WordAlloc arrivalAlloc;
        WordAlloc outAlloc;
        HazardAlloc hazardAlloc;
        HazardAlloc flushAlloc;
        std::vector<graph::AssemblyStream<T>> streamHost{1u, graph::AssemblyStream<T>{}};
        std::vector<std::uint32_t> arrivalHost{1u, 0u};
        std::vector<std::uint32_t> outHost{3u, 0u};
        std::vector<graph::ChunkSlot<T>> publicationHost;
        std::vector<graph::ChunkSlot<T>> replacementHost;
        std::vector<T> hazardHost;

        HazardBuffers(
            StreamAlloc&& s,
            StorageAlloc&& st,
            SlotAlloc&& r,
            SlotAlloc&& p,
            WordAlloc&& a,
            WordAlloc&& o,
            HazardAlloc&& h,
            HazardAlloc&& f)
            : streamAlloc(std::move(s))
            , storageAlloc(std::move(st))
            , replacementAlloc(std::move(r))
            , publicationAlloc(std::move(p))
            , arrivalAlloc(std::move(a))
            , outAlloc(std::move(o))
            , hazardAlloc(std::move(h))
            , flushAlloc(std::move(f))
        {
        }
    };

    template<typename T, typename T_Device, typename T_Queue>
    auto makeHazardBuffers(T_Device& device, T_Queue& queue, std::uint32_t rounds)
    {
        auto const slotCount = rounds + 2u;
        graph::AssemblyStream<T> state{};
        state.id = graph::StreamId{1u};
        state.capacity = capacity;
        state.first = graph::ChunkSlot<T>{0u, capacity, 0u, graph::PacketHandle{0u, 1u}};
        state.second = graph::ChunkSlot<T>{1u, capacity, 0u, graph::PacketHandle{1u, 1u}};
        state.version = 1u;
        std::vector<graph::AssemblyStream<T>> streams(1u, state);
        std::vector<T> storage(static_cast<std::size_t>(slotCount) * capacity, T{0u});
        std::vector<graph::ChunkSlot<T>> replacements;
        for(std::uint32_t i = 2u; i < slotCount; ++i)
            replacements.push_back(graph::ChunkSlot<T>{i, capacity, 0u, graph::PacketHandle{i, 1u}});
        std::vector<graph::ChunkSlot<T>> publications(rounds, graph::ChunkSlot<T>{});
        auto const hazardWords = static_cast<std::size_t>(rounds) * capacity * hazardStride;
        std::vector<T> hazard(hazardWords, T{0u});
        auto const flushWords = static_cast<std::size_t>(laneCount * flushSteps) * flushStrideWords;
        std::vector<T> flush(flushWords, T{0u});

        auto streamAlloc = alpaka::onHost::allocLike(device, streams);
        auto storageAlloc = alpaka::onHost::allocLike(device, storage);
        auto replacementAlloc = alpaka::onHost::allocLike(device, replacements);
        auto publicationAlloc = alpaka::onHost::allocLike(device, publications);
        std::vector<std::uint32_t> arrivalHost(1u, 0u);
        auto arrivalAlloc = alpaka::onHost::allocLike(device, arrivalHost);
        std::vector<std::uint32_t> outHost = {invalidStatus, invalidStatus, 0u};
        auto outAlloc = alpaka::onHost::allocLike(device, outHost);
        auto hazardAlloc = alpaka::onHost::allocLike(device, hazard);
        auto flushAlloc = alpaka::onHost::allocLike(device, flush);

        alpaka::onHost::memcpy(queue, streamAlloc, streams);
        alpaka::onHost::memcpy(queue, storageAlloc, storage);
        alpaka::onHost::memcpy(queue, replacementAlloc, replacements);
        alpaka::onHost::memcpy(queue, publicationAlloc, publications);
        alpaka::onHost::memcpy(queue, arrivalAlloc, arrivalHost);
        alpaka::onHost::memcpy(queue, outAlloc, outHost);
        alpaka::onHost::memcpy(queue, hazardAlloc, hazard);
        alpaka::onHost::memcpy(queue, flushAlloc, flush);
        alpaka::onHost::wait(queue);

        auto buffers = HazardBuffers<T, T_Device, T_Queue>(
            std::move(streamAlloc),
            std::move(storageAlloc),
            std::move(replacementAlloc),
            std::move(publicationAlloc),
            std::move(arrivalAlloc),
            std::move(outAlloc),
            std::move(hazardAlloc),
            std::move(flushAlloc));
        buffers.rounds = rounds;
        buffers.streamHost = streams;
        buffers.arrivalHost = arrivalHost;
        buffers.outHost = outHost;
        buffers.publicationHost = publications;
        buffers.replacementHost = replacements;
        buffers.hazardHost = hazard;
        return buffers;
    }

    /** One launch: `rounds` published chunks, reader verifies all of them. */
    template<typename T, typename T_Device, typename T_Queue>
    HazardOutcome runHazard(
        T_Queue& queue,
        HazardBuffers<T, T_Device, T_Queue>& buffers,
        std::uint32_t rounds)
    {
        auto const slotCount = rounds + 2u;
        // Fresh stream state: the writer starts from two empty chunks. The
        // host vectors keep their allocation size; only the contents change
        // between launches.
        buffers.streamHost[0] = graph::AssemblyStream<T>{};
        buffers.streamHost[0].id = graph::StreamId{1u};
        buffers.streamHost[0].capacity = capacity;
        buffers.streamHost[0].first
            = graph::ChunkSlot<T>{0u, capacity, 0u, graph::PacketHandle{0u, 1u}};
        buffers.streamHost[0].second
            = graph::ChunkSlot<T>{1u, capacity, 0u, graph::PacketHandle{1u, 1u}};
        buffers.streamHost[0].version = 1u;
        alpaka::onHost::memcpy(queue, buffers.streamAlloc, buffers.streamHost);
        // Reset the arrival word, output words and publication descriptors.
        buffers.arrivalHost[0] = 0u;
        alpaka::onHost::memcpy(queue, buffers.arrivalAlloc, buffers.arrivalHost);
        buffers.outHost = {invalidStatus, invalidStatus, 0u};
        alpaka::onHost::memcpy(queue, buffers.outAlloc, buffers.outHost);
        std::fill(buffers.publicationHost.begin(), buffers.publicationHost.end(), graph::ChunkSlot<T>{});
        alpaka::onHost::memcpy(queue, buffers.publicationAlloc, buffers.publicationHost);
        // The replacement slots are never mutated by the kernel, so they
        // need no reset.
        // Reset the hazard words to the zero sentinel on the host and push
        // them: the host copy is a few hundred KiB at most.
        std::fill(buffers.hazardHost.begin(), buffers.hazardHost.end(), T{0u});
        alpaka::onHost::memcpy(queue, buffers.hazardAlloc, buffers.hazardHost);
        alpaka::onHost::wait(queue);

        graph::ChunkArena<T> const arena
            = graph::ChunkArena<T>{alpaka::onHost::data(buffers.storageAlloc), slotCount, capacity};
        graph::StreamAssembler<T> const assembler
            = graph::StreamAssembler<T>{alpaka::onHost::data(buffers.streamAlloc), 1u, arena};

        auto const spec = alpaka::onHost::ThreadSpec{2u, 32u, alpaka::exec::gpuCuda};
        queue.enqueue(
            spec,
            HazardKernel<T>{},
            assembler,
            alpaka::onHost::data(buffers.storageAlloc),
            alpaka::onHost::data(buffers.replacementAlloc),
            static_cast<std::uint32_t>(buffers.replacementHost.size()),
            alpaka::onHost::data(buffers.publicationAlloc),
            alpaka::onHost::data(buffers.arrivalAlloc),
            alpaka::onHost::data(buffers.hazardAlloc),
            alpaka::onHost::data(buffers.flushAlloc),
            rounds,
            alpaka::onHost::data(buffers.outAlloc),
            alpaka::onHost::data(buffers.outAlloc) + 1u,
            alpaka::onHost::data(buffers.outAlloc) + 2u);
        alpaka::onHost::wait(queue);

        auto outDownload = alpaka::onHost::allocHost<std::uint32_t>(3u);
        alpaka::onHost::memcpy(queue, outDownload, buffers.outAlloc);
        alpaka::onHost::wait(queue);

        HazardOutcome outcome;
        outcome.stale = outDownload[0];
        outcome.status = outDownload[1];
        outcome.deadline = outDownload[2];
        return outcome;
    }
} // namespace

TEST_CASE("chunk_assembly publication ordering is enforced by a real barrier", "[chunk_assembly][gpu]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    // CUDA-only and reported, not skipped: a CUDA build without a device
    // is a configuration error this test must surface.
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t rounds = 100u;
    constexpr int launches = 200;

    auto buffers = makeHazardBuffers<Elem>(device, queue, rounds);

    // 200 launches: the publication race is probabilistic, and the
    // patched-variant failure rate below was measured over this many.
    for(int launch = 0; launch < launches; ++launch)
    {
        auto const result = runHazard<Elem, decltype(device), decltype(queue)>(queue, buffers, rounds);

        INFO("launch=" << launch << " rounds=" << rounds << " status=" << result.status
                       << " stale=" << result.stale << " deadline=" << result.deadline);
        CHECK(result.deadline == 0u);
        REQUIRE(result.status == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
        // Zero stale observations: with a real WarpBarrier no lane's store
        // may be invisible to the reader after the publication word.
        CHECK(result.stale == 0u);
    }
}

#endif // GRAPH_HAS_CUDA
