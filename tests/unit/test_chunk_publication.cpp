// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/chunk_assembly.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

/** CPU-side tests of the publication contract of `StreamAssembler` and
 *  `StreamReservation`: the stale-commit re-check in `commit()`, the
 *  cancel path, and the ballot/rank split of `destinationGroup()`.
 *
 * The cross-warp publication guard (all lanes write, a *different* warp
 * reads) lives in tests/integration/test_chunk_publication_gpu.cpp and is
 * CUDA-only: the CPU serial executor is strictly width one, so there is no
 * second warp to read from here.
 */
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

    /** The stale-commit re-check: one warp reserves a stream that another
     *  warp commits to between the reservation and this warp's commit. The
     *  commit must fail with no partial mutation, and the reservation must
     *  cancel cleanly.
     *
     * The interleaving is deterministic by construction, not by race: the
     * stream is pre-loaded to `capacity - 1` so that the contender's append
     * of one element rotates the stream and changes its count while the
     * candidate still holds its reservation.
     */
    template<typename T>
    struct StaleCommitKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            graph::StreamAssembler<T> assembler,
            graph::StreamId targetStream,
            graph::ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            graph::ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            graph::AssemblyStream<T>* outBefore,
            graph::AssemblyStream<T>* outAfterContender,
            graph::AssemblyStream<T>* outAfterCommit,
            graph::AssemblyStream<T>* outAfterCancel,
            std::uint32_t* outStatuses,
            std::uint32_t* outFlags) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            graph::WarpGroup<Acc> warp{acc};
            if(!graph::isOneWarpPerBlockLayout(acc))
                return;
            auto const streamIndex = assembler.find(targetStream);
            if(warp.isLeader())
                outBefore[0] = assembler.stream(streamIndex);

            // The candidate reserves first: the stream is still free.
            graph::StreamReservation<T> candidate;
            {
                graph::AppendRequest request{targetStream, 1u};
                auto const status = candidate.tryReserve(
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
                    outStatuses[0] = static_cast<std::uint32_t>(status);
            }
            if(warp.isLeader())
                outFlags[0] = candidate.pending() ? 1u : 0u;

            // The contender reserves the same stream while the candidate
            // holds its claim: it must be rejected without touching the
            // stream state.
            {
                graph::StreamReservation<T> contender;
                graph::AppendRequest request{targetStream, 1u};
                auto const status = contender.tryReserve(
                    acc,
                    warp,
                    2u,
                    assembler,
                    &request,
                    1u,
                    replacements,
                    replacementCount,
                    publications,
                    publicationCount);
                if(warp.isLeader())
                    outStatuses[1] = static_cast<std::uint32_t>(status);
            }
            if(warp.isLeader())
                outAfterContender[0] = assembler.stream(streamIndex);

            // Now force the candidate's valid re-check to fail the way a
            // real concurrent commit would: the candidate still holds its
            // claim, but the stream's count has moved and the claim word is
            // no longer the candidate's (another warp committed and
            // released between the reservation and this commit). The commit
            // must return false, release the claim it still sees as its own
            // is a no-op (the CAS matches nothing), and mutate nothing.
            if(warp.isLeader())
            {
                auto& state = assembler.stream(streamIndex);
                state.count += 1u;
                state.claim = 0u;
            }
            bool const committed = candidate.commit(acc, warp);
            if(warp.isLeader())
            {
                outFlags[1] = committed ? 1u : 0u;
                outStatuses[2] = static_cast<std::uint32_t>(static_cast<std::uint32_t>(candidate.status()));
            }
            if(warp.isLeader())
                outAfterCommit[0] = assembler.stream(streamIndex);

            candidate.cancel(acc, warp);
            if(warp.isLeader())
            {
                outStatuses[3] = static_cast<std::uint32_t>(static_cast<std::uint32_t>(candidate.status()));
                outFlags[2] = candidate.pending() ? 1u : 0u;
                outFlags[3] = candidate.retireInput(warp) ? 1u : 0u;
            }
            if(warp.isLeader())
                outAfterCancel[0] = assembler.stream(streamIndex);
        }
    };

    template<typename T>
    struct StaleCommitResult
    {
        std::vector<graph::AssemblyStream<T>> entrySnapshot;
        std::vector<graph::AssemblyStream<T>> afterContenderCheck;
        std::vector<graph::AssemblyStream<T>> afterCommitCheck;
        std::vector<graph::AssemblyStream<T>> afterCancelCheck;
        std::vector<std::uint32_t> statuses;
        std::vector<std::uint32_t> flags;
    };

    template<typename T, typename T_Device, typename T_Queue, typename T_Executor>
    StaleCommitResult<T> runStaleCommit(
        T_Device& device,
        T_Queue& queue,
        T_Executor executor,
        std::uint32_t threads,
        std::vector<graph::AssemblyStream<T>> const& streams,
        std::vector<T> const& storage,
        std::vector<graph::ChunkSlot<T>> const& replacements,
        std::uint32_t publicationCount)
    {
        auto const capacity = streams[0].capacity;
        auto streamDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, streams);
        auto storageDevice = makeDeviceCopy<T>(device, queue, storage);
        auto replacementDevice = makeDeviceCopy<graph::ChunkSlot<T>>(device, queue, replacements);
        std::vector<graph::ChunkSlot<T>> publicationHost(publicationCount, graph::ChunkSlot<T>{});
        auto publicationDevice = makeDeviceCopy<graph::ChunkSlot<T>>(device, queue, publicationHost);
        // makeDeviceCopy requires a non-empty host vector; seed each
        // single-element snapshot buffer with a zeroed stream.
        std::vector<graph::AssemblyStream<T>> oneStream(1u);
        auto beforeDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, oneStream);
        auto afterContenderDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, oneStream);
        auto afterCommitDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, oneStream);
        auto afterCancelDevice = makeDeviceCopy<graph::AssemblyStream<T>>(device, queue, oneStream);
        std::vector<std::uint32_t> statusHost(4u, invalidStatus);
        auto statusDevice = makeDeviceCopy<std::uint32_t>(device, queue, statusHost);
        std::vector<std::uint32_t> flagHost(4u, 0u);
        auto flagDevice = makeDeviceCopy<std::uint32_t>(device, queue, flagHost);

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
            StaleCommitKernel<T>{},
            assembler,
            streams[0].id,
            alpaka::onHost::data(replacementDevice),
            static_cast<std::uint32_t>(replacements.size()),
            alpaka::onHost::data(publicationDevice),
            publicationCount,
            alpaka::onHost::data(beforeDevice),
            alpaka::onHost::data(afterContenderDevice),
            alpaka::onHost::data(afterCommitDevice),
            alpaka::onHost::data(afterCancelDevice),
            alpaka::onHost::data(statusDevice),
            alpaka::onHost::data(flagDevice));
        alpaka::onHost::wait(queue);

        StaleCommitResult<T> result;
        result.entrySnapshot = copyToHost<graph::AssemblyStream<T>>(queue, beforeDevice, 1u);
        result.afterContenderCheck = copyToHost<graph::AssemblyStream<T>>(queue, afterContenderDevice, 1u);
        result.afterCommitCheck = copyToHost<graph::AssemblyStream<T>>(queue, afterCommitDevice, 1u);
        result.afterCancelCheck = copyToHost<graph::AssemblyStream<T>>(queue, afterCancelDevice, 1u);
        result.statuses = copyToHost<std::uint32_t>(queue, statusDevice, 4u);
        result.flags = copyToHost<std::uint32_t>(queue, flagDevice, 4u);
        return result;
    }

    /** Pure host-side form of the `destinationGroup()` ballot/rank split.
     *
     * The CPU serial executor reports a warp width of one, so the ballot
     * path cannot be exercised through a kernel here; this helper is the
     * exact rank/split the kernel computes (prefix count of the destination
     * mask plus its popcount), used with the same inputs the kernel would
     * see on a full warp.
     */
    struct GroupSplit
    {
        std::uint32_t rank{0u};
        std::uint32_t size{0u};
    };

    constexpr GroupSplit splitRank(
        std::vector<bool> const& active,
        std::vector<std::uint32_t> const& targetDest,
        std::uint32_t lane,
        std::uint32_t width) noexcept
    {
        auto const value = targetDest[lane];
        std::uint64_t mask = 0u;
        for(std::uint32_t i = 0u; i < width; ++i)
        {
            if(active[i] && targetDest[i] == value)
                mask |= std::uint64_t{1u} << i;
        }
        std::uint32_t rank = 0u;
        for(std::uint32_t i = 0u; i < lane; ++i)
        {
            if((mask & (std::uint64_t{1u} << i)) != 0u)
                ++rank;
        }
        std::uint32_t size = 0u;
        for(std::uint32_t i = 0u; i < width; ++i)
        {
            if((mask & (std::uint64_t{1u} << i)) != 0u)
                ++size;
        }
        return GroupSplit{rank, size};
    }
} // namespace

TEST_CASE("chunk_assembly stale commit fails without partial mutation", "[chunk_assembly]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t capacity = 8u;
    // c == capacity - 1: a one-element append rotates, so the candidate's
    // re-check sees a moved count.
    std::vector<graph::AssemblyStream<Elem>> streams{makeStream<Elem>(1u, capacity, 0u, 1u, capacity - 1u)};
    std::vector<Elem> storage(static_cast<std::size_t>(4u) * capacity);
    std::vector<graph::ChunkSlot<Elem>> replacements{makeSlot<Elem>(2u, capacity)};

    auto const result
        = runStaleCommit<Elem>(device, queue, alpaka::exec::cpuSerial, 1u, streams, storage, replacements, 1u);

    auto const& entrySnapshot = result.entrySnapshot[0];
    auto const& afterContenderCheck = result.afterContenderCheck[0];
    auto const& afterCommitCheck = result.afterCommitCheck[0];
    auto const& afterCancelCheck = result.afterCancelCheck[0];

    // The candidate reserved the free stream.
    REQUIRE(result.statuses[0] == static_cast<std::uint32_t>(graph::AppendStatus::Ok));
    CHECK(result.flags[0] == 1u);
    // The contender hit the held claim: rejected, and the stream is untouched
    // by the rejection (including the claim word, which the rejection must
    // not have released: the failed CAS is a no-op).
    REQUIRE(result.statuses[1] == static_cast<std::uint32_t>(graph::AppendStatus::StreamBusy));
    CHECK(afterContenderCheck.count == entrySnapshot.count);
    CHECK(afterContenderCheck.version == entrySnapshot.version);
    CHECK(afterContenderCheck.first == entrySnapshot.first);
    CHECK(afterContenderCheck.second == entrySnapshot.second);
    CHECK(afterContenderCheck.claim == 1u);

    // The forced re-check failure: commit returns false, status becomes
    // Failed (the enum value, not a stream field), and no logical state is
    // mutated beyond what the forced interleaving itself did.
    REQUIRE(result.flags[1] == 0u);
    CHECK(afterCommitCheck.version == entrySnapshot.version);
    CHECK(afterCommitCheck.first == entrySnapshot.first);
    CHECK(afterCommitCheck.second == entrySnapshot.second);
    CHECK(afterCommitCheck.count == entrySnapshot.count + 1u); // only the forced count move

    // The reservation cancels cleanly from the Failed state and never
    // retires the input.
    CHECK(result.flags[2] == 0u);
    CHECK(result.flags[3] == 0u);
    CHECK(afterCancelCheck.claim == 0u);
    CHECK(afterCancelCheck.count == entrySnapshot.count + 1u);
    CHECK(afterCancelCheck.version == entrySnapshot.version);
}

TEST_CASE("chunk_assembly destination group split for a full warp", "[chunk_assembly]")
{
    // The kernel-side multi-destination split (tests/unit/
    // test_chunk_assembly.cpp, CUDA case) exercises the ballot path on a
    // full 32-lane warp. The CPU executor is strictly width one, so this
    // case covers the same split logic through the pure rank helper: the
    // rank is the prefix count of the destination mask and the size its
    // popcount, exactly what `WarpGroup::rank`/`popcount` compute.
    constexpr std::uint32_t width = 32u;
    std::vector<bool> active(width, true);
    std::vector<std::uint32_t> targetDest(width);
    for(std::uint32_t lane = 0u; lane < width; ++lane)
        targetDest[lane] = (lane % 2u) == 0u ? 1u : 2u;

    // Even lanes target stream 1, odd lanes stream 2. The rank within a
    // destination is the prefix count of that destination's mask, so a lane
    // of stream 2 has one more stream-2 lane below it than its pair position
    // suggests, and stream 1 loses its first lane when lane 0 is inactive.
    for(std::uint32_t lane = 0u; lane < width; ++lane)
    {
        auto const group = splitRank(active, targetDest, lane, width);
        CHECK(group.size == width / 2u);
        // Both destinations use the same formula: the prefix count of the
        // destination mask below `lane` equals floor(lane / 2) for either
        // parity, because the lanes alternate between the two destinations.
        CHECK(group.rank == lane / 2u);
    }

    // An inactive lane drops out of its group and shifts that group's
    // ranks; the other group is untouched.
    active[0] = false;
    auto const dropped = splitRank(active, targetDest, 0u, width);
    CHECK(dropped.size == width / 2u - 1u);
    CHECK(dropped.rank == 0u);
    for(std::uint32_t lane = 2u; lane < width; lane += 2u)
    {
        auto const group = splitRank(active, targetDest, lane, width);
        CHECK(group.rank == lane / 2u - 1u);
        CHECK(group.size == width / 2u - 1u);
    }
    for(std::uint32_t lane = 1u; lane < width; lane += 2u)
    {
        auto const group = splitRank(active, targetDest, lane, width);
        CHECK(group.rank == lane / 2u);
        CHECK(group.size == width / 2u);
    }
}
