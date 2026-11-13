// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/barrier_diagnostics.hpp>
#include <graph/detail/concepts.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/stream_id.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

/** Per-destination chunk assembly with two physical chunks and one counter.
 *
 * This header implements the assembly contract from design section 6 without
 * any accelerator selection. The same code path runs on the CPU executor and
 * on device code.
 *
 * # Layout
 *
 * Every destination stream owns two physical chunks `A` (first) and `B`
 * (second), each with capacity `Y`, and one logical element counter `c`:
 *
 * ```
 * logical index:   0        c        c+k-1
 *                  |--------|---------|
 *                  A (0..Y-1)  B (Y..2Y-2)
 * ```
 *
 * An append of `k` elements writes logical indices `c .. c+k-1`. Indices
 * below `Y` address `A`, the remaining indices address `B` with offset `-Y`.
 * Because `0 <= c < Y` and `k <= Y`, the maximum occupancy after one append is
 * `2Y - 1`, so `B` needs at most `Y - 1` element slots and a single rotation
 * always suffices. `maxAppendOccupancy()` and `appendFits()` make that bound
 * explicit and are checked by `StreamReservation::tryReserve()`.
 *
 * A destination stream is identified by a `StreamId`, never by a numeric bin
 * id, so two stages that share a bin number still own independent state.
 *
 * # Ownership
 *
 * One writer warp owns a destination chunk pair until its writes and rotation
 * are complete (design section 6.3). `StreamReservation` takes that ownership
 * with a device-scope atomic compare-exchange on the stream's `claim` word.
 * Ownership is taken for every target or for none; a failed reservation
 * releases every acquired claim and leaves every logical counter unchanged.
 *
 * # One warp per block
 *
 * The initial worker layout is one hardware warp per block. All entry points
 * are warp-uniform: every active lane of the calling warp must execute them.
 * `isOneWarpPerBlockLayout()` asserts the layout for tests and startup checks.
 *
 * # Lane publication ordering
 *
 * The lanes of a worker warp write chunk elements through
 * `StreamReservation::writePointer()`, and the leader alone moves the stream
 * slots and hands the publication descriptor downstream. The chain that makes
 * every lane's element stores visible to that downstream consumer is:
 *
 * 1. every lane finishes its element stores;
 * 2. `WarpBarrier::barrier()` (`graph/detail/warp_barrier.hpp`) orders the
 *    lanes against each other. A device-scope release fence cannot stand in
 *    here, because a fence orders only the executing thread's own prior
 *    writes, and a warp vote (`WarpGroup::all`) observes predicates and
 *    orders nothing; and
 * 3. the leader's device-scope release fence orders its own metadata
 *    publication (slot moves, the publication descriptor) after everything
 *    the barrier ordered; a downstream consumer acquires through the
 *    scheduler's acquire chain.
 *
 * `commit()` and `flushPartial()` both execute that chain. A `WarpBarrier`
 * instance is per-warp local state, so every caller owns one and its
 * `refused()` flag must be checked by the caller (a kernel may assert on
 * it); the barrier refuses a multi-warp block instead of falling back to a
 * vote or an over-wide block barrier.
 */
namespace graph
{
    /** Maximum number of destination streams one reservation may target. */
    inline constexpr std::uint32_t maxAssemblyTargets = 8u;

    /** Sentinel index that names no physical chunk slot. */
    inline constexpr std::uint32_t invalidChunkSlot = ~std::uint32_t{0u};

    /** Description of one physical chunk slot in a `ChunkArena`.
     *
     * The slot index is position independent, so a stream state can be built
     * on the host, copied to the device, and used there without pointer
     * fixups. `count` is the logical number of elements currently stored.
     * `handle` is the packet-pool ownership handle that travels downstream
     * with a published chunk.
     */
    template<typename T>
    struct ChunkSlot
    {
        using ValueType = T;

        std::uint32_t slot{invalidChunkSlot};
        std::uint32_t capacity{0u};
        std::uint32_t count{0u};
        PacketHandle handle{};

        ALPAKA_FN_HOST_ACC constexpr bool valid() const noexcept
        {
            return slot != invalidChunkSlot && capacity != 0u;
        }

        ALPAKA_FN_HOST_ACC constexpr bool empty() const noexcept
        {
            return count == 0u;
        }

        friend constexpr bool operator==(ChunkSlot const&, ChunkSlot const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<ChunkSlot<float>>);
    static_assert(std::is_standard_layout_v<ChunkSlot<float>>);

    /** Fixed-capacity backing store for physical chunks.
     *
     * The arena owns no memory. It is a view over caller-provided storage of
     * `slotCount * elementCapacity` elements. Every slot has the same physical
     * capacity `Y == elementCapacity`, matching the two-chunk layout.
     */
    template<typename T>
    struct ChunkArena
    {
        T* storage{nullptr};
        std::uint32_t slotCount{0u};
        std::uint32_t elementCapacity{0u};
        /**
         * Distance between two slots in elements.
         *
         * The pool-backed arena reuses packet-pool slots whose payload stride
         * is larger than the logical chunk capacity. Zero keeps the historic
         * packed layout (`stride == elementCapacity`); the scheduler sets an
         * explicit stride when it overlays the arena on the packet pool.
         */
        std::uint32_t elementStride{0u};

        ALPAKA_FN_HOST_ACC constexpr std::uint32_t stride() const noexcept
        {
            return elementStride != 0u ? elementStride : elementCapacity;
        }

        /** True when `chunk` names a slot of this arena with the arena's capacity. */
        ALPAKA_FN_HOST_ACC constexpr bool contains(ChunkSlot<T> const& chunk) const noexcept
        {
            return storage != nullptr && chunk.slot < slotCount && chunk.capacity != 0u
                   && chunk.capacity == elementCapacity;
        }

        /** Element pointer of the first element of `chunk`. */
        ALPAKA_FN_HOST_ACC constexpr T* data(ChunkSlot<T> const& chunk) const noexcept
        {
            return storage + static_cast<std::size_t>(chunk.slot) * static_cast<std::size_t>(stride());
        }
    };

    /** State of one destination stream.
     *
     * `count` is the logical counter `c`. At every point where an append may
     * start, `0 <= count < capacity`, `first.count == count`, and
     * `second.count == 0`.
     */
    template<typename T>
    struct AssemblyStream
    {
        StreamId id{};
        std::uint32_t capacity{0u};
        std::uint32_t count{0u};
        ChunkSlot<T> first{};
        ChunkSlot<T> second{};
        /** Writers' ownership word: zero is free, nonzero is an owner token. */
        std::uint32_t claim{0u};
        /** Incremented on every committed append or partial flush. */
        std::uint32_t version{0u};
        /** Set when a terminal partial flush detached the last chunk. */
        bool drained{false};
        /**
         * Request seed of the last append.
         *
         * A drain-published partial chunk has no parent job to inherit a
         * request from, so the scheduler remembers the stream's deterministic
         * key here. Single-seed runs use it unchanged; a multi-seed producer
         * would need per-element state instead.
         */
        std::uint64_t seed{0u};
    };

    static_assert(std::is_trivially_copyable_v<AssemblyStream<float>>);
    static_assert(std::is_standard_layout_v<AssemblyStream<float>>);

    /** Result of a reservation attempt. */
    enum class AppendStatus : std::uint32_t
    {
        Ok = 0u,
        NoSuchStream,
        StreamBusy,
        StreamDrained,
        InvalidCapacity,
        InvalidChunk,
        InvalidReplacement,
        CounterOutOfRange,
        SecondChunkNotEmpty,
        InputTooLarge,
        OccupancyOverflow,
        ArithmeticOverflow,
        NotEnoughReplacements,
        NotEnoughPublications,
        InvalidOwnerToken,
        InvalidRequest,
        DuplicateStream,
        NotPending,
        AlreadyPending
    };

    /** One destination and the number of elements the input job emits to it. */
    struct AppendRequest
    {
        StreamId stream{};
        std::uint32_t itemCount{0u};

        friend constexpr bool operator==(AppendRequest const&, AppendRequest const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<AppendRequest>);

    /** A per-destination rank inside its destination group.
     *
     * `rank` is the position among the active lanes that target the same
     * destination in the current round; `size` is the group size. A caller
     * that processes more than `workerWidth()` elements per destination adds
     * the round base to `rank`.
     */
    struct DestinationGroup
    {
        std::uint32_t rank{0u};
        std::uint32_t size{0u};
    };

    /** Result of a partial flush. */
    enum class FlushStatus : std::uint32_t
    {
        Published = 0u,
        NothingToPublish,
        StreamBusy,
        InvalidStream,
        InvalidState,
        InvalidReplacement,
        InvalidOwnerToken
    };

    /** Assembly state view over a caller-owned stream array and chunk arena. */
    template<typename T>
    class StreamAssembler
    {
    public:
        using ValueType = T;

        constexpr StreamAssembler() noexcept = default;

        ALPAKA_FN_HOST_ACC constexpr StreamAssembler(
            AssemblyStream<T>* streams,
            std::uint32_t streamCount,
            ChunkArena<T> arena) noexcept
            : m_streams(streams)
            , m_streamCount(streamCount)
            , m_arena(arena)
        {
        }

        ALPAKA_FN_HOST_ACC constexpr std::uint32_t streamCount() const noexcept
        {
            return m_streamCount;
        }

        ALPAKA_FN_HOST_ACC constexpr ChunkArena<T> const& arena() const noexcept
        {
            return m_arena;
        }

        ALPAKA_FN_HOST_ACC constexpr AssemblyStream<T>& stream(std::uint32_t index) const noexcept
        {
            return m_streams[index];
        }

        ALPAKA_FN_HOST_ACC constexpr bool validStreamIndex(std::uint32_t index) const noexcept
        {
            return index < m_streamCount;
        }

        /** Find a stream by its explicit id. Returns `invalidChunkSlot` if absent. */
        ALPAKA_FN_ACC constexpr std::uint32_t find(StreamId id) const noexcept
        {
            if(!id.isValid())
                return invalidChunkSlot;
            for(std::uint32_t i = 0u; i < m_streamCount; ++i)
            {
                if(m_streams[i].id == id)
                    return i;
            }
            return invalidChunkSlot;
        }

        /** The `2Y - 1` occupancy bound from design section 6.2. */
        static constexpr std::uint64_t maxAppendOccupancy(std::uint64_t y) noexcept
        {
            return 2u * y - 1u;
        }

        /** True when an append of `k` elements fits the two-chunk layout. */
        static constexpr bool appendFits(std::uint64_t y, std::uint64_t c, std::uint64_t k) noexcept
        {
            return y != 0u && c < y && k <= y && c + k <= maxAppendOccupancy(y);
        }

        /** Initialize one stream (warp-uniform, leader writes).
         *
         * The arena capacity is the stream capacity `Y`. Both chunks must be
         * distinct slots of the arena and empty. Returns `false` and changes
         * nothing when the id is already present, the index is occupied, or a
         * chunk is invalid.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC bool initialize(
            T_Acc const& acc,
            WarpGroup<T_Acc> const& warp,
            std::uint32_t index,
            StreamId id,
            ChunkSlot<T> first,
            ChunkSlot<T> second,
            WarpBarrierDiagnostics<T_Acc> const& diagnostics = {}) noexcept
        {
            bool ok = false;
            if(warp.isLeader())
            {
                if(validStreamIndex(index) && id.isValid() && !m_streams[index].id.isValid() && first.valid()
                   && second.valid() && first.slot != second.slot && first.count == 0u && second.count == 0u
                   && m_arena.contains(first) && m_arena.contains(second) && find(id) == invalidChunkSlot)
                {
                    auto& state = m_streams[index];
                    state.id = id;
                    state.capacity = m_arena.elementCapacity;
                    state.count = 0u;
                    state.first = first;
                    state.first.capacity = state.capacity;
                    state.first.count = 0u;
                    state.second = second;
                    state.second.capacity = state.capacity;
                    state.second.count = 0u;
                    state.claim = 0u;
                    state.version = 1u;
                    state.drained = false;
                    state.seed = 0u;
                    ok = true;
                }
            }
            // A leader-only region can leave the warp in two convergence
            // groups under CUDA's independent thread scheduling. Shuffles do
            // not reconverge them, so synchronize before the broadcast.
            auto const blockThreads = static_cast<std::uint32_t>(
                acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
            WarpBarrier<T_Acc> convergence{diagnostics};
            convergence.barrier(acc, blockThreads, warp, barrier_id::assemblyInitialize, ~std::uint32_t{0u}, 0u);
            return broadcastBool(warp, ok);
        }

        /** Ballot-based per-destination rank and group size.
         *
         * Lanes that target the same destination form one group. The rank is
         * computed from the ballot mask of active lanes matching the calling
         * lane's destination. The implementation broadcasts every lane's
         * destination in turn, so the collective is executed uniformly by all
         * lanes; width one is handled by the `WarpGroup` fallback.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC DestinationGroup destinationGroup(
            WarpGroup<T_Acc> const& warp,
            StreamId destination,
            bool active) const noexcept
        {
            using BallotMask = typename WarpGroup<T_Acc>::BallotMask;
            BallotMask mask{0u};
            auto const width = warp.workerWidth();
            for(std::uint32_t src = 0u; src < width; ++src)
            {
                auto const srcActive = warp.template broadcast<std::uint32_t>(active ? 1u : 0u, src) != 0u;
                auto const srcDestination = warp.template broadcast<std::uint32_t>(destination.value, src);
                auto const candidate = warp.ballot(active && destination.value == srcDestination);
                if(srcActive && srcDestination == destination.value)
                    mask = candidate;
            }
            return DestinationGroup{warp.rank(mask), WarpGroup<T_Acc>::popcount(mask)};
        }

        /** Publish the contents of `A` with the real logical count `c`.
         *
         * Used by the drain path. `c == 0` publishes nothing and leaves the
         * stream untouched (`NothingToPublish`). When `c > 0` the current `A`
         * is handed to `published` with `count == c`; the ownership passes
         * downstream. A valid `replacement` becomes the new, empty `A`.
         * Without a replacement the stream becomes terminal (`drained`), which
         * is the expected shape of a final drain.
         *
         * Warp-uniform. The stream is claimed for the duration of the call.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC FlushStatus flushPartial(
            T_Acc const& acc,
            WarpGroup<T_Acc> const& warp,
            std::uint32_t index,
            std::uint32_t ownerToken,
            ChunkSlot<T> replacement,
            ChunkSlot<T>& published,
            WarpBarrierDiagnostics<T_Acc> const& diagnostics = {}) noexcept
        {
            std::uint32_t status = static_cast<std::uint32_t>(FlushStatus::InvalidStream);
            ChunkSlot<T> result{};
            auto const blockThreads = static_cast<std::uint32_t>(
                acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
            // The lanes wrote the chunk elements this flush publishes. The
            // barrier orders those stores across the lanes; it is what makes
            // every lane's writes visible before the leader publishes the
            // descriptor below. The release fence after the barrier then
            // orders the leader's own metadata publication after the ordered
            // element writes.
            WarpBarrier<T_Acc> barrier{diagnostics};
            barrier.barrier(acc, blockThreads, warp, barrier_id::assemblyFlush, ~std::uint32_t{0u}, 0u);
            if(warp.isLeader())
            {
                if(validStreamIndex(index) && ownerToken != 0u)
                {
                    auto& state = m_streams[index];
                    if(tryClaim(acc, &state.claim, ownerToken))
                    {
                        status = static_cast<std::uint32_t>(evaluateFlush(state, replacement, result));
                        if(status == static_cast<std::uint32_t>(FlushStatus::Published))
                        {
                            applyFlush(state, replacement);
                        }
                        // The leader's own release fence: orders its metadata
                        // publication (applyFlush and the claim release) after
                        // the lane element stores the barrier ordered above.
                        // The downstream consumer of the published chunk
                        // acquires through the scheduler's acquire chain.
                        alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
                        releaseClaim(acc, &state.claim, ownerToken);
                    }
                    else
                    {
                        status = static_cast<std::uint32_t>(FlushStatus::StreamBusy);
                    }
                }
                else if(ownerToken == 0u)
                {
                    status = static_cast<std::uint32_t>(FlushStatus::InvalidOwnerToken);
                }
            }
            // Reconverge after the leader-only claim/apply region before any
            // shuffle distributes the result.
            {
                WarpBarrier<T_Acc> convergence{diagnostics};
                convergence.barrier(
                    acc,
                    blockThreads,
                    warp,
                    barrier_id::assemblyFlushConverge,
                    ~std::uint32_t{0u},
                    0u);
            }
            status = warp.template broadcast<std::uint32_t>(status, 0u);
            result.slot = warp.template broadcast<std::uint32_t>(result.slot, 0u);
            result.capacity = warp.template broadcast<std::uint32_t>(result.capacity, 0u);
            result.count = warp.template broadcast<std::uint32_t>(result.count, 0u);
            result.handle.index = warp.template broadcast<std::uint32_t>(result.handle.index, 0u);
            result.handle.generation = warp.template broadcast<std::uint32_t>(result.handle.generation, 0u);
            published = result;
            return static_cast<FlushStatus>(status);
        }

        /** Try to take a stream's writer claim. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static bool tryClaim(T_Acc const& acc, std::uint32_t* claim, std::uint32_t ownerToken) noexcept
        {
            return alpaka::onAcc::atomicCas(acc, claim, 0u, ownerToken, alpaka::onAcc::scope::device) == 0u;
        }

        /** Release a stream's writer claim if it is still ours. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static void releaseClaim(
            T_Acc const& acc,
            std::uint32_t* claim,
            std::uint32_t ownerToken) noexcept
        {
            (void) alpaka::onAcc::atomicCas(acc, claim, ownerToken, 0u, alpaka::onAcc::scope::device);
        }

    private:
        template<typename T_Warp>
        ALPAKA_FN_ACC static bool broadcastBool(WarpGroup<T_Warp> const& warp, bool value) noexcept
        {
            return warp.template broadcast<std::uint32_t>(value ? 1u : 0u, 0u) != 0u;
        }

        ALPAKA_FN_HOST_ACC FlushStatus evaluateFlush(
            AssemblyStream<T> const& state,
            ChunkSlot<T> const& replacement,
            ChunkSlot<T>& result) const noexcept
        {
            if(state.drained || state.capacity == 0u)
                return FlushStatus::InvalidState;
            if(state.count >= state.capacity)
                return FlushStatus::InvalidState;
            if(state.second.count != 0u)
                return FlushStatus::InvalidState;
            if(!m_arena.contains(state.first) || state.first.capacity != state.capacity
               || state.first.count != state.count)
                return FlushStatus::InvalidState;
            if(state.count != 0u && replacement.valid())
            {
                if(!m_arena.contains(replacement) || replacement.count != 0u || replacement.slot == state.first.slot
                   || replacement.slot == state.second.slot)
                    return FlushStatus::InvalidReplacement;
            }
            if(state.count == 0u)
                return FlushStatus::NothingToPublish;
            result = state.first;
            result.count = state.count;
            return FlushStatus::Published;
        }

        ALPAKA_FN_HOST_ACC void applyFlush(AssemblyStream<T>& state, ChunkSlot<T> const& replacement) const noexcept
        {
            if(replacement.valid())
            {
                state.first = replacement;
                state.first.capacity = state.capacity;
                state.first.count = 0u;
            }
            else
            {
                state.first = ChunkSlot<T>{};
                state.second = ChunkSlot<T>{};
                state.drained = true;
            }
            state.count = 0u;
            state.second.count = 0u;
            state.version += 1u;
        }

        AssemblyStream<T>* m_streams{nullptr};
        std::uint32_t m_streamCount{0u};
        ChunkArena<T> m_arena{};
    };

    /** True when the block contains exactly one hardware warp.
     *
     * The initial worker layout is one warp per block; no block-wide barrier is
     * available for independently scheduled warps. This helper is the runtime
     * assertion used by startup checks and tests.
     */
    template<typename T_Acc>
    ALPAKA_FN_ACC bool isOneWarpPerBlockLayout(T_Acc const& acc) noexcept
    {
        auto const blockThreads = acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
        return static_cast<std::uint32_t>(blockThreads[0]) == alpaka::onAcc::warp::getSize(acc);
    }

    /** All-or-nothing reservation of destination stream slots.
     *
     * A reservation is built for one input job. `tryReserve()` acquires every
     * target stream's writer claim and binds one replacement chunk per rotating
     * target and one publication slot per rotating target. If any part fails,
     * every claim is released and no logical counter, chunk, or version is
     * changed.
     *
     * A successful reservation stays `Pending` until `commit()`. This is the
     * explicit continuation for the "reserved but not yet committed" case: the
     * caller may defer and commit later without recomputing the plan.
     * `cancel()` releases the claims without committing.
     *
     * `retireInput()` is the explicit scheduler-side input retirement. It is
     * only valid after a successful commit; the first call succeeds and any
     * further call is recorded by the test-only double-retire detector.
     */
    template<typename T, std::uint32_t T_MaxTargets = maxAssemblyTargets>
    class StreamReservation
    {
    public:
        using ValueType = T;
        static constexpr std::uint32_t maxTargets = T_MaxTargets;

        enum class Status : std::uint32_t
        {
            Empty = 0u,
            Pending,
            Committed,
            Cancelled,
            Failed
        };

        struct Target
        {
            std::uint32_t streamIndex{0u};
            std::uint32_t itemCount{0u};
            std::uint32_t baseCount{0u};
            bool rotate{false};
            ChunkSlot<T> replacement{};
            ChunkSlot<T> publication{};
        };

        constexpr StreamReservation() noexcept = default;

        /** Try to reserve all targets of one input job.
         *
         * @param ownerToken nonzero, unique among concurrent writer warps.
         * @param requests per-destination element counts; their sum must not
         *                 exceed the arena capacity `Y` (one job holds at most
         *                 `Y` elements and emits at most one per input element
         *                 across all outputs).
         * @param replacements bound to rotating targets in array order.
         * @param publications receive the filled `A` chunks at commit time.
         *
         * Warp-uniform; only the leader performs the atomic claim and the
         * state validation. On failure `Pending` is not entered and every
         * target counter is unchanged.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC AppendStatus tryReserve(
            T_Acc const& acc,
            WarpGroup<T_Acc> const& warp,
            std::uint32_t ownerToken,
            StreamAssembler<T> const& assembler,
            AppendRequest const* requests,
            std::uint32_t requestCount,
            ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount,
            ChunkSlot<T>* publications,
            std::uint32_t publicationCount,
            WarpBarrierDiagnostics<T_Acc> const& diagnostics = {}) noexcept
        {
            if(m_status == Status::Pending)
                return AppendStatus::AlreadyPending;
            m_assembler = assembler;
            m_retireCalls = 0u;
            m_inputRetired = false;
            std::uint32_t status = static_cast<std::uint32_t>(AppendStatus::Ok);
            if(warp.isLeader())
            {
                m_targetCount = 0u;
                m_ownerToken = ownerToken;
                m_publications = publications;
                m_publicationCount = publicationCount;
                m_publicationCursor = 0u;
                m_status = Status::Empty;
                auto const result
                    = reserveLeader(acc, ownerToken, requests, requestCount, replacements, replacementCount);
                status = static_cast<std::uint32_t>(result);
                m_status = result == AppendStatus::Ok ? Status::Pending : Status::Failed;
            }
            // Reconverge after the leader-only reserve region before the
            // shuffles that distribute the plan.
            {
                auto const blockThreads = static_cast<std::uint32_t>(
                    acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
                WarpBarrier<T_Acc> convergence{diagnostics};
                convergence.barrier(acc, blockThreads, warp, barrier_id::assemblyReserve, ~std::uint32_t{0u}, 0u);
            }
            status = warp.template broadcast<std::uint32_t>(status, 0u);
            m_targetCount = warp.template broadcast<std::uint32_t>(m_targetCount, 0u);
            m_ownerToken = warp.template broadcast<std::uint32_t>(m_ownerToken, 0u);
            m_status = static_cast<Status>(
                warp.template broadcast<std::uint32_t>(static_cast<std::uint32_t>(m_status), 0u));
            for(std::uint32_t i = 0u; i < m_targetCount; ++i)
            {
                m_targets[i].streamIndex = warp.template broadcast<std::uint32_t>(m_targets[i].streamIndex, 0u);
                m_targets[i].itemCount = warp.template broadcast<std::uint32_t>(m_targets[i].itemCount, 0u);
                m_targets[i].baseCount = warp.template broadcast<std::uint32_t>(m_targets[i].baseCount, 0u);
                m_targets[i].rotate = warp.template broadcast<std::uint32_t>(m_targets[i].rotate ? 1u : 0u, 0u) != 0u;
            }
            return static_cast<AppendStatus>(status);
        }

        ALPAKA_FN_HOST_ACC constexpr Status status() const noexcept
        {
            return m_status;
        }

        ALPAKA_FN_HOST_ACC constexpr bool pending() const noexcept
        {
            return m_status == Status::Pending;
        }

        ALPAKA_FN_HOST_ACC constexpr std::uint32_t targetCount() const noexcept
        {
            return m_targetCount;
        }

        ALPAKA_FN_HOST_ACC constexpr Target const& target(std::uint32_t index) const noexcept
        {
            return m_targets[index];
        }

        ALPAKA_FN_HOST_ACC constexpr std::uint32_t groupSize(std::uint32_t index) const noexcept
        {
            return m_targets[index].itemCount;
        }

        ALPAKA_FN_HOST_ACC constexpr std::uint32_t baseCount(std::uint32_t index) const noexcept
        {
            return m_targets[index].baseCount;
        }

        ALPAKA_FN_HOST_ACC constexpr bool rotates(std::uint32_t index) const noexcept
        {
            return m_targets[index].rotate;
        }

        /** Destination pointer for logical rank `rank` of `target`.
         *
         * Returns `nullptr` when the target or rank is out of range. The
         * optional `physicalIndex` receives the index inside the addressed
         * chunk (`A` or `B`). All active lanes may call this with their own
         * rank; the reservation metadata is broadcast from the leader.
         */
        ALPAKA_FN_HOST_ACC constexpr T* writePointer(
            std::uint32_t target,
            std::uint32_t rank,
            std::uint32_t* physicalIndex = nullptr) const noexcept
        {
            if(target >= m_targetCount || rank >= m_targets[target].itemCount)
                return nullptr;
            auto const& plan = m_targets[target];
            auto const& state = m_assembler.stream(plan.streamIndex);
            std::uint64_t const logical = static_cast<std::uint64_t>(plan.baseCount) + rank;
            if(logical < state.capacity)
            {
                if(physicalIndex != nullptr)
                    *physicalIndex = static_cast<std::uint32_t>(logical);
                return m_assembler.arena().data(state.first) + logical;
            }
            auto const offset = logical - state.capacity;
            if(physicalIndex != nullptr)
                *physicalIndex = static_cast<std::uint32_t>(offset);
            return m_assembler.arena().data(state.second) + offset;
        }

        /** Apply every reserved append and hand filled `A` chunks downstream.
         *
         * The caller must have finished all writes first. Warp-uniform; only
         * the leader mutates state. On success the input packet may be retired
         * with `retireInput()`.
         *
         * The lane element stores written through `writePointer()` are
         * ordered across the lanes by the `WarpBarrier` before the leader
         * moves any slot or hands a publication descriptor downstream; the
         * leader then release-fences its own metadata publication. The
         * downstream consumer of a published chunk acquires through the
         * scheduler's acquire chain. If the `valid` re-check below fails
         * (another warp committed to the same stream between the reservation
         * and this commit), no stream state is mutated at all: the claims are
         * released, the reservation is `Failed`, and the caller may
         * `cancel()` it.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC bool commit(
            T_Acc const& acc,
            WarpGroup<T_Acc> const& warp,
            WarpBarrierDiagnostics<T_Acc> const& diagnostics = {}) noexcept
        {
            bool ok = false;
            auto const blockThreads = static_cast<std::uint32_t>(
                acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
            // Order the lanes: the barrier is what makes every lane's element
            // stores ordered before the leader publishes below. A vote
            // (`warp.all`) would observe predicates and order nothing, and a
            // device-scope release fence orders only the executing thread's
            // own prior writes, so neither could publish the other lanes'
            // stores.
            WarpBarrier<T_Acc> barrier{diagnostics};
            barrier.barrier(acc, blockThreads, warp, barrier_id::assemblyCommit, ~std::uint32_t{0u}, 0u);
            if(warp.isLeader())
            {
                if(m_status == Status::Pending)
                {
                    bool valid = true;
                    for(std::uint32_t i = 0u; i < m_targetCount; ++i)
                    {
                        auto const& state = m_assembler.stream(m_targets[i].streamIndex);
                        if(state.claim != m_ownerToken || state.count != m_targets[i].baseCount)
                            valid = false;
                    }
                    if(valid)
                    {
                        // The leader's own release fence: orders its metadata
                        // publication (the slot moves in applyTarget and the
                        // publication descriptors) after the lane element
                        // stores the barrier ordered. The downstream consumer
                        // acquires through the scheduler's acquire chain.
                        alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
                        for(std::uint32_t i = 0u; i < m_targetCount; ++i)
                            applyTarget(i);
                        ok = true;
                    }
                    for(std::uint32_t i = 0u; i < m_targetCount; ++i)
                    {
                        auto& state = m_assembler.stream(m_targets[i].streamIndex);
                        StreamAssembler<T>::releaseClaim(acc, &state.claim, m_ownerToken);
                    }
                    m_status = ok ? Status::Committed : Status::Failed;
                }
            }
            // Reconverge after the leader-only apply region before the result
            // is distributed through a shuffle.
            {
                WarpBarrier<T_Acc> convergence{diagnostics};
                convergence.barrier(
                    acc,
                    blockThreads,
                    warp,
                    barrier_id::assemblyCommitConverge,
                    ~std::uint32_t{0u},
                    0u);
            }
            ok = warp.template broadcast<std::uint32_t>(ok ? 1u : 0u, 0u) != 0u;
            m_status = static_cast<Status>(
                warp.template broadcast<std::uint32_t>(static_cast<std::uint32_t>(m_status), 0u));
            return ok;
        }

        /** Release all claims without committing.
         *
         * No barrier is needed here: the claims are the only thing a
         * concurrent reader observes, and the claim CAS already orders them.
         * The broadcast at the end is the convergence point, not an ordering
         * primitive.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC void cancel(
            T_Acc const& acc,
            WarpGroup<T_Acc> const& warp,
            WarpBarrierDiagnostics<T_Acc> const& diagnostics = {}) noexcept
        {
            if(warp.isLeader())
            {
                if(m_status == Status::Pending)
                {
                    for(std::uint32_t i = 0u; i < m_targetCount; ++i)
                    {
                        auto& state = m_assembler.stream(m_targets[i].streamIndex);
                        StreamAssembler<T>::releaseClaim(acc, &state.claim, m_ownerToken);
                    }
                }
                m_status = Status::Cancelled;
            }
            // Reconverge after the leader-only claim release before the status
            // shuffle.
            {
                auto const blockThreads = static_cast<std::uint32_t>(
                    acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
                WarpBarrier<T_Acc> convergence{diagnostics};
                convergence.barrier(acc, blockThreads, warp, barrier_id::assemblyCancel, ~std::uint32_t{0u}, 0u);
            }
            m_status = static_cast<Status>(
                warp.template broadcast<std::uint32_t>(static_cast<std::uint32_t>(m_status), 0u));
        }

        /** Retire the input packet exactly once after a successful commit.
         *
         * Returns `true` for the single valid retirement. A second call after
         * a commit returns `false` and makes `doubleRetireDetected()` true.
         * Calls before a commit are rejected and do not count.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC bool retireInput(WarpGroup<T_Acc> const& warp) noexcept
        {
            bool firstTime = false;
            if(warp.isLeader())
            {
                if(m_status == Status::Committed)
                {
                    m_retireCalls += 1u;
                    if(!m_inputRetired)
                    {
                        m_inputRetired = true;
                        firstTime = true;
                    }
                }
            }
            firstTime = warp.template broadcast<std::uint32_t>(firstTime ? 1u : 0u, 0u) != 0u;
            m_retireCalls = warp.template broadcast<std::uint32_t>(m_retireCalls, 0u);
            m_inputRetired = warp.template broadcast<std::uint32_t>(m_inputRetired ? 1u : 0u, 0u) != 0u;
            return firstTime;
        }

        /** Test-only double-retire detector. */
        ALPAKA_FN_HOST_ACC constexpr bool doubleRetireDetected() const noexcept
        {
            return m_retireCalls > 1u;
        }

        ALPAKA_FN_HOST_ACC constexpr std::uint32_t retireCalls() const noexcept
        {
            return m_retireCalls;
        }

    private:
        template<typename T_Acc>
        ALPAKA_FN_ACC AppendStatus reserveLeader(
            T_Acc const& acc,
            std::uint32_t ownerToken,
            AppendRequest const* requests,
            std::uint32_t requestCount,
            ChunkSlot<T> const* replacements,
            std::uint32_t replacementCount) noexcept
        {
            if(ownerToken == 0u)
                return AppendStatus::InvalidOwnerToken;
            if(requestCount > T_MaxTargets)
                return AppendStatus::InvalidRequest;
            if(requestCount == 0u)
                return AppendStatus::Ok;
            if(requests == nullptr)
                return AppendStatus::InvalidRequest;

            std::uint32_t replacementCursor = 0u;
            std::uint64_t totalItems = 0u;
            for(std::uint32_t i = 0u; i < requestCount; ++i)
            {
                auto const index = m_assembler.find(requests[i].stream);
                if(index == invalidChunkSlot)
                {
                    releaseClaims(acc, i);
                    return AppendStatus::NoSuchStream;
                }
                for(std::uint32_t j = 0u; j < i; ++j)
                {
                    if(m_targets[j].streamIndex == index)
                    {
                        releaseClaims(acc, i);
                        return AppendStatus::DuplicateStream;
                    }
                }

                auto& state = m_assembler.stream(index);
                if(!StreamAssembler<T>::tryClaim(acc, &state.claim, ownerToken))
                {
                    releaseClaims(acc, i);
                    return AppendStatus::StreamBusy;
                }
                m_targets[i].streamIndex = index;

                auto const failure = validateTarget(index, requests[i].itemCount);
                if(failure != AppendStatus::Ok)
                {
                    releaseClaims(acc, i + 1u);
                    return failure;
                }

                totalItems += requests[i].itemCount;
                if(totalItems > static_cast<std::uint64_t>(m_assembler.arena().elementCapacity))
                {
                    releaseClaims(acc, i + 1u);
                    return AppendStatus::InputTooLarge;
                }

                std::uint64_t const sum = static_cast<std::uint64_t>(state.count) + requests[i].itemCount;
                if(sum > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
                {
                    releaseClaims(acc, i + 1u);
                    return AppendStatus::ArithmeticOverflow;
                }
                if(sum > StreamAssembler<T>::maxAppendOccupancy(state.capacity))
                {
                    releaseClaims(acc, i + 1u);
                    return AppendStatus::OccupancyOverflow;
                }

                bool const rotate = sum >= state.capacity;
                if(rotate)
                {
                    if(replacements == nullptr || replacementCursor >= replacementCount)
                    {
                        releaseClaims(acc, i + 1u);
                        return AppendStatus::NotEnoughReplacements;
                    }
                    auto const replacement = replacements[replacementCursor];
                    if(!replacementAvailable(replacement, i, state.capacity))
                    {
                        releaseClaims(acc, i + 1u);
                        return AppendStatus::InvalidReplacement;
                    }
                    m_targets[i].replacement = replacement;
                    replacementCursor += 1u;
                    if(m_publications == nullptr || m_publicationCursor >= m_publicationCount)
                    {
                        releaseClaims(acc, i + 1u);
                        return AppendStatus::NotEnoughPublications;
                    }
                }

                m_targets[i].itemCount = requests[i].itemCount;
                m_targets[i].baseCount = state.count;
                m_targets[i].rotate = rotate;
            }
            m_targetCount = requestCount;
            return AppendStatus::Ok;
        }

        ALPAKA_FN_HOST_ACC AppendStatus validateTarget(std::uint32_t index, std::uint32_t itemCount) const noexcept
        {
            auto const& state = m_assembler.stream(index);
            if(state.drained)
                return AppendStatus::StreamDrained;
            if(state.capacity == 0u)
                return AppendStatus::InvalidCapacity;
            if(!m_assembler.arena().contains(state.first) || !m_assembler.arena().contains(state.second)
               || state.first.capacity != state.capacity || state.second.capacity != state.capacity)
                return AppendStatus::InvalidChunk;
            if(state.count >= state.capacity)
                return AppendStatus::CounterOutOfRange;
            if(!state.second.empty())
                return AppendStatus::SecondChunkNotEmpty;
            if(state.first.count != state.count)
                return AppendStatus::InvalidChunk;
            if(itemCount > state.capacity)
                return AppendStatus::InputTooLarge;
            return AppendStatus::Ok;
        }

        ALPAKA_FN_HOST_ACC bool replacementAvailable(
            ChunkSlot<T> const& replacement,
            std::uint32_t targetIndex,
            std::uint32_t capacity) const noexcept
        {
            if(!m_assembler.arena().contains(replacement) || replacement.capacity != capacity
               || replacement.count != 0u)
                return false;
            for(std::uint32_t i = 0u; i < m_assembler.streamCount(); ++i)
            {
                auto const& state = m_assembler.stream(i);
                if(state.first.slot == replacement.slot || state.second.slot == replacement.slot)
                    return false;
            }
            for(std::uint32_t i = 0u; i < targetIndex; ++i)
            {
                if(m_targets[i].replacement.slot == replacement.slot)
                    return false;
            }
            return true;
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC void releaseClaims(T_Acc const& acc, std::uint32_t count) noexcept
        {
            for(std::uint32_t i = 0u; i < count; ++i)
            {
                auto& state = m_assembler.stream(m_targets[i].streamIndex);
                StreamAssembler<T>::releaseClaim(acc, &state.claim, m_ownerToken);
            }
            m_targetCount = 0u;
        }

        ALPAKA_FN_ACC void applyTarget(std::uint32_t index) noexcept
        {
            auto& state = m_assembler.stream(m_targets[index].streamIndex);
            auto const sum = m_targets[index].baseCount + m_targets[index].itemCount;
            if(m_targets[index].rotate)
            {
                auto published = state.first;
                published.count = state.capacity;
                if(m_publications != nullptr && m_publicationCursor < m_publicationCount)
                    m_publications[m_publicationCursor++] = published;
                state.first = state.second;
                state.first.capacity = state.capacity;
                state.first.count = sum - state.capacity;
                state.second = m_targets[index].replacement;
                state.second.capacity = state.capacity;
                state.second.count = 0u;
                state.count = sum - state.capacity;
                m_targets[index].publication = published;
            }
            else
            {
                state.first.capacity = state.capacity;
                state.first.count = sum;
                state.second.count = 0u;
                state.count = sum;
            }
            state.version += 1u;
        }

        StreamAssembler<T> m_assembler{};
        Target m_targets[T_MaxTargets]{};
        std::uint32_t m_targetCount{0u};
        std::uint32_t m_ownerToken{0u};
        ChunkSlot<T>* m_publications{nullptr};
        std::uint32_t m_publicationCount{0u};
        std::uint32_t m_publicationCursor{0u};
        Status m_status{Status::Empty};
        std::uint32_t m_retireCalls{0u};
        bool m_inputRetired{false};
    };
} // namespace graph
