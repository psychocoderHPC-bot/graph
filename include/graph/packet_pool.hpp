// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/concepts.hpp>
#include <graph/detail/device_atomic.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/detail/warp_handoff.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Fixed-capacity packet slot pool with explicit ownership.
 *
 * The pool owns one preallocated device buffer of `T_SlotCapacity` slots, each
 * `T_ElementCapacity` bytes of element storage wide. It hands out slot
 * ownership with `acquire()`/`release()`; a `PacketHandle` names a slot plus
 * the generation the pool had when it handed the slot out, so a stale handle
 * that still names the index is detectable instead of silently observing a
 * different packet.
 *
 * # Ring slot reuse versus packet reuse
 *
 * These are separate events and the API keeps them separate:
 *
 * - The ready ring reuses a *ring slot* when the entry it carries has been
 *   claimed. The ring stamp advances and the ring slot is writable again; the
 *   packet the entry named is untouched.
 * - The packet pool reuses a *packet slot* when its owner calls `release()`.
 *   The pool increments the slot's generation and the slot is acquirable
 *   again; the ring is untouched.
 *
 * A job that reads a packet must keep its handle until the last read
 * completes; releasing earlier is a use-after-release that the generation
 * check detects in test builds.
 *
 * # Cross-warp acquisition protocol
 *
 * Every mutable free-list field is a device-scope atomic word. The free list
 * is a singly linked stack of slots; the `next` pointer of a free slot lives in
 * the slot's own free-list field, and the head is a tagged word.
 *
 * - `freeListHead` packs a 32-bit slot index with a 32-bit epoch. A push (release)
 *   CAS-exchanges `index | (tag << 31)` into the head; a pop (acquire) reads
 *   the head, walks the slot's `next`, and CAS-exchanges the successor into the
 *   head. The tag distinguishes `invalidSlotIndex` (the terminator) from a
 *   real slot, so a pop never hands out the terminator and the ABA window is
 *   closed by the tag itself: a slot can only reappear at the head through a
 *   push that also advances the tag, and a stale pop's CAS fails against it.
 * - `freeCount` is a device-scope atomic 64-bit counter for capacity checks;
 *   `violations` is a device-scope atomic 64-bit counter for test hooks.
 * - Slot bookkeeping (`generation`, `active`) is written only by the winner of
 *   the head CAS, so no two warps ever update the same slot's meta concurrently.
 *
 * The winner publishes the slot's record with a device-scope release fence
 * before the head CAS that makes the slot visible, and every acquire that
 * observes the new head performs a matching acquire against the head read.
 * The generation is advanced exactly once per hand-out (in the winning CAS
 * path), so a slot's generation is monotonic and each issued handle is
 * distinct.
 *
 * `acquire()` and `release()` are warp-uniform: every active lane of the
 * calling warp calls the same method with the same arguments, and only the
 * lane selected by `WarpGroup::isLeader()` performs the atomic work. The
 * result is published through one block-shared slot followed by a block
 * barrier, never through a warp shuffle: the CAS retry loop is a leader-only
 * region and the pinned alpaka's shuffle is driven by `__activemask()`, which
 * does not reconverge the warp. See `graph/detail/warp_handoff.hpp`.
 *
 * # Capacity budget (spec 6)
 *
 * The pool is sized by the documented formula `requiredSlots(P, B, W, R, S)`:
 *
 * ```
 * requiredSlots = P + 2*B + W*R + S
 * ```
 *
 * where
 *
 * - `P` is the live-particle admission limit: each ordinary detached nonempty
 *   packet can be charged to a distinct admitted particle, so at most `P`
 *   slots carry live particle data.
 * - `B` is the number of assembly streams. Each stream keeps two physical
 *   output chunks (`A`, `B`), so `2*B` slots.
 * - `W` is the number of concurrent workers and `R` is the maximum number of
 *   detached replacement or scratch packets one worker may hold at once, so
 *   `W*R` slots for temporary allocations and copied-but-uncommitted packets.
 * - `S` is the control/source allowance: slots for control packets and source
 *   request staging that do not carry admitted particles.
 *
 * `PacketPool::configure()` verifies at startup that the configured
 * `T_SlotCapacity` satisfies the formula for the requested budget and fails
 * loudly (records a violation, exposes it through `ownershipViolations()`, and
 * leaves no slot acquirable) when it does not. There is no silent clamping.
 *
 * # Storage
 *
 * No allocation after startup. All storage comes from a caller-provided device
 * buffer (an alpaka buffer) so the pool can be preallocated before the
 * persistent launch. The host sizes the buffer with `storageBytes()`, allocates
 * it, and hands the pointer to the startup action, which calls `configure()`.
 */
namespace graph
{
    /** Slot index type: an index into the pool's slot array. */
    using SlotIndex = std::uint32_t;

    /** Invalid slot index: no slot, and the free-list terminator. */
    inline constexpr SlotIndex invalidSlotIndex = ~SlotIndex{0u};

    /** One slot's bookkeeping, kept alongside the pool's slot array.
     *
     * `generation` starts at zero and increments on every reuse (that is, on
     * every successful `acquire()`), so a handle whose generation does not
     * match the slot's current generation names a slot that has been reused.
     * The pool records an ownership violation for such a mismatch (see
     * `PacketPool::ownershipViolations()`).
     *
     * `active` is set while a slot is held and cleared on release; a release
     * on a slot that is not active is a double-release and records a
     * violation.
     *
     * Both fields are written only by the winner of the free-list head CAS,
     * so no two warps update the same slot concurrently; the head CAS is the
     * synchronization that protects them.
     */
    struct SlotMeta
    {
        std::uint32_t generation{0u};
        std::uint32_t active{0u};
    };

    static_assert(std::is_trivially_copyable_v<SlotMeta>);
    static_assert(std::is_standard_layout_v<SlotMeta>);

    /** Compile-time capacity budget parameters (spec 6). */
    struct PacketPoolBudget
    {
        /** Live-particle admission limit `P`. */
        std::uint32_t liveParticles{0u};
        /** Assembly streams `B` (each keeps two physical chunks). */
        std::uint32_t assemblyStreams{0u};
        /** Concurrent workers `W`. */
        std::uint32_t workers{0u};
        /** Maximum detached replacement or scratch packets per worker `R`. */
        std::uint32_t replacementsPerWorker{0u};
        /** Control/source allowance `S`. */
        std::uint32_t controlAllowance{0u};

        friend constexpr bool operator==(PacketPoolBudget const&, PacketPoolBudget const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<PacketPoolBudget>);

    /** Slot count required by the budget formula.
     *
     * ```
     * requiredSlots = P + 2*B + W*R + S
     * ```
     *
     * The formula is the conservative capacity budget from spec 6. It is
     * compile-time-checkable (all inputs are integers) and runtime-visible
     * (the pool exposes it through `requiredSlots()`).
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t requiredSlots(
        std::uint32_t P,
        std::uint32_t B,
        std::uint32_t W,
        std::uint32_t R,
        std::uint32_t S = 0u) noexcept
    {
        return P + 2u * B + W * R + S;
    }

    ALPAKA_FN_HOST_ACC constexpr std::uint32_t requiredSlots(PacketPoolBudget const& b) noexcept
    {
        return requiredSlots(
            b.liveParticles,
            b.assemblyStreams,
            b.workers,
            b.replacementsPerWorker,
            b.controlAllowance);
    }

    namespace pool_detail
    {
        /** Round `value` up to a multiple of `alignment` (both powers of two). */
        constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        /** Device-scope release fence. */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr void publishRelease(T_Acc const& acc) noexcept
        {
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
        }

        /** Device-scope acquire fence: orders the payload reads that follow
         *  against the release that preceded the observed head advance. */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr void claimAcquire(T_Acc const& acc) noexcept
        {
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);
        }

        /** Device-scope atomic compare-exchange on a naturally aligned word. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC constexpr T atomicCasDevice(T_Acc const& acc, T* addr, T expected, T desired) noexcept
        {
            return alpaka::onAcc::atomicCas(acc, addr, expected, desired, alpaka::onAcc::scope::device);
        }

        /** Device-scope atomic add. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC constexpr T atomicAddDevice(T_Acc const& acc, T* addr, T value) noexcept
        {
            return alpaka::onAcc::atomicAdd(acc, addr, value, alpaka::onAcc::scope::device);
        }
    } // namespace pool_detail

    /** Fixed-capacity packet slot pool over preallocated device storage.
     *
     * `T_SlotCapacity` is the number of slots; `T_ElementCapacity` is the
     * physical element storage of each slot in bytes. Both are compile time.
     *
     * All public methods are device-side (`ALPAKA_FN_ACC`) and warp-uniform:
     * every active lane of the calling warp must call the same method with the
     * same arguments, and only the lane selected by `WarpGroup::isLeader()`
     * performs the atomic work.
     *
     * The pool is a view over preallocated storage: it performs no allocation
     * and never outlives the storage it references.
     */
    template<typename T_Acc, std::size_t T_SlotCapacity = 1024u, std::size_t T_ElementCapacity = 4096u>
    class PacketPool
    {
        static_assert(T_SlotCapacity >= 1u);
        static_assert(T_ElementCapacity >= 1u);
        // The tagged head uses 31 index bits; a slot array larger than 2^31
        // would not fit in the tag field.
        static_assert(T_SlotCapacity < (1u << 31u), "slot capacity must fit the tagged free-list head");

    public:
        using Acc = std::remove_cvref_t<T_Acc>;

        static constexpr std::size_t slotCapacity = T_SlotCapacity;
        static constexpr std::size_t elementCapacity = T_ElementCapacity;

        /** One slot: the element storage plus the per-slot bookkeeping.
         *
         * The pool is element-type-agnostic: the storage is raw bytes and the
         * typed views are reconstructed through the vocabulary
         * (`PacketRecord` schema + `makePacketView`).
         *
         * Free-list encoding: while a slot is free, its `record.handle.index`
         * holds the next free slot index (`invalidSlotIndex` terminates the
         * list) and its `record.handle.generation` is zero. While a slot is
         * held, `record` describes its payload and the free-list fields are
         * not used. The free list is a singly linked stack over the slot
         * array; no separate next-pointer array is needed.
         */
        struct PoolSlot
        {
            std::uint8_t data[T_ElementCapacity]{};
            SlotMeta meta{};
            PacketRecord record{};
        };

        static_assert(std::is_trivially_copyable_v<PoolSlot>);
        static_assert(std::is_standard_layout_v<PoolSlot>);

        /** Pool header. Each word owns its own 64-byte cache line.
         *
         * `freeListHead` is a tagged word: bits 32..63 are an epoch, bits
         * 0..31 are the slot index. `0` (index 0, tag 0) is the initial head after
         * `configure()`; the terminator is `1u << 31` (tag 1, no index).
         * `freeCount`, `violations`, and `headVersion` are device-scope atomic
         * words updated with `atomicAdd`/`atomicCas` from multiple warps.
         */
        struct PoolHeader
        {
            /**
             * Free-list head: low 32 bits are the slot index, high 32 bits are
             * a monotonically increasing epoch.
             *
             * The epoch is what closes the ABA window. A one-bit tag derived
             * from the successor (the previous form) allowed a stale CAS to
             * succeed after the head had cycled back to the same index, and
             * handed one slot to two owners. Every successful head change now
             * bumps the epoch.
             */
            std::uint64_t freeListHead{0u};
            std::uint64_t padHead[7]{};
            std::uint64_t freeCount{0u};
            std::uint64_t padFree[7]{};
            std::uint64_t violations{0u};
            std::uint64_t padViolations[7]{};
            std::uint32_t headVersion{0u};
            std::uint64_t padVersion[7]{};
            // Diagnostic counters (test builds): CAS failure and meta-check
            // counts. Zero in steady state; nonzero indicates a protocol bug.
            std::uint64_t casFailures{0u};
            std::uint64_t padCasFail[7]{};
            std::uint64_t metaChecks{0u};
            std::uint64_t padMeta[7]{};
        };

        static_assert(std::is_trivially_copyable_v<PoolHeader>);
        static_assert(std::is_standard_layout_v<PoolHeader>);

        static constexpr std::size_t headerBytes = pool_detail::alignUp(sizeof(PoolHeader), 64u);
        static constexpr std::size_t slotBytes = sizeof(PoolSlot);
        static constexpr std::size_t storageBytes = headerBytes + T_SlotCapacity * slotBytes;

        /** One acquisition result, distributed through block-shared memory.
         *
         * The leader writes it once; every lane reads the same bytes after the
         * handoff barrier. `valid == 0` leaves `handle` unspecified, so a
         * caller must consult `valid` first.
         */
        struct Claim
        {
            PacketHandle handle{};
            std::uint32_t valid{0u};
        };

        static_assert(std::is_trivially_copyable_v<Claim>);
        static_assert(std::is_standard_layout_v<Claim>);

        /** Construct a view over preallocated device storage of at least
         *  `storageBytes` bytes. The storage must stay valid and exclusively
         *  owned by this pool for its lifetime. */
        ALPAKA_FN_ACC constexpr explicit PacketPool(void* storage) noexcept : m_storage(static_cast<char*>(storage))
        {
        }

        /** Initialize the pool for the given budget.
         *
         * The configured `T_SlotCapacity` must satisfy
         * `requiredSlots(budget)`; when it does not, the call fails loudly: it
         * records a violation (visible through `ownershipViolations()`),
         * leaves no slot acquirable, and returns false. There is no silent
         * clamping. All lanes call it warp-uniformly; only the leader writes.
         * Returns `true` on success.
         */
        ALPAKA_FN_ACC constexpr bool configure(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            PacketPoolBudget const& budget) noexcept
        {
            bool ok = true;
            if(warp.isLeader())
            {
                auto* hdr = &header();
                if(requiredSlots(budget) > static_cast<std::uint32_t>(T_SlotCapacity))
                {
                    hdr->freeListHead = encodeHead(invalidSlotIndex, 0u);
                    hdr->freeCount = 0u;
                    hdr->violations = 1u;
                    hdr->headVersion = 0u;
                    hdr->casFailures = 0u;
                    hdr->metaChecks = 0u;
                    // Initialize slot metas so a misconfigured pool is still in
                    // a defined (if unusable) state.
                    for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                    {
                        slotAt(i).meta.generation = 0u;
                        slotAt(i).meta.active = 0u;
                    }
                    pool_detail::publishRelease(acc);
                    ok = false;
                }
                else
                {
                    hdr->violations = 0u;
                    hdr->headVersion = 0u;
                    hdr->casFailures = 0u;
                    hdr->metaChecks = 0u;
                    hdr->freeCount = static_cast<std::uint64_t>(T_SlotCapacity);
                    hdr->freeListHead = encodeHead(0u, 0u); // index 0, epoch 0
                    for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                    {
                        auto* slot = &slotAt(i);
                        slot->meta.generation = 0u;
                        slot->meta.active = 0u;
                        // Free-list encoding: next = i+1, last slot terminates.
                        slot->record.handle.index
                            = (i + 1u < T_SlotCapacity) ? static_cast<SlotIndex>(i + 1u) : invalidSlotIndex;
                        slot->record.handle.generation = 0u;
                        slot->record.elementCount = 0u;
                        slot->record.capacity = static_cast<std::uint32_t>(T_ElementCapacity);
                        slot->record.schema = ElementSchema{};
                    }
                    // Publish every slot's free-list state and the initial head
                    // before any warp may pop.
                    pool_detail::publishRelease(acc);
                }
            }
            WarpHandoff<Acc, std::uint32_t, handoff_id::packetPoolConfigure> handoff;
            handoff.publish(acc, warp, static_cast<std::uint32_t>(ok ? 1u : 0u));
            handoff.sync(acc);
            return handoff.get(acc) != 0u;
        }

        /** Acquire one slot.
         *
         * Warp-uniform. On success, `out` holds the handle (index plus the
         * slot's new generation) and the slot's `record` is reset so the
         * caller fills in its schema and element count. The leader performs
         * the atomic work, writes `{handle, valid}` into the block-shared
         * `Claim`, and every lane reads it after the barrier.
         *
         * Returns `true` on success, `false` when no slot is free.
         */
        ALPAKA_FN_ACC constexpr bool acquire(Acc const& acc, WarpGroup<Acc> const& warp, PacketHandle& out) noexcept
        {
            WarpHandoff<Acc, Claim, handoff_id::packetPoolClaim> handoff;
            Claim claim{};
            if(warp.isLeader())
            {
                PacketHandle handle{};
                claim.valid = popSlot(acc, handle) ? 1u : 0u;
                claim.handle = handle;
            }
            handoff.publish(acc, warp, claim);
            handoff.sync(acc);
            Claim const result = handoff.get(acc);
            out = result.handle;
            return result.valid != 0u;
        }

        /** Release one slot.
         *
         * Warp-uniform. The handle must name a slot the caller owns (its
         * generation must match the slot's current generation and the slot
         * must be active). On success the slot's `meta.active` is cleared, its
         * generation is incremented (so a stale handle no longer matches), and
         * the slot is pushed back onto the free list.
         *
         * A release with a mismatched generation (use-after-release) or a
         * release of an inactive slot (double-release) records an ownership
         * violation in `violations`, leaves the free list untouched, and
         * returns false.
         */
        ALPAKA_FN_ACC constexpr bool release(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            PacketHandle const& handle) noexcept
        {
            WarpHandoff<Acc, std::uint32_t, handoff_id::packetPoolRelease> handoff;
            std::uint32_t ok = 0u;
            if(warp.isLeader())
                ok = pushSlot(acc, handle) ? 1u : 0u;
            handoff.publish(acc, warp, ok);
            handoff.sync(acc);
            return handoff.get(acc) != 0u;
        }

        /** True when `handle` names a slot the pool currently considers owned
         *  by the caller (generation matches, slot active). This is the
         *  generation-mismatch check that makes a use-after-release
         *  detectable in test builds. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr bool isValid(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            PacketHandle const& handle) const noexcept
        {
            (void) acc;
            (void) warp;
            auto const& slot = slotAt(handle.index);
            return slot.meta.generation == handle.generation && slot.meta.active == 1u;
        }

        /** Raw element pointer for a slot. The caller must hold the slot
         *  (the handle must be valid). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint8_t* slotData(PacketHandle const& handle) const noexcept
        {
            return slotAt(handle.index).data;
        }

        /** The slot's current record (the caller must hold the slot). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr PacketRecord const& slotRecord(PacketHandle const& handle) const noexcept
        {
            return slotAt(handle.index).record;
        }

        /** Coherent read of the generation the pool currently holds for a slot.
         *
         * The record is written by the warp that won the slot in *another*
         * block, and the reading block may already hold a stale line: every
         * loser of the free-list CAS reads the popped slot's record while
         * walking the list. The read must therefore be the synchronizing
         * acquire load that matches the writer's release, not a weak load.
         */
        [[nodiscard]] ALPAKA_FN_ACC std::uint32_t observedGeneration(Acc const& acc, SlotIndex index) const noexcept
        {
            (void) acc;
            return detail::dev::loadAcquire(&slotAt(index).record.handle.generation);
        }

        /** Coherent snapshot of a slot's record.
         *
         * Same reason as `observedGeneration`. The word-wise acquire copy is
         * once per job and is the only way to read `schema`, `capacity`, and
         * `elementCount` that another block published.
         */
        [[nodiscard]] ALPAKA_FN_ACC PacketRecord observedRecord(Acc const& acc, PacketHandle const& handle)
            const noexcept
        {
            static_assert(sizeof(PacketRecord) % sizeof(std::uint32_t) == 0u);
            PacketRecord result{};
            auto const* source = reinterpret_cast<std::uint32_t const*>(&slotAt(handle.index).record);
            auto* destination = reinterpret_cast<std::uint32_t*>(&result);
            for(std::size_t word = 0u; word < sizeof(PacketRecord) / sizeof(std::uint32_t); ++word)
                destination[word] = detail::dev::loadAcquire(&source[word]);
            return result;
        }

        /** Set the logical element count of a slot's record. The caller must
         *  hold the slot. Publishes with a device-scope release so a consumer
         *  that acquires the slot observes the new count. */
        ALPAKA_FN_ACC constexpr void setElementCount(
            Acc const& acc,
            PacketHandle const& handle,
            std::uint32_t count) noexcept
        {
            slotAt(handle.index).record.elementCount = count;
            pool_detail::publishRelease(acc);
        }

        /** Reconstruct a typed view from a slot through the vocabulary.
         *
         * The caller must hold the slot and the slot's record must describe a
         * payload of element type `T` and rank `T_rank`. The returned view
         * borrows the pool's storage; it stays valid until the slot is
         * released and reused.
         */
        template<typename T, std::size_t T_rank = 1u>
        [[nodiscard]] ALPAKA_FN_ACC constexpr auto makeView(PacketHandle const& handle) const noexcept
        {
            auto const& slot = slotAt(handle.index);
            return makePacketView<T, T_rank>(slot.record, static_cast<T*>(slot.data));
        }

        /** Number of free slots. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t freeCount() const noexcept
        {
            return header().freeCount;
        }

        /** Number of slots currently held. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t heldCount() const noexcept
        {
            return T_SlotCapacity - header().freeCount;
        }

        /** Ownership violations recorded by the test hook. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t ownershipViolations() const noexcept
        {
            return header().violations;
        }

        /** CAS failures during pop (diagnostic; zero in a correct run). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t casFailures() const noexcept
        {
            return header().casFailures;
        }

        /** Number of pop attempts that passed the terminator/index check (diagnostic). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t metaChecks() const noexcept
        {
            return header().metaChecks;
        }

    private:
        static constexpr std::uint32_t headInvalidIndex = invalidSlotIndex;

        /** Encode a slot index plus epoch into the 64-bit head word. */
        static constexpr std::uint64_t encodeHead(std::uint32_t index, std::uint32_t epoch) noexcept
        {
            return (static_cast<std::uint64_t>(epoch) << 32u) | static_cast<std::uint64_t>(index);
        }

        /** Decode the slot index of a head word. */
        static constexpr std::uint32_t headIndex(std::uint64_t head) noexcept
        {
            return static_cast<std::uint32_t>(head);
        }

        /** Decode the epoch of a head word. */
        static constexpr std::uint32_t headEpoch(std::uint64_t head) noexcept
        {
            return static_cast<std::uint32_t>(head >> 32u);
        }

        /** True when the head word is the terminator (no free slot). */
        static constexpr bool headIsTerminator(std::uint64_t head) noexcept
        {
            return headIndex(head) == headInvalidIndex;
        }

        /** Maximum number of CAS retries in popSlot. A failed CAS means
         *  another warp advanced the head; the next iteration reads the new
         *  head and retries. 2^20 is more than enough for realistic
         *  contention (16-1024 warps on a 64-slot pool). */
        static constexpr std::uint32_t maxPopAttempts = 1u << 20u;

        /** Pop one slot from the free list via a CAS on the tagged head.
         *
         * The winner advances the head, increments the slot's generation and
         * sets `active`, and publishes the record with a device-scope release
         * before the head CAS. Returns false when the pool is empty.
         *
         * The CAS is retried up to `T_MaxPopAttempts` times: under multi-warp
         * contention a CAS failure means another warp advanced the head first,
         * so the next iteration reads the new head and retries. The meta and
         * record stores are performed only after a successful CAS to avoid
         * corrupting the slot's free-list pointer (which encodes the
         * successor) before the pop is committed.
         */
        ALPAKA_FN_ACC constexpr bool popSlot(Acc const& acc, PacketHandle& out) const noexcept
        {
            auto* hdr = &header();
            for(std::uint32_t attempt = 0u; attempt < maxPopAttempts; ++attempt)
            {
                // Read the head with a raw volatile load: the value is only a
                // candidate for the CAS that follows, and a stale read is safe
                // because the CAS re-validates it atomically.
                std::uint64_t const expected = *reinterpret_cast<std::uint64_t volatile*>(&hdr->freeListHead);
                if(headIsTerminator(expected))
                    return false;
                std::uint32_t const index = headIndex(expected);
                if(index >= static_cast<std::uint32_t>(T_SlotCapacity))
                    return false;
                auto* slot = &slotAt(index);
                // Acquire: order the slot reads below against the release that
                // preceded the head advance that made this slot visible.
                pool_detail::claimAcquire(acc);
                // The successor is encoded in the slot's free-list field; it is
                // stable while this slot is at the head (only a push of this
                // slot can change it, and a push requires a release, which
                // requires the slot to be held, which requires it off the list).
                std::uint32_t const next = detail::dev::loadAcquire(&slot->record.handle.index);
                // Build the desired head: the successor's index with the
                // terminator tag. The tag is what closes the ABA window: a
                // stale CAS against a head that has cycled through the same
                // index fails because the tag differs.
                std::uint32_t const desiredEpoch = headEpoch(expected) + 1u;
                std::uint64_t const desired = encodeHead(next, desiredEpoch);
                std::uint64_t const result = pool_detail::atomicCasDevice(acc, &hdr->freeListHead, expected, desired);
                if(result == expected)
                {
                    // A free slot must be inactive. Observing an active slot
                    // here means the same slot was handed to two owners: the
                    // exact double-ownership failure this pool must never
                    // produce. Record it instead of silently overwriting.
                    if(slot->meta.active != 0u)
                        (void) pool_detail::atomicAddDevice(acc, &hdr->violations, std::uint64_t{1u});
                    // CAS succeeded: commit the slot's meta and record.
                    // The meta stores are safe now: this warp owns the slot
                    // (it just advanced the head), and no other warp can pop
                    // this slot until it is released back to the free list.
                    slot->meta.generation += 1u;
                    slot->meta.active = 1u;
                    slot->record.handle.index = index;
                    slot->record.elementCount = 0u;
                    slot->record.schema = ElementSchema{};
                    // Publish the record with a *release store on the generation
                    // word*, and store the generation last. The generation is
                    // exactly the word a consumer acquire-loads to decide
                    // ownership (`observedGeneration()`), so it must be the word
                    // that publishes every store above it. A release fence
                    // placed *after* the record stores (the previous form) has
                    // no following store to announce them and does not order
                    // them for `ld.acquire`, so a consumer in another block may
                    // legitimately still read the initial free-list generation 0
                    // and drop a job whose slot this CAS already handed out.
                    detail::dev::storeRelease(&slot->record.handle.generation, slot->meta.generation);
                    (void) pool_detail::atomicAddDevice(acc, &hdr->headVersion, std::uint32_t{1u});
                    // Decrement: freeCount is a uint64_t; the value -1 wraps to
                    // 0xFFFF...FFFF which is the correct 2's-complement decrement.
                    (void) pool_detail::atomicAddDevice(acc, &hdr->freeCount, ~std::uint64_t{0u});
                    out.index = index;
                    out.generation = slot->meta.generation;
                    return true;
                }
                // CAS failed: another warp moved the head. Retry with the new
                // head. Count the failure for diagnostics.
                (void) pool_detail::atomicAddDevice(acc, &hdr->casFailures, std::uint64_t{1u});
            }
            return false;
        }

        /** Push one slot back onto the free list via a CAS on the tagged head.
         *
         * The handle must be valid (generation matches, slot active). On
         * success the slot's `meta.active` is cleared, its generation is
         * incremented (so a stale handle no longer matches), and the slot is
         * linked at the head. A violation leaves the free list untouched.
         */
        ALPAKA_FN_ACC constexpr bool pushSlot(Acc const& acc, PacketHandle const& handle) const noexcept
        {
            auto* hdr = &header();
            if(handle.index >= static_cast<std::uint32_t>(T_SlotCapacity))
                return false;
            auto* slot = &slotAt(handle.index);
            if(slot->meta.generation != handle.generation || slot->meta.active == 0u)
            {
                (void) pool_detail::atomicAddDevice(acc, &hdr->violations, std::uint64_t{1u});
                return false;
            }
            // Serialize releases of one slot. Two threads holding the same
            // handle used to be able to prepare and push the slot concurrently:
            // the loser's CAS failed, it restored `active` on a slot that the
            // winner had already published, and the retry then linked the slot
            // to itself. That free-list self-loop handed one slot to several
            // owners. The transition 1 -> 2 claims the single release right;
            // a second releaser observes 0/2 and fails instead of mutating the
            // list.
            if(pool_detail::atomicCasDevice(acc, &slot->meta.active, std::uint32_t{1u}, std::uint32_t{2u})
               != std::uint32_t{1u})
            {
                (void) pool_detail::atomicAddDevice(acc, &hdr->violations, std::uint64_t{1u});
                return false;
            }
            if(slot->meta.generation != handle.generation)
            {
                slot->meta.active = 1u;
                (void) pool_detail::atomicAddDevice(acc, &hdr->violations, std::uint64_t{1u});
                return false;
            }
            for(std::uint32_t attempt = 0u; attempt < maxPopAttempts; ++attempt)
            {
                std::uint64_t const expected = *reinterpret_cast<std::uint64_t volatile*>(&hdr->freeListHead);
                // All slot state must be final *before* the head CAS makes the
                // slot reachable. A concurrent popper can pop this slot and
                // write its own meta immediately after the CAS succeeds; a
                // post-CAS meta store by this pusher would then clear `active`
                // and bump the generation of a slot that is already owned by
                // that popper. That exact race allowed a live packet to look
                // free and to be handed to a second owner, so the push commits
                // `active = 0` and the generation bump first and restores them
                // only when the CAS fails (the slot is private until then).
                slot->record.handle.index = headIsTerminator(expected) ? invalidSlotIndex : headIndex(expected);
                slot->record.handle.generation = 0u;
                slot->record.elementCount = 0u;
                slot->record.schema = ElementSchema{};
                slot->meta.active = 0u;
                slot->meta.generation += 1u;
                // Publish the link, the record, and the meta before the head
                // CAS. The popper acquires after observing the new head.
                pool_detail::publishRelease(acc);
                std::uint32_t const desiredEpoch = headEpoch(expected) + 1u;
                std::uint64_t const desired = encodeHead(handle.index, desiredEpoch);
                std::uint64_t const result = pool_detail::atomicCasDevice(acc, &hdr->freeListHead, expected, desired);
                if(result == expected)
                {
                    (void) pool_detail::atomicAddDevice(acc, &hdr->headVersion, std::uint32_t{1u});
                    (void) pool_detail::atomicAddDevice(acc, &hdr->freeCount, std::uint64_t{1u});
                    return true;
                }
                // CAS failed: another warp moved the head and never published
                // this slot. We still hold the release claim, so restore the
                // private state (2 = being released) and retry.
                slot->meta.active = 2u;
                slot->meta.generation -= 1u;
                (void) pool_detail::atomicAddDevice(acc, &hdr->casFailures, std::uint64_t{1u});
            }
            return false;
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr PoolHeader& header() const noexcept
        {
            return *reinterpret_cast<PoolHeader*>(m_storage);
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr PoolSlot& slotAt(std::size_t index) const noexcept
        {
            return *reinterpret_cast<PoolSlot*>(m_storage + headerBytes + index * slotBytes);
        }

        char* m_storage;
    };
} // namespace graph
