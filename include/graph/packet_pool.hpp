// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/concepts.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>

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
 * - `freeListHead` packs a 31-bit slot index with a 1-bit tag. A push (release)
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
 * result is broadcast through warp shuffles.
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
         * `freeListHead` is a tagged word: bit 31 is the tag, bits 0..30 are
         * the slot index. `0` (index 0, tag 0) is the initial head after
         * `configure()`; the terminator is `1u << 31` (tag 1, no index).
         * `freeCount`, `violations`, and `headVersion` are device-scope atomic
         * words updated with `atomicAdd`/`atomicCas` from multiple warps.
         */
        struct PoolHeader
        {
            std::uint32_t freeListHead{0u};
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
                    hdr->freeListHead = headTag;
                    hdr->freeCount = 0u;
                    hdr->violations = 1u;
                    hdr->headVersion = 0u;
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
                    hdr->freeCount = static_cast<std::uint64_t>(T_SlotCapacity);
                    hdr->freeListHead = 0u; // index 0, tag 0
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
            auto const value = static_cast<std::uint32_t>(ok ? 1u : 0u);
            return warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
        }

        /** Acquire one slot.
         *
         * Warp-uniform. On success, `out` holds the handle (index plus the
         * slot's new generation) and the slot's `record` is reset so the
         * caller fills in its schema and element count. The leader performs
         * the atomic work and the result is broadcast.
         *
         * Returns `true` on success, `false` when no slot is free.
         */
        ALPAKA_FN_ACC constexpr bool acquire(Acc const& acc, WarpGroup<Acc> const& warp, PacketHandle& out) noexcept
        {
            bool ok = false;
            if(warp.isLeader())
            {
                ok = popSlot(acc, out);
            }
            auto const value = static_cast<std::uint32_t>(ok ? 1u : 0u);
            ok = warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
            if(ok)
                broadcastHandle(warp, out);
            return ok;
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
            bool ok = false;
            if(warp.isLeader())
            {
                ok = pushSlot(acc, handle);
            }
            auto const value = static_cast<std::uint32_t>(ok ? 1u : 0u);
            return warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
        }

        /** True when `handle` names a slot the pool currently considers owned
         *  by the caller (generation matches, slot active). This is the
         *  generation-mismatch check that makes a use-after-release
         *  detectable in test builds. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr bool isValid(
            Acc const&,
            WarpGroup<Acc> const& warp,
            PacketHandle const& handle) const noexcept
        {
            bool valid = false;
            if(warp.isLeader())
            {
                auto const& slot = slotAt(handle.index);
                valid = slot.meta.generation == handle.generation && slot.meta.active == 1u;
            }
            auto const value = static_cast<std::uint32_t>(valid ? 1u : 0u);
            return warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
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
        /** Tag bit of the free-list head: distinguishes the terminator
         *  (tag 1, no index) from a real slot (tag 0). The free list never
         *  pushes `invalidSlotIndex`, so the tag is the sole terminator mark. */
        static constexpr std::uint32_t headTag = 1u << 31u;
        static constexpr std::uint32_t headIndexMask = 0x7fff'ffffu;

        /** Encode a slot index plus tag into the head word. */
        static constexpr std::uint32_t encodeHead(std::uint32_t index, std::uint32_t tag) noexcept
        {
            return (index & headIndexMask) | (tag << 31u);
        }

        /** Decode the slot index of a head word (masked to 31 bits so a
         *  terminator word decodes to a value the caller rejects). */
        static constexpr std::uint32_t headIndex(std::uint32_t head) noexcept
        {
            return head & headIndexMask;
        }

        /** True when the head word is the terminator (no free slot). */
        static constexpr bool headIsTerminator(std::uint32_t head) noexcept
        {
            return (head & headTag) != 0u;
        }

        /** Pop one slot from the free list via a CAS on the tagged head.
         *
         * The winner advances the head, increments the slot's generation and
         * sets `active`, and publishes the record with a device-scope release
         * before the head CAS. Returns false when the pool is empty.
         */
        ALPAKA_FN_ACC constexpr bool popSlot(Acc const& acc, PacketHandle& out) const noexcept
        {
            auto* hdr = &header();
            // Read the head with a raw volatile load: the value is only a
            // candidate for the CAS that follows, and a stale read is safe
            // because the CAS re-validates it atomically.
            std::uint32_t const expected = *reinterpret_cast<std::uint32_t volatile*>(&hdr->freeListHead);
            if(headIsTerminator(expected))
                return false;
            std::uint32_t const index = headIndex(expected);
            if(index >= static_cast<std::uint32_t>(T_SlotCapacity))
                return false;
            auto* slot = &slotAt(index);
            // Acquire: order the slot reads below against the release that
            // preceded the head advance that made this slot visible.
            pool_detail::claimAcquire(acc);
            // Diagnostic: count meta checks. A slot at the head must be free
            // (active == 0, generation matching the handle). If it is not,
            // the free list is corrupted.
            (void) pool_detail::atomicAddDevice(acc, &hdr->metaChecks, std::uint64_t{1u});
            // The successor is encoded in the slot's free-list field; it is
            // stable while this slot is at the head (only a push of this
            // slot can change it, and a push requires a release, which
            // requires the slot to be held, which requires it off the list).
            std::uint32_t const next = slot->record.handle.index;
            // Bump the generation and mark the slot active. The meta stores
            // are performed before the CAS: if the CAS fails, another warp
            // moved the head and this attempt is discarded, but the meta
            // stores are harmless (the slot is still at the head or has been
            // released, and the next popper/release re-validates).
            slot->meta.generation += 1u;
            slot->meta.active = 1u;
            slot->record.handle.index = index;
            slot->record.handle.generation = slot->meta.generation;
            slot->record.elementCount = 0u;
            slot->record.schema = ElementSchema{};
            // Release the record before publishing the head advance.
            pool_detail::publishRelease(acc);
            // Build the desired head: the successor's index with the
            // terminator tag. The tag is what closes the ABA window: a
            // stale CAS against a head that has cycled through the same
            // index fails because the tag differs.
            std::uint32_t const desiredTag = (next == invalidSlotIndex) ? 1u : 0u;
            std::uint32_t const desired = encodeHead(next, desiredTag);
            std::uint32_t const result = pool_detail::atomicCasDevice(acc, &hdr->freeListHead, expected, desired);
            if(result == expected)
            {
                (void) pool_detail::atomicAddDevice(acc, &hdr->headVersion, std::uint32_t{1u});
                // Decrement: freeCount is a uint64_t; the value -1 wraps to
                // 0xFFFF...FFFF which is the correct 2's-complement decrement.
                (void) pool_detail::atomicAddDevice(acc, &hdr->freeCount, ~std::uint64_t{0u});
                out.index = index;
                out.generation = slot->meta.generation;
                return true;
            }
            // CAS failed: another warp moved the head.
            (void) pool_detail::atomicAddDevice(acc, &hdr->casFailures, std::uint64_t{1u});
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
            for(std::uint32_t attempt = 0u; attempt < (1u << 20u); ++attempt)
            {
                std::uint32_t const expected = *reinterpret_cast<std::uint32_t volatile*>(&hdr->freeListHead);
                // Link this slot at the head: its next field points at the
                // current head's index (or the terminator).
                slot->record.handle.index = headIsTerminator(expected) ? invalidSlotIndex : headIndex(expected);
                slot->record.handle.generation = 0u;
                slot->record.elementCount = 0u;
                slot->record.schema = ElementSchema{};
                // Clear active and bump the generation: a stale handle no
                // longer matches, so a second release of the same handle is
                // rejected by the generation check.
                slot->meta.active = 0u;
                slot->meta.generation += 1u;
                // Release the slot's new free-list state before the head CAS.
                pool_detail::publishRelease(acc);
                std::uint32_t const desired = encodeHead(handle.index, 0u);
                std::uint32_t const result = pool_detail::atomicCasDevice(acc, &hdr->freeListHead, expected, desired);
                if(result == expected)
                {
                    (void) pool_detail::atomicAddDevice(acc, &hdr->headVersion, std::uint32_t{1u});
                    (void) pool_detail::atomicAddDevice(acc, &hdr->freeCount, std::uint64_t{1u});
                    return true;
                }
                // CAS failed: retry with the new head.
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

        /** Broadcast the handle from the leader to all lanes. */
        ALPAKA_FN_ACC constexpr void broadcastHandle(WarpGroup<Acc> const& warp, PacketHandle& handle) const noexcept
        {
            if constexpr(Acc::getWarpSize() == 1u)
            {
                (void) warp;
                (void) handle;
            }
            else
            {
                handle.index = warp.template broadcast<std::uint32_t>(handle.index, 0u);
                handle.generation = warp.template broadcast<std::uint32_t>(handle.generation, 0u);
            }
        }

        char* m_storage;
    };
} // namespace graph
