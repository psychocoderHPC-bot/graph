// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/concepts.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Fixed-capacity packet slot pool with explicit ownership.
 *
 * The pool owns one preallocated device buffer of `T_SlotCapacity` slots,
 * each `T_ElementCapacity` bytes of element storage wide. It hands out slot
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
 *   claimed. The ring stamp advances and the ring slot is writable again;
 *   the packet the entry named is untouched.
 * - The packet pool reuses a *packet slot* when its owner calls
 *   `release()`. The pool increments the slot's generation and the slot is
 *   acquirable again; the ring is untouched.
 *
 * A job that reads a packet must keep its handle until the last read
 * completes; releasing earlier is a use-after-release that the generation
 * check detects in test builds.
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
 * - `P` is the live-particle admission limit: each ordinary detached
 *   nonempty packet can be charged to a distinct admitted particle, so at
 *   most `P` slots are live particle data.
 * - `B` is the number of assembly streams (Bin 0, Bin 1, and the common
 *   feedback input in the required example). Each stream keeps two physical
 *   output chunks (`A`, `B`), so `2*B` slots.
 * - `W` is the number of concurrent workers and `R` is the maximum number of
 *   detached replacement or scratch packets one worker may hold at once, so
 *   `W*R` slots for temporary allocations and copied-but-uncommitted
 *   packets.
 * - `S` is the control/source allowance: slots for control packets and
 *   source request staging that do not carry admitted particles.
 *
 * `PacketPool::configure()` verifies at startup that the configured
 * `T_SlotCapacity` satisfies the formula for the requested budget and fails
 * loudly (records a violation, exposes it through `ownershipViolations()`,
 * and leaves no slot acquirable) when it does not. There is no silent
 * clamping.
 *
 * The example values used by the required particle-feedback example are
 * parameters, not magic numbers; the tests instantiate the pool with small
 * values of each.
 *
 * # Storage
 *
 * No allocation after startup. All storage comes from a
 * caller-provided device buffer (an alpaka buffer) so the pool can be
 * preallocated before the persistent launch. The host calls
 * `PacketPool::storageBytes()` to size the buffer, allocates it, and hands
 * the pointer to the startup action of the persistent kernel, which calls
 * `configure()`.
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
     * every successful `release()` that returns the slot to the free list),
     * so a handle whose generation does not match the slot's current
     * generation names a slot that has been reused. The pool records an
     * ownership violation for such a mismatch (see
     * `PacketPool::ownershipViolations()`).
     *
     * `active` is set while a slot is held and cleared on release; a release
     * on a slot that is not active is a double-release and records a
     * violation.
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
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t
    requiredSlots(std::uint32_t P, std::uint32_t B, std::uint32_t W, std::uint32_t R, std::uint32_t S = 0u) noexcept
    {
        return P + 2u * B + W * R + S;
    }

    ALPAKA_FN_HOST_ACC constexpr std::uint32_t requiredSlots(PacketPoolBudget const& b) noexcept
    {
        return requiredSlots(b.liveParticles, b.assemblyStreams, b.workers, b.replacementsPerWorker, b.controlAllowance);
    }

    namespace detail
    {
        /** Device-scope release fence: the stores issued before it become
         *  visible to any warp that performs the matching acquire fence after
         *  observing the lock release. Shared with the ready ring: the pool
         *  and the ring use the same publication protocol. */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr void publishRelease(T_Acc const& acc) noexcept
        {
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
        }

        /** Warp-level lane rendezvous (not a memory barrier): maps to the
         *  warp primitive `all(true)`, the width-one identity on the host. */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr void warpRendezvous(WarpGroup<T_Acc> const& warp) noexcept
        {
            (void)warp.all(true);
        }

        /** Device-scope atomic compare-exchange on a naturally aligned word. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC constexpr T atomicCasDevice(T_Acc const& acc, T* addr, T expected, T desired) noexcept
        {
            return alpaka::onAcc::atomicCas(acc, addr, expected, desired, alpaka::onAcc::scope::device);
        }
    } // namespace detail

    /** Fixed-capacity packet slot pool over preallocated device storage.
     *
     * `T_SlotCapacity` is the number of slots; `T_ElementCapacity` is the
     * physical element storage of each slot in bytes. Both are compile time.
     *
     * All public methods are device-side (`ALPAKA_FN_ACC`) and warp-uniform:
     * every active lane of the calling warp must call the same method with
     * the same arguments, and only the lane selected by
     * `WarpGroup::isLeader()` performs the locked work.
     *
     * The pool is a view over preallocated storage: it performs no
     * allocation and never outlives the storage it references.
     */
    template<typename T_Acc, std::size_t T_SlotCapacity = 1024u, std::size_t T_ElementCapacity = 4096u>
    class PacketPool
    {
        static_assert(T_SlotCapacity >= 1u);
        static_assert(T_ElementCapacity >= 1u);

    public:
        using Acc = std::remove_cvref_t<T_Acc>;

        static constexpr std::size_t slotCapacity = T_SlotCapacity;
        static constexpr std::size_t elementCapacity = T_ElementCapacity;

        /** One slot: the element storage plus the per-slot bookkeeping.
         *
         * The pool is element-type-agnostic: the storage is raw bytes and
         * the typed views are reconstructed through the vocabulary
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

        /** Pool header. Each word owns its own 64-byte cache line so the
         *  contended `lock` word does not false-share with the counters. */
        struct PoolHeader
        {
            std::uint32_t lock{0u};
            std::uint64_t padLock[7]{};
            SlotIndex freeListHead{invalidSlotIndex};
            std::uint64_t padHead[7]{};
            std::uint64_t freeCount{0u};
            std::uint64_t padFree[7]{};
            std::uint64_t violations{0u};
            std::uint64_t padViolations[7]{};
        };

        static_assert(std::is_trivially_copyable_v<PoolHeader>);
        static_assert(std::is_standard_layout_v<PoolHeader>);
        static_assert(sizeof(PoolHeader) == 256u, "header layout is cache-line separated");

        /** Bytes per slot and total storage size in bytes. */
        static constexpr std::size_t slotBytes = sizeof(PoolSlot);
        static constexpr std::size_t storageBytes = sizeof(PoolHeader) + T_SlotCapacity * slotBytes;

        /** Construct a view over preallocated device storage of at least
         *  `storageBytes` bytes.
         *
         * The storage must stay valid and exclusively owned by this pool for
         * its lifetime. Call `configure()` once (device side, startup
         * action) before the first `acquire()`.
         */
        ALPAKA_FN_ACC constexpr explicit PacketPool(void* storage) noexcept : m_storage(static_cast<char*>(storage))
        {
        }

        /** Initialize the pool for the given budget.
         *
         * The configured `T_SlotCapacity` must satisfy
         * `requiredSlots(budget)`; when it does not, the call fails loudly:
         * it records a violation (visible through `ownershipViolations()`),
         * leaves no slot acquirable, and returns false. There is no silent
         * clamping.
         *
         * All lanes must call it warp-uniformly; only the leader writes.
         * Returns `true` on success.
         */
        ALPAKA_FN_ACC constexpr bool configure(Acc const& acc, WarpGroup<Acc> const& warp, PacketPoolBudget const& budget)
            noexcept
        {
            bool ok = true;
            if(warp.isLeader())
            {
                auto* hdr = &header();
                // Initialize slot metas in both branches so a misconfigured
                // pool is still in a defined (if unusable) state.
                for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                {
                    slotAt(i).meta.generation = 0u;
                    slotAt(i).meta.active = 0u;
                }
                if(requiredSlots(budget) > T_SlotCapacity)
                {
                    hdr->lock = 0u;
                    hdr->freeListHead = invalidSlotIndex;
                    hdr->freeCount = 0u;
                    hdr->violations = 1u;
                    detail::publishRelease(acc);
                    ok = false;
                }
                else
                {
                    hdr->lock = 0u;
                    hdr->violations = 0u;
                    hdr->freeListHead = 0u;
                    hdr->freeCount = T_SlotCapacity;
                    for(std::size_t i = 0u; i < T_SlotCapacity; ++i)
                    {
                        auto* slot = &slotAt(i);
                        slot->meta.generation = 0u;
                        slot->meta.active = 0u;
                        // Free-list encoding: next = i+1, terminated.
                        slot->record.handle.index
                            = (i + 1u < T_SlotCapacity) ? static_cast<SlotIndex>(i + 1u) : invalidSlotIndex;
                        slot->record.handle.generation = 0u;
                        slot->record.elementCount = 0u;
                        slot->record.capacity = static_cast<std::uint32_t>(T_ElementCapacity);
                        slot->record.schema = ElementSchema{};
                    }
                    detail::publishRelease(acc);
                }
            }
            detail::warpRendezvous(warp);
            auto const value = static_cast<std::uint32_t>(ok ? 1u : 0u);
            return warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
        }

        /** Acquire one slot.
         *
         * Warp-uniform. On success, `out` holds the handle (index plus the
         * slot's new generation) and the slot's `record` is reset so the
         * caller fills in its schema and element count. The leader performs
         * the locked work; the other lanes rendezvous before and after.
         *
         * Returns `true` on success, `false` when no slot is free or the
         * pool is locked.
         */
        ALPAKA_FN_ACC constexpr bool acquire(Acc const& acc, WarpGroup<Acc> const& warp, PacketHandle& out) noexcept
        {
            bool ok = false;
            if(warp.isLeader())
            {
                if(!tryLock(acc))
                {
                    ok = false;
                }
                else
                {
                    auto* hdr = &header();
                    if(hdr->freeCount == 0u)
                    {
                        unlock(acc);
                        ok = false;
                    }
                    else
                    {
                        auto const index = hdr->freeListHead;
                        auto* slot = &slotAt(index);
                        // Pop: the next free slot is encoded in the slot's
                        // free-list fields.
                        auto const nextFree = slot->record.handle.index;
                        slot->meta.generation += 1u;
                        slot->meta.active = 1u;
                        slot->record.handle.index = index;
                        slot->record.handle.generation = slot->meta.generation;
                        slot->record.elementCount = 0u;
                        slot->record.schema = ElementSchema{};
                        // Publish the record before publishing the free-list
                        // advance, so a concurrent acquirer that observes the
                        // new head also observes the popped slot's new state.
                        detail::publishRelease(acc);
                        hdr->freeListHead = nextFree;
                        hdr->freeCount -= 1u;
                        unlock(acc);
                        ok = true;
                        out = slot->record.handle;
                    }
                }
            }
            detail::warpRendezvous(warp);
            auto const value = static_cast<std::uint32_t>(ok ? 1u : 0u);
            ok = warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
            if(ok)
            {
                broadcastHandle(warp, out);
            }
            return ok;
        }

        /** Release one slot.
         *
         * Warp-uniform. The handle must name a slot the caller owns (its
         * generation must match the slot's current generation and the slot
         * must be active). On success the slot's `meta.active` is cleared,
         * its generation is incremented (so a stale handle no longer
         * matches), and the slot is pushed back onto the free list.
         *
         * A release with a mismatched generation (use-after-release) or a
         * release of an inactive slot (double-release) records an ownership
         * violation in `violations`, leaves the free list untouched, and
         * returns false.
         */
        ALPAKA_FN_ACC constexpr bool release(Acc const& acc, WarpGroup<Acc> const& warp, PacketHandle const& handle)
            noexcept
        {
            bool ok = false;
            if(warp.isLeader())
            {
                if(!tryLock(acc))
                {
                    ok = false;
                }
                else
                {
                    auto* hdr = &header();
                    auto* slot = &slotAt(handle.index);
                    auto const violation
                        = slot->meta.generation != handle.generation || slot->meta.active == 0u;
                    if(violation)
                    {
                        hdr->violations += 1u;
                        unlock(acc);
                        ok = false;
                    }
                    else
                    {
                        slot->meta.active = 0u;
                        slot->meta.generation += 1u;
                        // Push onto the free list: the slot's free-list field
                        // points at the current head, then the head advances.
                        slot->record.handle.index = hdr->freeListHead;
                        slot->record.handle.generation = 0u;
                        slot->record.elementCount = 0u;
                        slot->record.schema = ElementSchema{};
                        // Publish the slot's new free-list state before the
                        // head advances.
                        detail::publishRelease(acc);
                        hdr->freeListHead = handle.index;
                        hdr->freeCount += 1u;
                        unlock(acc);
                        ok = true;
                    }
                }
            }
            detail::warpRendezvous(warp);
            auto const value = static_cast<std::uint32_t>(ok ? 1u : 0u);
            return warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
        }

        /** True when `handle` names a slot the pool currently considers
         *  owned by the caller (generation matches, slot active).
         *
         * This is the generation-mismatch check that makes a
         * use-after-release detectable in test builds.
         */
        ALPAKA_FN_ACC constexpr bool isValid(Acc const& acc, WarpGroup<Acc> const& warp, PacketHandle const& handle)
            const noexcept
        {
            bool valid = false;
            if(warp.isLeader())
            {
                auto const& slot = slotAt(handle.index);
                valid = slot.meta.generation == handle.generation && slot.meta.active == 1u;
            }
            detail::warpRendezvous(warp);
            auto const value = static_cast<std::uint32_t>(valid ? 1u : 0u);
            return warp.template broadcast<std::uint32_t>(value, 0u) != 0u;
        }

        /** Raw element pointer for a slot.
         *
         * The caller must hold the slot (the handle must be valid). The
         * pointer stays valid until the slot is released and reused; the
         * generation check is the caller's responsibility (use `isValid()` in
         * test builds).
         */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint8_t* slotData(PacketHandle const& handle) const noexcept
        {
            return slotAt(handle.index).data;
        }

        /** The slot's current record (the caller must hold the slot). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr PacketRecord const& slotRecord(PacketHandle const& handle) const noexcept
        {
            return slotAt(handle.index).record;
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
        ALPAKA_FN_ACC constexpr std::uint64_t freeCount(Acc const& acc) const noexcept
        {
            return header().freeCount;
        }

        /** Number of slots currently held. */
        ALPAKA_FN_ACC constexpr std::uint64_t heldCount(Acc const& acc) const noexcept
        {
            return T_SlotCapacity - header().freeCount;
        }

        /** Ownership violations recorded by the test hook. */
        ALPAKA_FN_ACC constexpr std::uint64_t ownershipViolations(Acc const& acc) const noexcept
        {
            return header().violations;
        }

    private:
        [[nodiscard]] ALPAKA_FN_ACC constexpr PoolHeader& header() const noexcept
        {
            return *reinterpret_cast<PoolHeader*>(m_storage);
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr PoolSlot& slotAt(std::size_t index) const noexcept
        {
            auto const slotsBegin = alignUp(sizeof(PoolHeader), 64u);
            return *reinterpret_cast<PoolSlot*>(m_storage + slotsBegin + index * slotBytes);
        }

        ALPAKA_FN_ACC constexpr bool tryLock(Acc const& acc) const noexcept
        {
            return detail::atomicCasDevice(acc, &header().lock, std::uint32_t{0u}, std::uint32_t{1u})
                   == std::uint32_t{0u};
        }

        ALPAKA_FN_ACC constexpr void unlock(Acc const& acc) const noexcept
        {
            detail::publishRelease(acc);
            detail::atomicCasDevice(acc, &header().lock, std::uint32_t{1u}, std::uint32_t{0u});
        }

        /** Broadcast the handle from the leader to all lanes. */
        ALPAKA_FN_ACC constexpr void broadcastHandle(WarpGroup<Acc> const& warp, PacketHandle& handle) const noexcept
        {
            if constexpr(Acc::getWarpSize() == 1u)
            {
                (void)warp;
                (void)handle;
            }
            else
            {
                handle.index = warp.template broadcast<std::uint32_t>(handle.index, 0u);
                handle.generation = warp.template broadcast<std::uint32_t>(handle.generation, 0u);
            }
        }

        static constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        char* m_storage;
    };
} // namespace graph
