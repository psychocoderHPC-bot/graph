// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/detail/warp_handoff.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Bounded multi-producer, multi-consumer ready ring in device memory.
 *
 * The ring is the scheduler's work queue: a worker warp publishes a
 * `JobDescriptor` it wants to run next, and a worker warp claims the next
 * descriptor to execute. All storage is one caller-provided device buffer
 * (an alpaka buffer allocated before the persistent launch); the ring performs
 * no allocation of its own and holds nothing across a warp collective, a wait,
 * or a callback.
 *
 * # Sequence scheme (corrected)
 *
 * Two unsigned modular counters drive the ring:
 *
 * - `pushCounter` advances by one per accepted push.
 * - `claimCounter` advances by one per accepted claim.
 *
 * A slot is writable iff its `sequence` field equals
 * `claimCounter & slotMask`. This is the **stale-consumer protection**: a
 * consumer that lags a full lap sees the slot re-stamped by the producer, and
 * the `sequence == claimCounter & slotMask` check fails. The invariant
 * `0 <= claimCounter - pushCounter < capacity` (checked by **delta**, never by
 * absolute comparison, so it is wrap-safe) is the capacity check; it is read
 * under the lock, so a push is accepted only when a slot is actually free and a
 * claim only when an entry is actually in flight.
 *
 * Because every push and every claim advances its counter by exactly one, and
 * the oldest in-flight entry always carries `sequence == claimCounter`, claim
 * order equals push order and each stamp is retired by exactly one consumer:
 * the exactly-once claim property.
 *
 * # Copy out before reclaim
 *
 * On a claim the leader reads the whole entry into a local copy *before* the
 * slot is marked reclaimable (the `claimCounter` advance). The result of the
 * claim is then written into one block-shared `ClaimResult` slot and the warp
 * runs a block barrier; every lane reads the result from shared memory. The
 * slot is writable again only once `claimCounter` has advanced, and the next
 * producer writes the new entry under the lock, so the consumer's local copy is
 * independent of the slot being reused underneath it.
 *
 * The claim result is **not** distributed through warp shuffles. The pinned
 * alpaka implements a shuffle with `__activemask()`, which does not reconverge
 * the warp, so a shuffle after a leader-only region can silently deliver the
 * caller's own value. See `graph/detail/warp_handoff.hpp` for the contract and
 * the measurement.
 *
 * # The lock
 *
 * One naturally aligned `uint32_t` word: `0` free, `1` held. The attempt is a
 * single device-scope compare-exchange; a failed attempt returns `contended`
 * and never blocks. Only warp leaders attempt the lock. Nobody holds the lock
 * across a barrier, a wait, or a callback: `enqueue` and `dequeue` take the
 * lock, do their short critical section, and unlock, then run the barrier
 * *outside* the lock.
 *
 * # Backoff
 *
 * The methods are nonblocking. The retry schedule is the caller's policy. The
 * documented bounded backoff is: after a `contended`, `full`, or `empty`
 * result, the caller spins at most a bounded number of consecutive attempts and
 * then yields the worker. A finite attempt budget keeps a livelock visible as a
 * deadline expiry in tests instead of a hang.
 *
 * # Publication chain
 *
 * Producer: the lanes finish writing their contribution to the caller-owned
 * entry, the caller runs `WarpBarrier::barrier`, then the leader's `enqueue`
 * publishes the slot with a device-scope release before it retakes the lock.
 * The returned status is written to a block-shared word and read by every lane
 * after the ring's own block barrier, so control flow downstream of an
 * `enqueue` is genuinely warp-uniform.
 *
 * Consumer: the leader's `dequeue` acquires under the lock, copies the entry
 * into the block-shared `ClaimResult`, advances `claimCounter`, and unlocks;
 * the ring then runs the block barrier and every lane reads the claim result
 * from shared memory.
 *
 *
 * # Layout
 *
 * Header words live on separate 64-byte cache lines (`lock`, `pushCounter`,
 * `claimCounter`, `violations`) so the contended lock does not false-share
 * with the counters. The slots follow, each aligned to its natural
 * alignment.
 */
namespace graph
{
    /** Outcome of a nonblocking `enqueue()` or `dequeue()` attempt. */
    enum class RingStatus : std::uint32_t
    {
        /** The entry was accepted (`enqueue`) or claimed (`dequeue`). */
        Accepted = 0u,
        /** The ring is full: every slot is in flight. */
        Full,
        /** The ring is empty: no in-flight entry. */
        Empty,
        /** Another warp holds the lock; retry with backoff. */
        Contended
    };

    /** One ready entry: the full job descriptor, copied bitwise into the ring. */
    using ReadyEntry = JobDescriptor;

    /** The complete outcome of one claim, distributed through block-shared
     *  memory.
     *
     * The leader fills this once per `dequeue` and never mutates it afterwards.
     * Every lane reads the same bytes after the ring's block barrier, so the
     * descriptor, the packet handle, and the status are identical on all lanes
     * by construction instead of by a shuffle. `valid` is the single predicate
     * every downstream branch may use; `entry` is the immutable descriptor the
     * claim read out of the ring slot.
     */
    struct ClaimResult
    {
        /** The claimed descriptor (undefined unless `valid` is one). */
        ReadyEntry entry{};
        /** Ring slot the claim read (`claimCounter & slotMask`). */
        std::uint64_t slotIndex{0u};
        /** Monotonic claim stamp, the value the slot was validated against. */
        std::uint64_t generation{0u};
        /** Operation id of the claimed descriptor. */
        OperationId operationId{};
        /** Packet the claimed descriptor names. */
        PacketHandle packet{};
        /** Outcome of the claim attempt. */
        RingStatus status{RingStatus::Empty};
        /** One when `entry` is a real claim, zero otherwise. */
        std::uint32_t valid{0u};
    };

    static_assert(std::is_trivially_copyable_v<ClaimResult>);
    static_assert(std::is_standard_layout_v<ClaimResult>);

    namespace ring_detail
    {
        /** Round `value` up to a multiple of `alignment` (both powers of two). */
        constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        /** Device-scope release fence: the payload stores issued before it
         *  become visible to any warp that performs the matching acquire fence
         *  after observing the counter advance that follows it. */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr void publishRelease(T_Acc const& acc) noexcept
        {
            alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
        }

        /** Device-scope acquire fence. */
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

        /** Device-scope atomic fetch-add. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC constexpr T atomicFetchAddDevice(T_Acc const& acc, T* addr, T value) noexcept
        {
            return alpaka::onAcc::atomicAdd(acc, addr, value, alpaka::onAcc::scope::device);
        }
    } // namespace ring_detail

    /** Bounded multi-producer, multi-consumer ready ring over preallocated
     *  device storage.
     *
     * `T_Capacity` is the number of slots (compile time, power of two). All
     * public methods are device-side (`ALPAKA_FN_ACC`) and warp-uniform: every
     * active lane of the calling warp must call the same method with the same
     * arguments, and only the lane selected by `WarpGroup::isLeader()` performs
     * the locked work. `enqueue` and `dequeue` run a block barrier themselves
     * (inside the block-shared handoff) so the leader's result is visible to
     * every lane before they return; the caller's own producer barrier before
     * an `enqueue` is still required. The ring is used one warp per block, so
     * that barrier is a warp barrier.
     */
    template<typename T_Acc, std::size_t T_Capacity = 1024u>
    class ReadyRing
    {
        static_assert(T_Capacity >= 1u && (T_Capacity & (T_Capacity - 1u)) == 0u, "capacity must be a power of two");

    public:
        using Acc = std::remove_cvref_t<T_Acc>;

        static constexpr std::size_t capacity = T_Capacity;
        static constexpr std::size_t slotMask = T_Capacity - 1u;
        static constexpr std::uint64_t capacityStamp = static_cast<std::uint64_t>(T_Capacity);

        /** A ring slot: the entry payload plus a `sequence` stamp.
         *
         * `sequence` is the `pushCounter` value that wrote this entry. A slot
         * is reclaimable iff `sequence == claimCounter & slotMask`.
         */
        struct RingSlot
        {
            ReadyEntry entry{};
            std::uint64_t sequence{0u};
        };

        static_assert(std::is_trivially_copyable_v<RingSlot>);
        static_assert(std::is_standard_layout_v<RingSlot>);

        /** Ring header. Each counter owns its own 64-byte cache line. */
        struct RingHeader
        {
            std::uint32_t lock{0u};
            std::uint64_t padLock[7]{};
            std::uint64_t pushCounter{0u};
            std::uint64_t padPush[7]{};
            std::uint64_t claimCounter{0u};
            std::uint64_t padClaim[7]{};
            std::uint64_t violations{0u};
            std::uint64_t padViolations[7]{};
        };

        static_assert(std::is_trivially_copyable_v<RingHeader>);
        static_assert(std::is_standard_layout_v<RingHeader>);

        static constexpr std::size_t headerBytes = ring_detail::alignUp(sizeof(RingHeader), 64u);
        static constexpr std::size_t slotBytes = sizeof(RingSlot);
        static constexpr std::size_t storageBytes = headerBytes + T_Capacity * slotBytes;

        /** Construct a view over preallocated device storage of at least
         *  `storageBytes` bytes. The storage must stay valid and exclusively
         *  owned by this ring for its lifetime. */
        ALPAKA_FN_ACC constexpr explicit ReadyRing(void* storage) noexcept : m_storage(static_cast<char*>(storage))
        {
        }

        /** Zero-initialize the ring storage.
         *
         * Called once at startup (device side, a startup action, or a single
         * warp before the pool starts). All lanes call it warp-uniformly; only
         * the leader writes. The caller should run a `WarpBarrier::barrier`
         * after this so the leader's stores are visible to the lanes.
         */
        ALPAKA_FN_ACC constexpr void initialize(Acc const& acc, WarpGroup<Acc> const& warp) noexcept
        {
            if(warp.isLeader())
            {
                auto* hdr = &header();
                hdr->lock = 0u;
                hdr->pushCounter = 0u;
                hdr->claimCounter = 0u;
                hdr->violations = 0u;
                for(std::size_t i = 0u; i < T_Capacity; ++i)
                {
                    slotAt(i).sequence = 0u;
                    slotAt(i).entry = ReadyEntry{};
                }
            }
        }

        /** Nonblocking attempt to publish one entry.
         *
         * Warp-uniform: every active lane calls it with the same `entry`. The
         * leader performs the locked work; the status is written to a
         * block-shared word, the ring runs a block barrier, and every lane
         * reads the same status. The caller is responsible for the producer
         * barrier *before* this call (so the lanes' stores into `entry` are
         * complete) and for any consumer-side ordering the protocol needs. The
         * ring does not block.
         *
         * On `Full` the ring is at capacity; the caller backs off and retries.
         * On `Contended` another warp holds the lock; the caller backs off.
         */
        ALPAKA_FN_ACC constexpr RingStatus enqueue(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            ReadyEntry const& entry) noexcept
        {
            WarpHandoff<Acc, std::uint32_t, handoff_id::readyRingPush> handoff;
            std::uint32_t status = static_cast<std::uint32_t>(RingStatus::Contended);
            if(warp.isLeader())
                status = static_cast<std::uint32_t>(pushOnce(acc, entry));
            handoff.publish(acc, warp, status);
            handoff.sync(acc);
            return static_cast<RingStatus>(handoff.get(acc));
        }

        /** Nonblocking attempt to claim the next entry.
         *
         * Warp-uniform. The leader claims under the lock and writes the whole
         * `ClaimResult` into one block-shared slot; the ring then runs a block
         * barrier and every lane reads the claim result from that slot. On
         * `Accepted`, `out` holds the claimed entry on every lane and the
         * status is identical everywhere. The ring does not block.
         *
         * On `Empty` the ring has no in-flight entry. On `Contended` another
         * warp holds the lock.
         */
        ALPAKA_FN_ACC constexpr RingStatus dequeue(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            ReadyEntry& out) noexcept
        {
            WarpHandoff<Acc, ClaimResult, handoff_id::readyRingClaim> handoff;
            ClaimResult claim{};
            if(warp.isLeader())
                claim = claimOnce(acc);
            handoff.publish(acc, warp, claim);
            handoff.sync(acc);
            ClaimResult const result = handoff.get(acc);
            if(result.valid != 0u)
                out = result.entry;
            return result.status;
        }

        /** Number of in-flight entries (push minus claim, as a delta). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t inFlightCount() const noexcept
        {
            auto const& hdr = header();
            return hdr.pushCounter - hdr.claimCounter;
        }

        /** Stamps pushed so far (next accepted push gets this stamp). */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t nextPushSequence() const noexcept
        {
            return header().pushCounter;
        }

        /** Stamps claimed so far. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t claimCounter() const noexcept
        {
            return header().claimCounter;
        }

        /** Ownership violations recorded by the test hook. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr std::uint64_t ownershipViolations() const noexcept
        {
            return header().violations;
        }

    private:
        ALPAKA_FN_ACC constexpr bool tryLock(Acc const& acc) const noexcept
        {
            return ring_detail::atomicCasDevice(acc, &header().lock, std::uint32_t{0u}, std::uint32_t{1u})
                   == std::uint32_t{0u};
        }

        ALPAKA_FN_ACC constexpr void unlock(Acc const& acc) const noexcept
        {
            ring_detail::publishRelease(acc);
            ring_detail::atomicCasDevice(acc, &header().lock, std::uint32_t{1u}, std::uint32_t{0u});
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr RingHeader& header() const noexcept
        {
            return *reinterpret_cast<RingHeader*>(m_storage);
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr RingSlot& slotAt(std::size_t index) const noexcept
        {
            return *reinterpret_cast<RingSlot*>(m_storage + headerBytes + index * slotBytes);
        }

        /** Publish one entry under the lock. Leader only.
         *
         * Kept separate from `enqueue` so the collective wrapper has exactly
         * one leader-only region and one handoff.
         */
        ALPAKA_FN_ACC constexpr RingStatus pushOnce(Acc const& acc, ReadyEntry const& entry) const noexcept
        {
            if(!tryLock(acc))
                return RingStatus::Contended;
            auto* hdr = &header();
            auto const inFlight = hdr->pushCounter - hdr->claimCounter; // delta; wraps safely
            if(inFlight >= capacityStamp)
            {
                unlock(acc);
                return RingStatus::Full;
            }
            // Reserve and publish under the lock. The slot is the push
            // counter's position; it is free because inFlight < capacity.
            auto const stamp = hdr->pushCounter;
            auto* slot = &slotAt(stamp & slotMask);
            // Device-scope release: the payload stores the lanes made before
            // the caller's producer barrier are ordered before the lock release
            // / counter advance that the consumer will acquire against.
            slot->entry = entry;
            slot->sequence = stamp;
            ring_detail::publishRelease(acc);
            hdr->pushCounter = stamp + 1u;
            unlock(acc);
            return RingStatus::Accepted;
        }

        /** Claim the oldest entry under the lock. Leader only.
         *
         * Copies the entry into the returned `ClaimResult` *before* advancing
         * `claimCounter`, so the result never observes a slot the producer has
         * already reused.
         */
        ALPAKA_FN_ACC constexpr ClaimResult claimOnce(Acc const& acc) const noexcept
        {
            ClaimResult result{};
            result.status = RingStatus::Contended;
            if(!tryLock(acc))
                return result;
            auto* hdr = &header();
            auto const inFlight = hdr->pushCounter - hdr->claimCounter; // delta; wraps safely
            if(inFlight == 0u)
            {
                unlock(acc);
                result.status = RingStatus::Empty;
                return result;
            }
            // The oldest in-flight entry carries `sequence == claimCounter`.
            // Device-scope acquire: ordered against the producer's release so
            // the entry payload is visible.
            auto const stamp = hdr->claimCounter;
            auto* slot = &slotAt(stamp & slotMask);
            ring_detail::claimAcquire(acc);
            // Ownership-violation hook (test builds): a valid claim finds the
            // slot stamped exactly with `stamp`. Anything else means a consumer
            // lagged a full lap and the slot was reused underneath it.
            if(slot->sequence != stamp)
                (void) ring_detail::atomicFetchAddDevice(acc, &hdr->violations, std::uint64_t{1u});
            // Copy the entry out BEFORE reclaiming the slot. The slot is
            // reclaimable only after claimCounter advances.
            result.entry = slot->entry;
            result.slotIndex = stamp & slotMask;
            result.generation = stamp;
            result.operationId = result.entry.operation;
            result.packet = result.entry.packet;
            result.valid = 1u;
            result.status = RingStatus::Accepted;
            ring_detail::publishRelease(acc);
            hdr->claimCounter = stamp + 1u;
            unlock(acc);
            return result;
        }

        char* m_storage;
    };
} // namespace graph
