// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>

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
 * slot is marked reclaimable (the `claimCounter` advance). The entry is then
 * broadcast field by field through warp shuffles. The slot is writable again
 * only once `claimCounter` has advanced, and the next producer writes the new
 * entry under the lock, so the consumer's local copy is independent of the
 * slot being reused underneath it.
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
 *
 * Consumer: the leader's `dequeue` acquires under the lock, copies the entry
 * out, advances `claimCounter`, and unlocks. The caller then runs
 * `WarpBarrier::barrier` so the leader's release is ordered before any lane
 * reads the broadcast entry.
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
     * the locked work. The methods do **not** run a barrier themselves; the
     * caller runs `WarpBarrier::barrier` around them so the leader's
     * publication/acquisition is ordered against the lanes.
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
         * leader performs the locked work and the result is broadcast. The
         * caller is responsible for the producer barrier *before* this call
         * (so the lanes' stores into `entry` are complete) and for any
         * consumer-side ordering the protocol needs. The ring does not block.
         *
         * On `Full` the ring is at capacity; the caller backs off and retries.
         * On `Contended` another warp holds the lock; the caller backs off.
         */
        ALPAKA_FN_ACC constexpr RingStatus enqueue(Acc const& acc, WarpGroup<Acc> const& warp, ReadyEntry const& entry)
            noexcept
        {
            RingStatus status = RingStatus::Contended;
            if(warp.isLeader())
            {
                if(!tryLock(acc))
                {
                    status = RingStatus::Contended;
                }
                else
                {
                    auto* hdr = &header();
                    auto const inFlight = hdr->pushCounter - hdr->claimCounter; // delta; wraps safely
                    if(inFlight >= capacityStamp)
                    {
                        unlock(acc);
                        status = RingStatus::Full;
                    }
                    else
                    {
                        // Reserve and publish under the lock. The slot is the
                        // push counter's position; it is free because inFlight
                        // < capacity.
                        auto const stamp = hdr->pushCounter;
                        auto* slot = &slotAt(stamp & slotMask);
                        // Device-scope release: the payload stores the lanes
                        // made before the caller's producer barrier are ordered
                        // before the lock release / counter advance that the
                        // consumer will acquire against.
                        slot->entry = entry;
                        slot->sequence = stamp;
                        ring_detail::publishRelease(acc);
                        hdr->pushCounter = stamp + 1u;
                        unlock(acc);
                        status = RingStatus::Accepted;
                    }
                }
            }
            return broadcastStatus(acc, warp, status);
        }

        /** Nonblocking attempt to claim the next entry.
         *
         * Warp-uniform. On `Accepted`, `out` holds the claimed entry (copied
         * out of the slot before the slot is reclaimable) and the entry is
         * broadcast field by field so every lane holds an identical copy. The
         * caller runs `WarpBarrier::barrier` *after* this call before any lane
         * reads the payload, ordering the leader's acquisition against the
         * lanes. The ring does not block.
         *
         * On `Empty` the ring has no in-flight entry. On `Contended` another
         * warp holds the lock.
         */
        ALPAKA_FN_ACC constexpr RingStatus dequeue(Acc const& acc, WarpGroup<Acc> const& warp, ReadyEntry& out) noexcept
        {
            RingStatus status = RingStatus::Contended;
            if(warp.isLeader())
            {
                if(!tryLock(acc))
                {
                    status = RingStatus::Contended;
                }
                else
                {
                    auto* hdr = &header();
                    auto const inFlight = hdr->pushCounter - hdr->claimCounter; // delta; wraps safely
                    if(inFlight == 0u)
                    {
                        unlock(acc);
                        status = RingStatus::Empty;
                    }
                    else
                    {
                        // The oldest in-flight entry carries `sequence ==
                        // claimCounter`. Device-scope acquire: ordered against
                        // the producer's release so the entry payload is
                        // visible.
                        auto const stamp = hdr->claimCounter;
                        auto* slot = &slotAt(stamp & slotMask);
                        ring_detail::claimAcquire(acc);

                        // Ownership-violation hook (test builds): a valid
                        // claim finds the slot stamped exactly with `stamp`.
                        // Anything else means a consumer lagged a full lap and
                        // the slot was reused underneath it.
                        if(slot->sequence != stamp)
                        {
                            ring_detail::atomicFetchAddDevice(acc, &hdr->violations, std::uint64_t{1u});
                        }

                        // Copy the entry out BEFORE reclaiming the slot. The
                        // slot is reclaimable only after claimCounter advances.
                        out = slot->entry;
                        ring_detail::publishRelease(acc);
                        hdr->claimCounter = stamp + 1u;
                        unlock(acc);
                        status = RingStatus::Accepted;
                    }
                }
            }
            status = broadcastStatus(acc, warp, status);
            if(status == RingStatus::Accepted)
                broadcastEntry(warp, out);
            return status;
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

        ALPAKA_FN_ACC constexpr RingStatus broadcastStatus(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            RingStatus status) const noexcept
        {
            auto const value = static_cast<std::uint32_t>(status);
            auto const b = warp.template broadcast<std::uint32_t>(value, 0u);
            (void)acc;
            return static_cast<RingStatus>(b);
        }

        /** Broadcast every field of the entry from the leader to all lanes.
         *
         * The entry is a trivially copyable POD; it is exchanged field by field
         * through warp broadcasts (shuffle on the device, identity on the
         * width-one host fallback) so every lane ends with an identical copy
         * without the leader writing to lane-private memory.
         */
        ALPAKA_FN_ACC constexpr void broadcastEntry(WarpGroup<Acc> const& warp, ReadyEntry& entry) const noexcept
        {
            if constexpr(Acc::getWarpSize() == 1u)
            {
                (void)warp;
                (void)entry;
            }
            else
            {
                entry.operation.value = warp.template broadcast<std::uint32_t>(entry.operation.value, 0u);
                entry.kind
                    = static_cast<JobKind>(
                        warp.template broadcast<std::uint32_t>(static_cast<std::uint32_t>(entry.kind), 0u));
                entry.packet.index = warp.template broadcast<std::uint32_t>(entry.packet.index, 0u);
                entry.packet.generation = warp.template broadcast<std::uint32_t>(entry.packet.generation, 0u);
                entry.routing.inputPort = warp.template broadcast<std::uint32_t>(entry.routing.inputPort, 0u);
                entry.routing.outputPort = warp.template broadcast<std::uint32_t>(entry.routing.outputPort, 0u);
                entry.routing.outputPortCount
                    = warp.template broadcast<std::uint32_t>(entry.routing.outputPortCount, 0u);
                entry.request.firstId = warp.template broadcast<std::uint64_t>(entry.request.firstId, 0u);
                entry.request.count = warp.template broadcast<std::uint64_t>(entry.request.count, 0u);
                entry.request.seed = warp.template broadcast<std::uint64_t>(entry.request.seed, 0u);
                entry.sequenceNumber = warp.template broadcast<std::uint64_t>(entry.sequenceNumber, 0u);
            }
        }

        char* m_storage;
    };
} // namespace graph
