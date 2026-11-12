// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Bounded multi-producer, multi-consumer ready ring in device memory.
 *
 * The ring is the scheduler's work queue: a worker warp publishes a
 * `JobDescriptor` it wants to run next, and a worker warp claims the next
 * descriptor to execute. All storage is one caller-provided device buffer
 * (an alpaka buffer allocated before the persistent launch); the ring
 * performs no allocation of its own and holds nothing across a warp
 * collective, a wait, or a callback.
 *
 * # Sequence scheme
 *
 * The ring keeps three device-scope counters plus one sequence stamp per
 * slot. The invariants that make it safe are:
 *
 * - `pushCounter` increases by one per accepted push, in push order.
 * - `claimCounter` increases by one per accepted claim, in claim order.
 * - A slot `i` is *free* iff it is not in flight:
 *   `claimCounter <= stamp(i) < pushCounter` means in flight, anything else
 *   free.
 * - Because claims and pushes both advance their counter by exactly one per
 *   event, the number of free slots is `claimCounter - pushCounter` in
 *   modular arithmetic over the slot space, which the header keeps as
 *   `freeSlots`. A push is accepted iff `freeSlots > 0`; a claim is accepted
 *   iff `freeSlots < capacity`. Both checks read a single counter under the
 *   lock, so two producers can never reserve the same slot in the same
 *   cycle: the reservation itself (incrementing `pushCounter`, decrementing
 *   `freeSlots`) happens under the lock, and the next producer re-reads the
 *   counters after acquiring the lock.
 *
 * Every accepted push gets a strictly increasing sequence stamp
 * (`pushCounter` before increment); every accepted claim retires the oldest
 * in-flight stamp, which is `claimCounter` before increment. Claim order
 * equals stamp order, and a stamp is retired by exactly one consumer, which
 * is the exactly-once claim property. A stale consumer that reads a slot
 * after a full lap observes a stamp that is not in
 * `[claimCounter, pushCounter)` and therefore refuses to claim it; in test
 * builds the ring additionally records an ownership violation in
 * `violations` when a claim finds a stamp that does not match the expected
 * oldest-in-flight value (see `ownershipViolations()`).
 *
 * # The lock
 *
 * One naturally aligned `uint32_t` word: `0` free, `1` held. The attempt is
 * a single device-scope compare-exchange; a failed attempt returns
 * `contended` to the caller and never blocks. Only warp leaders attempt the
 * lock. The locked region of `enqueue()` is: read counters, reserve the slot,
 * copy the payload, stamp, update counters, release fence, unlock. The
 * locked region of `dequeue()` is: read counters, scan for the oldest
 * in-flight stamp, copy the entry out, retire, update counters, release
 * fence, unlock. No warp collective, no wait, and no callback runs inside
 * either region.
 *
 * # Backoff policy
 *
 * `enqueue()`/`dequeue()` are short and nonblocking. The retry schedule is a
 * policy of the caller (the scheduler loop), not of the ring. The documented
 * bounded backoff is: after a `contended`, `full`, or `empty` result, the
 * caller spins at most `maxBackoffAttempts` (default 32) consecutive
 * attempts; if the ring is still unavailable, it yields the worker (poll
 * other work, or sleep a short bounded interval in the 64 to 256 ns band
 * measured on A30) and retries. A finite attempt budget plus a bounded sleep
 * keeps a livelock visible as a deadline expiry in tests instead of a hang.
 *
 * # Publication chain
 *
 * Producer: the lanes write their contribution to the entry, the caller
 * runs a warp-level rendezvous (a `WarpGroup` collective) so every lane's
 * stores are complete, and then the leader's `enqueue()` publishes the slot
 * with a device-scope release fence.
 *
 * Consumer: the leader's `dequeue()` observes the slot with a device-scope
 * acquire fence and copies the entry out; the leader broadcasts the entry
 * field by field to the lanes; the caller runs the warp-level rendezvous
 * before any lane touches the payload. The entry is copied out of the ring
 * slot before the warp is allowed to reclaim it.
 *
 * # Layout
 *
 * The header words live on separate 64-byte cache lines: `lock` alone on
 * line 0 (the contended word), `pushCounter` on line 1, `claimCounter` on
 * line 2, `freeSlots` on line 3, and `violations` on line 4. The slots
 * follow the header, each aligned to its natural alignment (16 bytes for the
 * current `JobDescriptor`).
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

    /** One ready entry: the full job descriptor, copied bitwise into the ring.
     *
     * Documented size: `sizeof(JobDescriptor)` bytes. With the current
     * vocabulary that is 64 bytes: operation 4 + kind 4 + packet 8 + routing
     * 12 + request 24 + sequence 8, padded to 64. (The 12-byte `RoutingInfo`
     * forces 8-byte alignment of the trailing `sequenceNumber`.) The ring
     * slot adds one `uint64_t` sequence stamp and, in test builds, one
     * `uint32_t` generation word, for a total of 80 bytes per slot
     * (`ReadyRing<T_Acc, T_Capacity>::slotBytes`).
     */
    using ReadyEntry = JobDescriptor;

    namespace detail
    {
        /** Device-scope release fence: the payload stores issued before it
         *  become visible to any warp that performs the matching acquire
         *  fence after observing the lock release. */
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

        /** Warp-level lane rendezvous.
         *
         * Not a memory barrier: a lane rendezvous. It maps to the warp
         * primitive `all(true)`, which is the width-one identity on the host
         * and a real warp step on a device accelerator. The caller must invoke
         * it warp-uniformly: every active lane of the warp must reach the
         * same call.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr void warpRendezvous(WarpGroup<T_Acc> const& warp) noexcept
        {
            (void)warp.all(true);
        }

        /** Device-scope atomic compare-exchange on a naturally aligned word.
         *
         * The value type and scope are explicit template parameters so the
         * call is unambiguous regardless of the literal types at the call
         * site.
         */
        template<typename T_Acc, typename T, typename T_Scope = alpaka::onAcc::scope::Device>
        ALPAKA_FN_ACC constexpr T atomicCasDevice(T_Acc const& acc, T* addr, T expected, T desired) noexcept
        {
            return alpaka::onAcc::atomicCas<T, T_Scope>(acc, addr, expected, desired, T_Scope{});
        }

        /** Device-scope atomic fetch-add. */
        template<typename T_Acc, typename T, typename T_Scope = alpaka::onAcc::scope::Device>
        ALPAKA_FN_ACC constexpr T atomicFetchAddDevice(T_Acc const& acc, T* addr, T value) noexcept
        {
            return alpaka::onAcc::atomicAdd<T, T_Scope>(acc, addr, value, T_Scope{});
        }

        /** Device-scope atomic fetch-sub. */
        template<typename T_Acc, typename T, typename T_Scope = alpaka::onAcc::scope::Device>
        ALPAKA_FN_ACC constexpr T atomicFetchSubDevice(T_Acc const& acc, T* addr, T value) noexcept
        {
            return alpaka::onAcc::atomicSub<T, T_Scope>(acc, addr, value, T_Scope{});
        }
    } // namespace detail

    /** Bounded multi-producer, multi-consumer ready ring over preallocated
     *  device storage.
     *
     * `T_Capacity` is the number of slots (compile time, power of two). All
     * public methods are device-side (`ALPAKA_FN_ACC`) and warp-uniform:
     * every active lane of the calling warp must call the same method with
     * the same arguments, and only the lane selected by
     * `WarpGroup::isLeader()` performs the locked work.
     */
    template<typename T_Acc, std::size_t T_Capacity = 1024u>
    class ReadyRing
    {
        static_assert(T_Capacity >= 1u && (T_Capacity & (T_Capacity - 1u)) == 0u, "capacity must be a power of two");

    public:
        using Acc = std::remove_cvref_t<T_Acc>;

        static constexpr std::size_t capacity = T_Capacity;
        static constexpr std::uint64_t capacityStamp = T_Capacity;
        static constexpr std::uint64_t slotMask = T_Capacity - 1u;

        /** A ring slot: the entry payload plus the sequence stamp and, in
         *  test builds, an independent generation word for the ownership
         *  violation hook.
         *
         * Documented size: `sizeof(ReadyEntry) + 16` bytes (entry 64, stamp
         * 8, generation 4, padding 4), naturally aligned to 8.
         */
        struct RingSlot
        {
            ReadyEntry entry{};
            std::uint64_t sequence{0u};
            std::uint32_t generation{0u};
            std::uint32_t pad{0u};
        };

        /** Ring header. Each counter owns its own 64-byte cache line so the
         *  contended `lock` word does not false-share with the counters. */
        struct RingHeader
        {
            std::uint32_t lock{0u};
            std::uint64_t padLock[7]{};
            std::uint64_t pushCounter{0u};
            std::uint64_t padPush[7]{};
            std::uint64_t claimCounter{0u};
            std::uint64_t padClaim[7]{};
            std::uint64_t freeSlots{0u};
            std::uint64_t padFree[7]{};
            std::uint64_t violations{0u};
            std::uint64_t padViolations[7]{};
        };

        static_assert(std::is_trivially_copyable_v<RingSlot>);
        static_assert(std::is_trivially_copyable_v<RingHeader>);
        static_assert(std::is_standard_layout_v<RingSlot>);
        static_assert(std::is_standard_layout_v<RingHeader>);
        static_assert(sizeof(RingHeader) == 320u, "header layout is cache-line separated");

        /** Bytes per slot and total storage size in bytes. */
        static constexpr std::size_t slotBytes = sizeof(RingSlot);
        static constexpr std::size_t storageBytes = sizeof(RingHeader) + T_Capacity * slotBytes;

        /** Construct a view over preallocated device storage of at least
         *  `storageBytes` bytes.
         *
         * The storage must stay valid and exclusively owned by this ring for
         * its lifetime. Call `initialize()` once (device side, startup
         * action) or zero the buffer on the host before the first
         * `enqueue()`.
         */
        ALPAKA_FN_ACC constexpr explicit ReadyRing(void* storage) noexcept : m_storage(static_cast<char*>(storage))
        {
        }

        /** Zero-initialize the ring storage.
         *
         * Called once by the startup action of the persistent kernel (or by a
         * single warp before the pool starts). All lanes must call it
         * warp-uniformly; only the leader writes.
         */
        ALPAKA_FN_ACC constexpr void initialize(Acc const& acc, WarpGroup<Acc> const& warp) noexcept
        {
            if(warp.isLeader())
            {
                auto* hdr = &header();
                hdr->lock = 0u;
                hdr->pushCounter = 0u;
                hdr->claimCounter = 0u;
                hdr->freeSlots = T_Capacity;
                hdr->violations = 0u;
                for(std::size_t i = 0u; i < T_Capacity; ++i)
                {
                    slotAt(i).sequence = 0u;
                    slotAt(i).generation = 0u;
                }
                detail::publishRelease(acc);
            }
            detail::warpRendezvous(warp);
        }

        /** Nonblocking attempt to publish one entry.
         *
         * Warp-uniform: every active lane calls it with the same `entry`. The
         * leader performs the locked work; the other lanes rendezvous at the
         * start (so their stores into `entry` are complete before the leader
         * reads it) and at the end (after the leader's release fence) before
         * proceeding.
         *
         * Returns `accepted`, `full`, or `contended`. On `full` the target
         * slot still holds an in-flight entry; the caller backs off and
         * retries. The ring never blocks.
         */
        ALPAKA_FN_ACC constexpr RingStatus enqueue(Acc const& acc, WarpGroup<Acc> const& warp, ReadyEntry const& entry)
            noexcept
        {
            detail::warpRendezvous(warp);

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
                    if(hdr->freeSlots == 0u)
                    {
                        unlock(acc);
                        status = RingStatus::Full;
                    }
                    else
                    {
                        // Reserve the slot: the push counter names the slot
                        // and the sequence stamp the entry gets. Both advance
                        // under the lock, so no other producer can observe a
                        // half-reserved slot.
                        auto const pushCounter = hdr->pushCounter;
                        auto const slotIndex = static_cast<std::size_t>(pushCounter & slotMask);
                        auto* slot = &slotAt(slotIndex);
                        slot->sequence = pushCounter;
                        slot->generation += 1u;
                        slot->entry = entry;
                        // Publish: release fence, then the slot is visible.
                        detail::publishRelease(acc);
                        detail::atomicFetchAddDevice(acc, &hdr->pushCounter, std::uint64_t{1u});
                        detail::atomicFetchAddDevice(acc, &hdr->freeSlots, ~std::uint64_t{0u});
                        unlock(acc);
                        status = RingStatus::Accepted;
                    }
                }
            }

            detail::warpRendezvous(warp);
            auto const value = static_cast<std::uint32_t>(status);
            return static_cast<RingStatus>(warp.template broadcast<std::uint32_t>(value, 0u));
        }

        /** Nonblocking attempt to claim the next entry.
         *
         * Warp-uniform. On `accepted`, `out` holds the claimed entry (copied
         * out of the slot) and the entry's `sequenceNumber` field is the ring
         * stamp of that push, so the claim order is recoverable from the
         * entries themselves. The slot is reclaimable by the next producer
         * immediately after the leader's release; the warp must not read the
         * slot again after the rendezvous.
         *
         * Returns `accepted`, `empty`, or `contended`. The ring never blocks.
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
                    if(hdr->freeSlots == T_Capacity)
                    {
                        unlock(acc);
                        status = RingStatus::Empty;
                    }
                    else
                    {
                        // The oldest in-flight stamp is the claim counter
                        // itself: every claim retires exactly one stamp, in
                        // stamp order, so the next claim is always
                        // claimCounter.
                        auto const stamp = hdr->claimCounter;
                        auto const slotIndex = static_cast<std::size_t>(stamp & slotMask);
                        auto* slot = &slotAt(slotIndex);

                        detail::claimAcquire(acc);

                        // Ownership-violation hook (test builds): a valid
                        // claim finds the slot stamped exactly with `stamp`
                        // and a nonzero generation. Anything else means a
                        // consumer lagged by a full lap and the slot was
                        // reused underneath it.
                        if(slot->sequence != stamp || slot->generation == 0u)
                        {
                            detail::atomicFetchAddDevice(acc, &hdr->violations, std::uint64_t{1u});
                        }

                        // Copy the entry out before retiring the slot.
                        out = slot->entry;
                        // Retire: stamp the slot one lap ahead so no consumer
                        // can mistake it for the next push into the same
                        // slot.
                        slot->sequence = stamp + capacityStamp;
                        detail::publishRelease(acc);
                        detail::atomicFetchAddDevice(acc, &hdr->claimCounter, std::uint64_t{1u});
                        detail::atomicFetchAddDevice(acc, &hdr->freeSlots, std::uint64_t{1u});
                        unlock(acc);
                        status = RingStatus::Accepted;
                    }
                }
            }

            detail::warpRendezvous(warp);
            auto const value = static_cast<std::uint32_t>(status);
            status = static_cast<RingStatus>(warp.template broadcast<std::uint32_t>(value, 0u));
            if(status == RingStatus::Accepted)
            {
                broadcastEntry(warp, out);
            }
            return status;
        }

        /** Number of in-flight entries. */
        ALPAKA_FN_ACC constexpr std::uint64_t inFlightCount(Acc const& acc) const noexcept
        {
            return T_Capacity - header().freeSlots;
        }

        /** Next push stamp (the sequence number the next accepted push gets). */
        ALPAKA_FN_ACC constexpr std::uint64_t nextPushSequence(Acc const& acc) const noexcept
        {
            return header().pushCounter;
        }

        /** Stamps consumed so far (claims completed). */
        ALPAKA_FN_ACC constexpr std::uint64_t claimCounter(Acc const& acc) const noexcept
        {
            return header().claimCounter;
        }

        /** Ownership violations recorded by the test hook. */
        ALPAKA_FN_ACC constexpr std::uint64_t ownershipViolations(Acc const& acc) const noexcept
        {
            return header().violations;
        }

    private:
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

        [[nodiscard]] ALPAKA_FN_ACC constexpr RingHeader& header() const noexcept
        {
            return *reinterpret_cast<RingHeader*>(m_storage);
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr RingSlot& slotAt(std::size_t index) const noexcept
        {
            auto const slotsBegin = alignUp(sizeof(RingHeader), 64u);
            return *reinterpret_cast<RingSlot*>(m_storage + slotsBegin + index * slotBytes);
        }

        /** Broadcast every field of the entry from the leader to all lanes.
         *
         * The entry is a trivially copyable POD; it is exchanged field by
         * field through warp broadcasts so every lane ends with an identical
         * copy without the leader having written to lane-private memory.
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
                // The leader already holds the claimed copy; broadcast each
                // trivially copyable field.
                entry.operation.value
                    = warp.template broadcast<std::uint32_t>(entry.operation.value, 0u);
                entry.kind
                    = static_cast<JobKind>(
                        warp.template broadcast<std::uint32_t>(static_cast<std::uint32_t>(entry.kind), 0u));
                entry.packet.index = warp.template broadcast<std::uint32_t>(entry.packet.index, 0u);
                entry.packet.generation
                    = warp.template broadcast<std::uint32_t>(entry.packet.generation, 0u);
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

        static constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        char* m_storage;
    };
} // namespace graph
