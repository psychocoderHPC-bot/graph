// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/system_atomic.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>

/** @file
 *
 * Mapped, concurrently CPU/GPU-accessible single-producer/single-consumer
 * ingress ring.
 *
 * The ring carries small request descriptors from one host producer to one GPU
 * importer lane. Storage is a caller-created mapped pinned allocation
 * (`alpaka::onHost::allocMapped`) initialised before the persistent kernel
 * starts; nothing is allocated afterwards.
 *
 * Protocol (design section 5.1):
 * 1. The host observes `consumed` with a system-scope acquire load, writes a
 *    batch into the slots after `published`, then publishes the new head with
 *    a system-scope release store.
 * 2. The importer observes `published` with a system-scope acquire load, reads
 *    the batch, resolves/stages the accepted jobs, and only after taking
 *    ownership of every acknowledged descriptor publishes `consumed` with a
 *    system-scope release store.
 * 3. The explicit `JobKind::END_OF_INPUT` marker is inserted through the same
 *    ring; it is only accepted when it fits and closes the host side. It is
 *    ordered after every earlier descriptor and the importer processes it last.
 *
 * Ordering rules:
 * - Only acquire loads and release stores touch the shared counters. This ring
 *   never uses a read-modify-write handshake: concurrent host/device RMWs on
 *   mapped memory lose updates on the target GPU, so the protocol is pure
 *   load/store ordering.
 * - `graph::systemLoadAcquire`/`graph::systemStoreRelease` on the device and
 *   `graph::host::hostLoadAcquire`/`graph::host::hostStoreRelease` on the host.
 * - A release store alone publishes the preceding descriptor writes; nothing in
 *   this header depends on `__threadfence_system` for publication. The extra
 *   acquire/release fences around each ordered counter pair are explicit
 *   system-scope markers (design 5.2); they are redundant for publication and
 *   are not a second barrier, so they must not be relied on as one.
 *
 * Threading:
 * - Exactly one host thread may call a mutating host method that writes the
 *   `published` counter (`tryEnqueue`, `enqueueBlocking`, `closeInput`,
 *   `setCountersForTest`) at a time. Callers that share the ring must serialise
 *   their submissions before entering it.
 * - `requestStop()` is the one documented exception. It only release-stores the
 *   `closed` flag, which no producer writes, so a supervising host thread may
 *   call it while a producer is blocked inside `enqueueBlocking`. It is the
 *   supported way to cancel a blocked producer. A graceful `closeInput()` is
 *   *not* safe in that position: it publishes the ordered end marker through
 *   `published`, so it needs both a free slot and exclusive producer ownership,
 *   which a producer waiting for space does not grant.
 * - Exactly one device lane may call `pollAndImport`. Other workers coordinate
 *   through device-scope synchronisation on their own ready ring; they never
 *   touch this host-facing ring. `pollAndImportHost` is the CPU-side twin and
 *   follows the same single-consumer rule; it may run concurrently with the
 *   producer.
 *
 * This header is header-only and dispatches on the accelerator API tag
 * (`alpaka::api::Cuda` / `alpaka::api::Host`). It contains no accelerator
 * selection through preprocessor switches.
 */
namespace graph
{
    /** Cache-line size used for counter separation.
     *
     * 128 bytes covers the 64-byte x86 line and the larger NVIDIA L2 sector
     * granularity, so the two counters never share a line even when a backend
     * fetches 128 bytes at a time.
     */
    inline constexpr std::size_t ingressCacheLineSize = 128u;

    /** Status of a host submission. */
    enum class EnqueueStatus
    {
        /// Everything requested was accepted and published.
        accepted,
        /// The ring had less free space than requested; `accepted` reports the
        /// prefix that was written. The ring remains open.
        ring_full,
        /// The ring is closed; this call published nothing. `enqueueBlocking`
        /// additionally reports in `accepted` a prefix it published before a
        /// concurrent `requestStop()` became visible.
        closed,
        /// Invalid input or an internal counter/range violation; nothing was
        /// published. See `EnqueueResult::error`.
        error
    };

    /** Status of one importer poll. */
    enum class ImportStatus
    {
        /// No descriptor was available.
        nothing_new,
        /// At least one descriptor was staged; no end marker was seen.
        imported,
        /// The ordered end-of-input marker was staged as the last descriptor of
        /// this batch. `consumed` now covers it.
        end_marker_seen,
        /// Protocol or argument error; `consumed` was not advanced.
        error
    };

    /** Detailed error code for the `error` statuses. */
    enum class IngressError : std::uint32_t
    {
        none = 0u,
        /// A descriptor used `JobKind::END_OF_INPUT` outside `closeInput`.
        invalid_descriptor = 1u,
        /// A descriptor appeared after the end-of-input marker.
        descriptor_after_end = 2u,
        /// `published - consumed` exceeded the capacity.
        counter_corrupt = 3u,
        /// The staging or batch bound was zero or too small.
        staging_too_small = 4u,
        /// The requested capacity violates the counter-bound contract.
        capacity_invalid = 5u,
        /// A debug counter overflowed; the protocol aborted safely.
        counter_overflow = 6u
    };

    inline constexpr std::string_view toString(EnqueueStatus status) noexcept
    {
        switch(status)
        {
        case EnqueueStatus::accepted:
            return "accepted";
        case EnqueueStatus::ring_full:
            return "ring_full";
        case EnqueueStatus::closed:
            return "closed";
        case EnqueueStatus::error:
            return "error";
        }
        return "unknown";
    }

    inline constexpr std::string_view toString(ImportStatus status) noexcept
    {
        switch(status)
        {
        case ImportStatus::nothing_new:
            return "nothing_new";
        case ImportStatus::imported:
            return "imported";
        case ImportStatus::end_marker_seen:
            return "end_marker_seen";
        case ImportStatus::error:
            return "error";
        }
        return "unknown";
    }

    inline constexpr std::string_view toString(IngressError error) noexcept
    {
        switch(error)
        {
        case IngressError::none:
            return "none";
        case IngressError::invalid_descriptor:
            return "invalid_descriptor";
        case IngressError::descriptor_after_end:
            return "descriptor_after_end";
        case IngressError::counter_corrupt:
            return "counter_corrupt";
        case IngressError::staging_too_small:
            return "staging_too_small";
        case IngressError::capacity_invalid:
            return "capacity_invalid";
        case IngressError::counter_overflow:
            return "counter_overflow";
        }
        return "unknown";
    }

    /** Result of one host submission. */
    struct EnqueueResult
    {
        EnqueueStatus status{EnqueueStatus::error};
        IngressError error{IngressError::none};
        /// Descriptors written and published by this call.
        std::uint32_t accepted{0u};
        /// Descriptors the caller offered (after a checked size conversion).
        std::uint32_t requested{0u};

        constexpr bool allAccepted() const noexcept
        {
            return status == EnqueueStatus::accepted;
        }
    };

    /** Result of one importer poll.
     *
     * `imported` counts every descriptor staged, including the end marker.
     * `jobCount` counts only non-control descriptors. On `ImportStatus::error`
     * both are zero and the staging contents are unspecified: nothing was
     * acknowledged and `consumed` did not move.
     */
    struct ImportBatch
    {
        ImportStatus status{ImportStatus::nothing_new};
        IngressError error{IngressError::none};
        std::uint32_t imported{0u};
        std::uint32_t jobCount{0u};
        bool endMarkerSeen{false};
    };

    /** Communication block shared by host and device.
     *
     * Every field that is touched by both sides lives on its own cache line.
     * `published` is written only by the host; `consumed`, `totalImported` and
     * `importError` are written only by the importer. `totalEnqueued` and
     * `peakOccupancy` are host-only diagnostics.
     *
     * The 128-byte alignment of the first member also aligns the whole block.
     */
    struct alignas(ingressCacheLineSize) IngressControl
    {
        /// Modular head, host producer only.
        alignas(ingressCacheLineSize) std::uint64_t published{0u};
        /// Modular tail, device importer only.
        alignas(ingressCacheLineSize) std::uint64_t consumed{0u};
        /// Host-only diagnostic: descriptors published, including the marker.
        alignas(ingressCacheLineSize) std::uint64_t totalEnqueued{0u};
        /// Importer-written diagnostic: descriptors acknowledged.
        alignas(ingressCacheLineSize) std::uint64_t totalImported{0u};
        /// Host-only diagnostic: highest observed published - consumed.
        alignas(ingressCacheLineSize) std::uint64_t peakOccupancy{0u};
        /// Host-only close flag, published with a release store.
        alignas(ingressCacheLineSize) std::uint32_t closed{0u};
        /// Importer-written error code (IngressError); host-facing diagnostic.
        alignas(ingressCacheLineSize) std::uint32_t importError{0u};
        /// Reserved padding; keeps the block a multiple of the line size.
        alignas(ingressCacheLineSize) std::uint64_t reserved{0u};
    };

    static_assert(alignof(IngressControl) == ingressCacheLineSize);
    static_assert(sizeof(IngressControl) % ingressCacheLineSize == 0u);
    static_assert(offsetof(IngressControl, consumed) >= ingressCacheLineSize);
    static_assert(
        offsetof(IngressControl, totalEnqueued) - offsetof(IngressControl, consumed) >= ingressCacheLineSize);
    static_assert(
        offsetof(IngressControl, totalImported) - offsetof(IngressControl, totalEnqueued) >= ingressCacheLineSize);
    static_assert(
        offsetof(IngressControl, peakOccupancy) - offsetof(IngressControl, totalImported) >= ingressCacheLineSize);
    static_assert(offsetof(IngressControl, closed) - offsetof(IngressControl, peakOccupancy) >= ingressCacheLineSize);
    static_assert(offsetof(IngressControl, importError) - offsetof(IngressControl, closed) >= ingressCacheLineSize);
    static_assert(offsetof(IngressControl, reserved) - offsetof(IngressControl, importError) >= ingressCacheLineSize);
    static_assert(alignof(IngressControl::published) >= alignof(std::uint64_t));

    namespace detail
    {
        /** Host-side ordered access to one mapped counter.
         *
         * The boolean selects the CUDA system-scope reference for a ring
         * allocated on a CUDA device. It is set from the device API tag at
         * creation time, not by an accelerator preprocessor switch. The CPU
         * build only ever stores `false`.
         */
        template<typename T>
        inline T hostCounterLoadAcquire(bool systemScope, T const* ptr)
        {
#if GRAPH_HAS_CUDA
            if(systemScope)
            {
                ::cuda::atomic_ref<T, ::cuda::thread_scope_system> ref(*const_cast<T*>(ptr));
                return graph::host::hostLoadAcquire(ref);
            }
#endif
            std::atomic_ref<T> ref(*const_cast<T*>(ptr));
            return graph::host::hostLoadAcquire(ref);
        }

        template<typename T>
        inline void hostCounterStoreRelease(bool systemScope, T* ptr, T value)
        {
#if GRAPH_HAS_CUDA
            if(systemScope)
            {
                ::cuda::atomic_ref<T, ::cuda::thread_scope_system> ref(*ptr);
                graph::host::hostStoreRelease(ref, value);
                return;
            }
#endif
            std::atomic_ref<T> ref(*ptr);
            graph::host::hostStoreRelease(ref, value);
        }

        /** Host system-scope acquire fence after observing importer progress. */
        inline void hostCounterFenceAcquire(bool systemScope)
        {
#if GRAPH_HAS_CUDA
            if(systemScope)
            {
                ::cuda::atomic_thread_fence(::cuda::std::memory_order_acquire, ::cuda::thread_scope_system);
                return;
            }
#else
            static_cast<void>(systemScope);
#endif
            graph::host::hostFenceAcquire();
        }

        /** Host system-scope release fence before acknowledging consumed slots. */
        inline void hostCounterFenceRelease(bool systemScope)
        {
#if GRAPH_HAS_CUDA
            if(systemScope)
            {
                ::cuda::atomic_thread_fence(::cuda::std::memory_order_release, ::cuda::thread_scope_system);
                return;
            }
#else
            static_cast<void>(systemScope);
#endif
            graph::host::hostFenceRelease();
        }

        inline std::uint32_t checkedU32(std::size_t value, bool& ok) noexcept
        {
            if(value > std::numeric_limits<std::uint32_t>::max())
            {
                ok = false;
                return 0u;
            }
            return static_cast<std::uint32_t>(value);
        }

    } // namespace detail

    /** Host-facing view of a mapped ingress ring.
     *
     * The view is a plain value type with raw pointers. Pass a copy to the
     * persistent kernel; keep the owning `IngressRing` alive on the host for as
     * long as the kernel may touch the storage.
     */
    template<typename T_Descriptor = JobDescriptor, std::uint32_t T_capacity = 256u>
    class IngressRing
    {
    public:
        using Descriptor = T_Descriptor;

        static constexpr std::uint32_t capacity = T_capacity;
        static constexpr std::uint64_t counterRange = std::numeric_limits<std::uint64_t>::max();

        static_assert(capacity >= 1u, "graph: the ingress ring needs at least one slot");
        static_assert(
            capacity <= counterRange / 2u,
            "graph: ingress capacity must stay well below half the modular counter range");
        static_assert(
            capacity <= std::numeric_limits<std::uint32_t>::max(),
            "graph: the ingress capacity must fit the 32-bit batch accounting");
        static_assert(
            std::is_trivially_copyable_v<Descriptor>,
            "graph: ingress descriptors must be trivially copyable");
        static_assert(std::is_standard_layout_v<Descriptor>, "graph: ingress descriptors must be standard layout");
        static_assert(
            sizeof(IngressControl) % alignof(Descriptor) == 0u,
            "graph: the slot array must stay naturally aligned after the control block");

        /** Slot array plus control block, shared by host and device. */
        struct View
        {
            IngressControl* control{nullptr};
            Descriptor* slots{nullptr};
            std::uint32_t slotCount{0u};
            std::uint32_t systemScopeHostAtomics{0u};

            /** Resolver that accepts every descriptor.
             *
             * The real pool replaces this with a resolver that maps operation
             * ids to device function pointers. Keeping it a policy here avoids
             * a dependency on the operation table.
             */
            struct AcceptAll
            {
                ALPAKA_FN_ACC bool operator()(Descriptor const&) const noexcept
                {
                    return true;
                }
            };

            /** Import at most `maxBatch` descriptors into GPU-owned staging.
             *
             * Call from exactly one lane. The descriptors are copied in ring
             * order, ids are passed through @p resolver, and `consumed` is
             * advanced only after every staged descriptor has been copied and
             * accepted.
             *
             * @param staging caller-owned device memory with room for at least
             *                `min(stagingCapacity, maxBatch)` descriptors.
             * @param stagingCapacity number of descriptors `staging` can hold.
             * @param maxBatch upper bound for one poll; keeps the importer's
             *                 scheduling opportunity bounded.
             * @param resolver returns false to reject an unresolvable id; on a
             *                 rejection nothing is acknowledged.
             */
            template<typename T_Acc, typename T_Resolver>
            ALPAKA_FN_ACC ImportBatch pollAndImport(
                T_Acc const& acc,
                Descriptor* staging,
                std::uint32_t stagingCapacity,
                std::uint32_t maxBatch,
                T_Resolver resolver) const
            {
                ImportBatch batch{};
                if(control == nullptr || slots == nullptr)
                {
                    batch.status = ImportStatus::error;
                    batch.error = IngressError::staging_too_small;
                    return batch;
                }

                // Acquire: everything the host wrote before publishing `head`
                // is visible. Never poll with an RMW.
                std::uint64_t const head = graph::systemLoadAcquire(acc, &control->published);
                // Explicit system acquire fence. The acquire load above already
                // orders the following plain descriptor reads; the fence makes
                // the ordered transport explicit and matches design 5.2, which
                // requires an acquire fence after observing progress.
                graph::systemFenceAcquire(acc);
                std::uint64_t const tail = graph::systemLoadAcquire(acc, &control->consumed);

                batch = importRange(head, tail, staging, stagingCapacity, maxBatch, resolver);
                if(batch.status == ImportStatus::error)
                {
                    graph::systemStoreRelease(acc, &control->importError, static_cast<std::uint32_t>(batch.error));
                    return batch;
                }
                if(batch.imported == 0u)
                    return batch;

                // The importer is the only writer of totalImported, so a plain
                // read/add/store is not a cross-side RMW (never fetch_add here).
                std::uint64_t const importedBefore = graph::systemLoadAcquire(acc, &control->totalImported);
                if(importedBefore > std::numeric_limits<std::uint64_t>::max() - batch.imported)
                {
                    batch.status = ImportStatus::error;
                    batch.error = IngressError::counter_overflow;
                    graph::systemStoreRelease(acc, &control->importError, static_cast<std::uint32_t>(batch.error));
                    return batch;
                }
                graph::systemStoreRelease(acc, &control->totalImported, importedBefore + batch.imported);

                // Release: all descriptor reads and the diagnostic update are
                // ordered before the acknowledgement, so the host cannot
                // overwrite a slot the importer is still reading. No RMW and no
                // explicit __threadfence_system are needed: the release store
                // is the publication.
                graph::systemFenceRelease(acc);
                graph::systemStoreRelease(acc, &control->consumed, tail + batch.imported);
                return batch;
            }

            template<typename T_Acc>
            ALPAKA_FN_ACC ImportBatch pollAndImport(
                T_Acc const& acc,
                Descriptor* staging,
                std::uint32_t stagingCapacity,
                std::uint32_t maxBatch) const
            {
                return pollAndImport(acc, staging, stagingCapacity, maxBatch, AcceptAll{});
            }

            /** Host-side importer for the CPU backend and host-only tests.
             *
             * Uses the same algorithm and ordering rules as the device path but
             * the host primitive set. On a CUDA ring it selects the CUDA
             * system-scope references for the mapped allocation. This is also
             * the "host importer thread" path used by the CPU integration test;
             * it is not GPU evidence.
             *
             * The error contract matches `pollAndImport`: a rejection by
             * @p resolver, a counter violation or an argument error returns
             * `ImportStatus::error`, records the code in `importError` and
             * leaves `consumed` unchanged. The ring is not poisoned, so a later
             * poll with an accepting resolver imports the same descriptors
             * again.
             */
            template<typename T_Resolver>
            ImportBatch pollAndImportHost(
                Descriptor* staging,
                std::uint32_t stagingCapacity,
                std::uint32_t maxBatch,
                T_Resolver resolver) const
            {
                ImportBatch batch{};
                if(control == nullptr || slots == nullptr)
                {
                    batch.status = ImportStatus::error;
                    batch.error = IngressError::staging_too_small;
                    return batch;
                }

                bool const systemScope = systemScopeHostAtomics != 0u;
                std::uint64_t const head = detail::hostCounterLoadAcquire(systemScope, &control->published);
                detail::hostCounterFenceAcquire(systemScope);
                std::uint64_t const tail = detail::hostCounterLoadAcquire(systemScope, &control->consumed);

                batch = importRange(head, tail, staging, stagingCapacity, maxBatch, resolver);
                if(batch.status == ImportStatus::error)
                {
                    detail::hostCounterStoreRelease(
                        systemScope,
                        &control->importError,
                        static_cast<std::uint32_t>(batch.error));
                    return batch;
                }
                if(batch.imported == 0u)
                    return batch;

                std::uint64_t const importedBefore
                    = detail::hostCounterLoadAcquire(systemScope, &control->totalImported);
                if(importedBefore > std::numeric_limits<std::uint64_t>::max() - batch.imported)
                {
                    batch.status = ImportStatus::error;
                    batch.error = IngressError::counter_overflow;
                    detail::hostCounterStoreRelease(
                        systemScope,
                        &control->importError,
                        static_cast<std::uint32_t>(batch.error));
                    return batch;
                }
                detail::hostCounterStoreRelease(systemScope, &control->totalImported, importedBefore + batch.imported);
                detail::hostCounterFenceRelease(systemScope);
                detail::hostCounterStoreRelease(systemScope, &control->consumed, tail + batch.imported);
                return batch;
            }

            /** Host-side importer that accepts every descriptor. */
            ImportBatch pollAndImportHost(Descriptor* staging, std::uint32_t stagingCapacity, std::uint32_t maxBatch)
                const
            {
                return pollAndImportHost(staging, stagingCapacity, maxBatch, AcceptAll{});
            }

        private:
            /** Copy and validate a ring range.
             *
             * The caller performs the acquire load of `head` before calling and
             * the release store of `consumed` after it returns. Reads here must
             * not be hoisted before the acquire.
             */
            template<typename T_Resolver>
            ALPAKA_FN_ACC ImportBatch importRange(
                std::uint64_t head,
                std::uint64_t tail,
                Descriptor* staging,
                std::uint32_t stagingCapacity,
                std::uint32_t maxBatch,
                T_Resolver resolver) const
            {
                ImportBatch batch{};
                if(slots == nullptr || staging == nullptr || slotCount == 0u || stagingCapacity == 0u
                   || maxBatch == 0u)
                {
                    batch.status = ImportStatus::error;
                    batch.error = IngressError::staging_too_small;
                    return batch;
                }

                std::uint64_t const available = head - tail;
                if(available > slotCount)
                {
                    batch.status = ImportStatus::error;
                    batch.error = IngressError::counter_corrupt;
                    return batch;
                }
                if(available == 0u)
                {
                    batch.status = ImportStatus::nothing_new;
                    return batch;
                }

                std::uint64_t limit = available;
                if(limit > maxBatch)
                    limit = maxBatch;
                if(limit > stagingCapacity)
                    limit = stagingCapacity;

                std::uint32_t n = 0u;
                while(static_cast<std::uint64_t>(n) < limit)
                {
                    Descriptor const descriptor = slots[(tail + n) % slotCount];
                    if(!resolver(descriptor))
                    {
                        batch.status = ImportStatus::error;
                        batch.error = IngressError::invalid_descriptor;
                        return batch;
                    }
                    staging[n] = descriptor;
                    ++n;

                    if(descriptor.kind == JobKind::END_OF_INPUT)
                    {
                        batch.endMarkerSeen = true;
                        // The close protocol and the single producer guarantee
                        // that the marker is last. Anything behind it means a
                        // producer bypassed `closeInput`.
                        if(static_cast<std::uint64_t>(n) < available)
                        {
                            batch.status = ImportStatus::error;
                            batch.error = IngressError::descriptor_after_end;
                            return batch;
                        }
                        break;
                    }
                }

                batch.imported = n;
                batch.jobCount = n - (batch.endMarkerSeen ? 1u : 0u);
                batch.status = batch.endMarkerSeen ? ImportStatus::end_marker_seen : ImportStatus::imported;
                return batch;
            }
        };

        /** Allocate and initialise the mapped ring.
         *
         * Must be called before the persistent kernel starts. The counters may
         * be initialised near their maximum to exercise rollover; the values
         * are validated so that `published - consumed <= capacity`.
         */
        template<typename T_Device>
        static IngressRing create(
            T_Device const& device,
            std::uint64_t initialPublished = 0u,
            std::uint64_t initialConsumed = 0u)
        {
            using Api = ALPAKA_TYPEOF(device.getApi());
            static_assert(
                detail::isCudaApi<Api> || detail::isHostApi<Api>,
                "graph: the ingress ring is implemented for the CPU and NVIDIA CUDA backends only");

            // Runtime mirror of the static counter bound, documented because the
            // capacity is a template value.
            if(capacity == 0u || static_cast<std::uint64_t>(capacity) > counterRange / 2u)
                throw std::invalid_argument("graph: ingress capacity violates the counter bound");

            constexpr std::size_t payloadBytes
                = sizeof(IngressControl) + static_cast<std::size_t>(capacity) * sizeof(Descriptor);
            constexpr std::size_t allocatedBytes = payloadBytes + ingressCacheLineSize;
            constexpr std::size_t words = (allocatedBytes + sizeof(std::uint64_t) - 1u) / sizeof(std::uint64_t);
            static_assert(
                words <= std::numeric_limits<std::uint32_t>::max(),
                "graph: the mapped ingress allocation is too large for the alpaka extent type");

            auto buffer = alpaka::onHost::allocMapped<std::uint64_t>(
                device,
                alpaka::Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(words)});

            auto* raw = reinterpret_cast<std::byte*>(alpaka::onHost::data(buffer));
            auto* base = reinterpret_cast<std::byte*>(
                (reinterpret_cast<std::uintptr_t>(raw) + (ingressCacheLineSize - 1u))
                & ~(static_cast<std::uintptr_t>(ingressCacheLineSize) - 1u));

            IngressRing ring;
            // Keep the owning alpaka buffer alive through the type-erased owner.
            auto holder = std::make_shared<ALPAKA_TYPEOF(buffer)>(std::move(buffer));
            ring.m_owner = holder;

            ring.m_view.control = reinterpret_cast<IngressControl*>(base);
            *ring.m_view.control = IngressControl{};
            ring.m_view.slots = reinterpret_cast<Descriptor*>(base + sizeof(IngressControl));
            ring.m_view.slotCount = capacity;
            ring.m_view.systemScopeHostAtomics = detail::isCudaApi<Api> ? 1u : 0u;

            for(std::uint32_t i = 0u; i < capacity; ++i)
                ring.m_view.slots[i] = Descriptor{};

            ring.setCountersForTest(initialPublished, initialConsumed);
            return ring;
        }

        /** Plain value view for the persistent kernel. */
        View view() const noexcept
        {
            return m_view;
        }

        /** Submit a batch without blocking.
         *
         * Accepts the largest prefix that fits in the free space. A submission
         * after `closeInput()` returns `closed` and mutates nothing. Descriptors
         * carrying `JobKind::END_OF_INPUT` are rejected here; use `closeInput()`
         * so the close ordering is enforced.
         */
        EnqueueResult tryEnqueue(std::span<Descriptor const> descriptors)
        {
            EnqueueResult result;
            if(m_view.control == nullptr)
            {
                result.status = EnqueueStatus::error;
                result.error = IngressError::capacity_invalid;
                return result;
            }

            bool ok = true;
            result.requested = detail::checkedU32(descriptors.size(), ok);
            if(!ok)
            {
                result.status = EnqueueStatus::error;
                result.error = IngressError::counter_overflow;
                return result;
            }

            if(isClosed())
                return makeClosed(result);

            return tryEnqueueInternal(descriptors, false);
        }

        /** Blocking convenience wrapper.
         *
         * Waits for free space until @p timeout elapses. Returns `ring_full`
         * with the accepted prefix on timeout, or `closed` when the ring is
         * already closed or `requestStop()` is called while waiting; in that
         * case `accepted` still reports the prefix published before the close
         * became visible. A zero timeout is equivalent to `tryEnqueue`. An
         * empty submission is forwarded to `tryEnqueue`, so an empty batch on a
         * closed ring still reports `closed`.
         *
         * The wait is bounded by @p timeout plus one submission attempt. It
         * yields between attempts and re-checks both the deadline and the close
         * flag on every iteration, so it cannot spin forever.
         */
        EnqueueResult enqueueBlocking(
            std::span<Descriptor const> descriptors,
            std::chrono::nanoseconds timeout = std::chrono::seconds(1))
        {
            auto const deadline = std::chrono::steady_clock::now() + timeout;
            std::size_t offset = 0u;
            std::uint32_t total = 0u;
            bool ok = true;
            std::uint32_t const requested = detail::checkedU32(descriptors.size(), ok);
            if(!ok)
            {
                EnqueueResult result;
                result.status = EnqueueStatus::error;
                result.error = IngressError::counter_overflow;
                return result;
            }
            // Forward the empty case so the closed/error status is reported
            // instead of a vacuous `accepted`.
            if(descriptors.empty())
                return tryEnqueue(descriptors);

            while(offset < descriptors.size())
            {
                EnqueueResult const result = tryEnqueue(descriptors.subspan(offset));
                if(result.status == EnqueueStatus::error || result.status == EnqueueStatus::closed)
                {
                    EnqueueResult combined;
                    combined.status = result.status;
                    combined.error = result.error;
                    combined.accepted = total;
                    combined.requested = requested;
                    return combined;
                }

                total += result.accepted;
                offset += result.accepted;
                if(offset >= descriptors.size())
                    break;

                if(std::chrono::steady_clock::now() >= deadline)
                {
                    EnqueueResult combined;
                    combined.status = EnqueueStatus::ring_full;
                    combined.accepted = total;
                    combined.requested = requested;
                    return combined;
                }
                std::this_thread::yield();
            }

            EnqueueResult combined;
            combined.status = EnqueueStatus::accepted;
            combined.accepted = total;
            combined.requested = requested;
            return combined;
        }

        /** Insert the ordered end-of-input marker if it fits.
         *
         * Once accepted, every later `tryEnqueue` returns `closed`. The marker
         * is visible to the importer only after all earlier descriptors, so the
         * device processes it last.
         *
         * It needs one free slot: on a full ring it reports `ring_full` and the
         * input side stays open. Because it writes `published`, it must be
         * serialised with the producer; use `requestStop()` to cancel a blocked
         * producer instead.
         */
        EnqueueResult closeInput()
        {
            EnqueueResult result;
            if(m_view.control == nullptr)
            {
                result.status = EnqueueStatus::error;
                result.error = IngressError::capacity_invalid;
                return result;
            }
            if(isClosed())
                return makeClosed(result);

            Descriptor marker{};
            marker.kind = JobKind::END_OF_INPUT;
            result = tryEnqueueInternal(std::span<Descriptor const>{&marker, 1u}, true);
            if(result.status == EnqueueStatus::accepted && result.accepted == 1u)
            {
                bool const systemScope = m_view.systemScopeHostAtomics != 0u;
                detail::hostCounterStoreRelease(systemScope, &m_view.control->closed, std::uint32_t{1u});
            }
            return result;
        }

        /** Cancel the input side and terminate a blocked `enqueueBlocking`.
         *
         * Release-stores the close flag only. It publishes no descriptor and
         * does not touch `published`, so a supervising host thread may call it
         * while the producer is blocked inside `enqueueBlocking`: the producer
         * stays the only writer of `published`. Every later submission and
         * `closeInput()` then reports `closed`.
         *
         * This is not a graceful close. No ordered end-of-input marker is
         * enqueued, so the importer simply sees no further descriptors;
         * descriptors already published stay importable. Use `closeInput()`
         * when the importer must observe the ordered end marker.
         */
        void requestStop() noexcept
        {
            if(m_view.control == nullptr)
                return;
            detail::hostCounterStoreRelease(
                m_view.systemScopeHostAtomics != 0u,
                &m_view.control->closed,
                std::uint32_t{1u});
        }

        bool isClosed() const
        {
            if(m_view.control == nullptr)
                return true;
            bool const systemScope = m_view.systemScopeHostAtomics != 0u;
            return detail::hostCounterLoadAcquire(systemScope, &m_view.control->closed) != 0u;
        }

        /** Debug counters plus the current modular positions.
         *
         * One call is not a consistent snapshot: the counters are loaded
         * independently, so `occupancy` can be a mix of two moments while a
         * producer or importer runs concurrently. Treat the values as
         * diagnostics. `importError` is sticky: it keeps the last recorded
         * importer error until `setCountersForTest` resets it, which is how the
         * host observes an error after the fact.
         */
        struct DebugCounters
        {
            std::uint64_t published{0u};
            std::uint64_t consumed{0u};
            std::uint64_t occupancy{0u};
            std::uint64_t totalEnqueued{0u};
            std::uint64_t totalImported{0u};
            std::uint64_t peakOccupancy{0u};
            bool closed{false};
            IngressError importError{IngressError::none};
        };

        DebugCounters debugCounters() const
        {
            DebugCounters counters;
            if(m_view.control == nullptr)
                return counters;
            bool const systemScope = m_view.systemScopeHostAtomics != 0u;
            counters.published = detail::hostCounterLoadAcquire(systemScope, &m_view.control->published);
            counters.consumed = detail::hostCounterLoadAcquire(systemScope, &m_view.control->consumed);
            counters.occupancy = counters.published - counters.consumed;
            counters.totalEnqueued = m_view.control->totalEnqueued;
            counters.totalImported = detail::hostCounterLoadAcquire(systemScope, &m_view.control->totalImported);
            counters.peakOccupancy = m_view.control->peakOccupancy;
            counters.closed = detail::hostCounterLoadAcquire(systemScope, &m_view.control->closed) != 0u;
            counters.importError
                = static_cast<IngressError>(detail::hostCounterLoadAcquire(systemScope, &m_view.control->importError));
            return counters;
        }

        std::uint64_t published() const
        {
            bool const systemScope = m_view.systemScopeHostAtomics != 0u;
            return detail::hostCounterLoadAcquire(systemScope, &m_view.control->published);
        }

        std::uint64_t consumed() const
        {
            bool const systemScope = m_view.systemScopeHostAtomics != 0u;
            return detail::hostCounterLoadAcquire(systemScope, &m_view.control->consumed);
        }

        std::uint64_t occupancy() const
        {
            return published() - consumed();
        }

        /** Test/diagnostic hook: pointer to the slot named by a modular counter.
         *
         * Not atomic. Do not dereference concurrently with an active producer or
         * importer unless the caller owns the corresponding side.
         */
        Descriptor const* descriptorSlot(std::uint64_t counter) const noexcept
        {
            return &m_view.slots[counter % capacity];
        }

        Descriptor* descriptorSlot(std::uint64_t counter) noexcept
        {
            return &m_view.slots[counter % capacity];
        }

        /** Test hook: overwrite both modular counters and reset diagnostics.
         *
         * The resulting occupancy must fit the capacity. Intended for rollover
         * tests that initialise the counters near the maximum value.
         */
        void setCountersForTest(std::uint64_t initialPublished, std::uint64_t initialConsumed)
        {
            if(m_view.control == nullptr)
                throw std::invalid_argument("graph: the ingress ring has no control block");
            std::uint64_t const occupancyNow = initialPublished - initialConsumed;
            if(occupancyNow > capacity)
                throw std::invalid_argument("graph: the initial ingress counters exceed the ring capacity");

            bool const systemScope = m_view.systemScopeHostAtomics != 0u;
            m_view.control->totalEnqueued = 0u;
            m_view.control->peakOccupancy = occupancyNow;
            detail::hostCounterStoreRelease(systemScope, &m_view.control->totalImported, std::uint64_t{0u});
            detail::hostCounterStoreRelease(systemScope, &m_view.control->importError, std::uint32_t{0u});
            detail::hostCounterStoreRelease(systemScope, &m_view.control->closed, std::uint32_t{0u});
            detail::hostCounterStoreRelease(systemScope, &m_view.control->consumed, initialConsumed);
            detail::hostCounterStoreRelease(systemScope, &m_view.control->published, initialPublished);
        }

    private:
        EnqueueResult makeClosed(EnqueueResult result) const noexcept
        {
            result.status = EnqueueStatus::closed;
            result.accepted = 0u;
            return result;
        }

        /** Shared body of `tryEnqueue` and `closeInput`. */
        EnqueueResult tryEnqueueInternal(std::span<Descriptor const> descriptors, bool allowEndMarker)
        {
            EnqueueResult result;
            bool ok = true;
            result.requested = detail::checkedU32(descriptors.size(), ok);
            if(!ok)
            {
                result.status = EnqueueStatus::error;
                result.error = IngressError::counter_overflow;
                return result;
            }

            for(Descriptor const& descriptor : descriptors)
            {
                if(!allowEndMarker && descriptor.kind == JobKind::END_OF_INPUT)
                {
                    result.status = EnqueueStatus::error;
                    result.error = IngressError::invalid_descriptor;
                    return result;
                }
            }

            bool const systemScope = m_view.systemScopeHostAtomics != 0u;
            std::uint64_t const head = detail::hostCounterLoadAcquire(systemScope, &m_view.control->published);
            std::uint64_t const tail = detail::hostCounterLoadAcquire(systemScope, &m_view.control->consumed);
            detail::hostCounterFenceAcquire(systemScope);

            std::uint64_t const used = head - tail;
            if(used > capacity)
            {
                result.status = EnqueueStatus::error;
                result.error = IngressError::counter_corrupt;
                return result;
            }

            std::uint64_t const free = static_cast<std::uint64_t>(capacity) - used;
            std::size_t const wanted = descriptors.size();
            std::size_t const accepted = wanted < free ? wanted : static_cast<std::size_t>(free);
            result.accepted = static_cast<std::uint32_t>(accepted);

            // Check the diagnostic counter before touching any slot so an
            // overflow failure leaves the published ring untouched.
            std::uint64_t const enqueuedBefore = m_view.control->totalEnqueued;
            if(accepted > 0u && enqueuedBefore > std::numeric_limits<std::uint64_t>::max() - accepted)
            {
                result.status = EnqueueStatus::error;
                result.error = IngressError::counter_overflow;
                return result;
            }

            for(std::size_t i = 0u; i < accepted; ++i)
                m_view.slots[(head + i) % capacity] = descriptors[i];

            if(accepted > 0u)
            {
                std::uint64_t const newHead = head + accepted;
                std::uint64_t const newOccupancy = used + accepted;

                m_view.control->totalEnqueued = enqueuedBefore + accepted;
                if(newOccupancy > m_view.control->peakOccupancy)
                    m_view.control->peakOccupancy = newOccupancy;

                detail::hostCounterStoreRelease(systemScope, &m_view.control->published, newHead);
            }

            result.status = (accepted == wanted) ? EnqueueStatus::accepted : EnqueueStatus::ring_full;
            return result;
        }

        View m_view{};
        /// Type-erased owning handle for the mapped allocation.
        std::shared_ptr<void> m_owner{};
    };
} // namespace graph
