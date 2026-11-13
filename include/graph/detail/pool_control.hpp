// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>
#include <graph/detail/device_atomic.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/launch.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/packet_pool.hpp>
#include <graph/ready_ring.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

/** @file
 *
 * Host-facing configuration, result vocabulary, and the device-side scheduler
 * state of the persistent warp pool.
 *
 * The header is shared by the host orchestration (`graph/pool.hpp`) and the
 * device scheduling loop (`graph/detail/scheduler.hpp`); it therefore contains
 * only plain value types and the small atomic helpers both sides need.
 *
 * # Two state blocks
 *
 * - `PoolDeviceState` lives in device memory. All hot counters are updated with
 *   device-scope atomics by the worker warps and copied to the host once, after
 *   the worker kernel is known to have stopped. No mapped-memory read-modify-write
 *   ever touches it.
 * - `PoolHostControl` lives in mapped pinned memory. It carries exactly two
 *   pieces of host<->device traffic: the host abort request (host release store,
 *   device acquire load) and the device progress publications (device release
 *   store, host acquire load). Both are pure load/store ordering; there is no
 *   RMW on mapped memory.
 */
namespace graph
{
    /** Compile-time capacities of the pool.
     *
     * The packet pool, ready ring, and ingress ring need compile-time
     * capacities because their storage layouts (free lists, slot stamps, modulo
     * arithmetic) are compile-time constants. The host configuration validates
     * the requested budget against these values and refuses an undersized pool
     * loudly instead of clamping.
     */
    inline constexpr std::size_t poolPacketSlotCapacity = 8192u;
    inline constexpr std::size_t poolPacketElementBytes = 4096u;
    inline constexpr std::size_t poolReadyCapacity = 8192u;
    inline constexpr std::uint32_t poolIngressCapacity = 256u;
    inline constexpr std::uint32_t poolStagingCapacity = 32u;
    inline constexpr std::uint32_t poolMaxWorkers = 4096u;

    /** Device error bits published in the pool error word. */
    enum class PoolError : std::uint32_t
    {
        none = 0u,
        /** A claimed job named an operation that has no device binding. */
        unknownOperation = 1u << 0u,
        /** `firstId + count` overflowed the 64-bit id range. */
        requestOverflow = 1u << 1u,
        /** A request left the configured id range. */
        idOutOfRange = 1u << 2u,
        /** The packet pool ran out of slots although the budget was validated. */
        packetExhausted = 1u << 3u,
        /** The ready ring hit its hard capacity; the reserved capacity failed. */
        readyRingFull = 1u << 4u,
        /** The startup rendezvous missed its deadline. */
        startupTimeout = 1u << 5u,
        /** The pool was stopped by a diagnostic abort (host or device). */
        aborted = 1u << 6u,
        /** No usable mapped-pinned ingress transport on this device. */
        transportUnavailable = 1u << 7u,
        /** The ready ring reported an ownership violation. */
        readyRingViolation = 1u << 8u,
        /** A scheduler invariant failed (should not happen). */
        internalState = 1u << 9u,
        /**
         * The launch geometry needs more workers than the pool can track.
         *
         * Every per-worker array in `PoolDeviceState` is sized by
         * `poolMaxWorkers`. A geometry above that capacity is refused with this
         * named error at validation, and the device kernel refuses to form an
         * out-of-range worker index and stops with the same bit. It is never
         * accepted and then indexed out of range.
         */
        workerCapacityExceeded = 1u << 10u,
        /**
         * A published packet's logical element count exceeded its capacity or
         * diverged from the committed plan.
         *
         * A consumer must never observe more elements than the producer wrote.
         * The pool records this bit at publication when the check fails, so a
         * count/length regression is a named, aborting failure instead of a
         * silent over-read.
         */
        packetExceedsCapacity = 1u << 11u
    };

    ALPAKA_FN_HOST_ACC constexpr std::uint32_t poolErrorBits(PoolError error) noexcept
    {
        return static_cast<std::uint32_t>(error);
    }

    ALPAKA_FN_HOST_ACC constexpr bool poolErrorSet(std::uint32_t word, PoolError error) noexcept
    {
        return (word & static_cast<std::uint32_t>(error)) != 0u;
    }

    /** Host configuration of one pool lifetime. */
    struct PoolConfig
    {
        /** Rank-1 element schema used for every pool-core packet. */
        ElementSchema packetSchema{};
        /** Number of elements per packet (`Y`). */
        std::uint32_t chunkCapacity{128u};
        /** Operation id used for host-submitted source requests. */
        OperationId sourceOperation{OperationId{1u}};
        /**
         * Packet-pool budget. A zero `workers` field is replaced with the
         * geometry's worker count at startup; every other field is taken as
         * given and checked by `requiredSlots()`.
         */
        PacketPoolBudget budget{};
        /** Maximum descriptors staged by one importer poll. Bounded by staging. */
        std::uint32_t maxImportBatch{16u};
        /** Maximum descriptors dispatched to the ready ring per loop iteration. */
        std::uint32_t importPerIteration{1u};
        /** Backoff (`__nanosleep`) applied by idle workers and the importer. */
        std::uint32_t idleNanosleep{128u};
        /**
         * Soft cap of ready entries created by the importer. Zero derives a
         * value from the packet and ready capacities. Internal successor
         * publications ignore this cap but never exceed the hard ring capacity;
         * the reserved capacity guarantees they cannot fill the ring.
         */
        std::uint32_t maxReadyInFlight{0u};
        /** Id-range check (`idLimit == 0` disables it). */
        std::uint32_t idLimit{0u};
        /**
         * Register hint used to size the launch wave. Zero selects the
         * conservative `fallbackKernelRegisters` value: the build-time capture
         * probes the rendezvous kernel, not the pool kernel, so reusing it would
         * overstate the resident wave (measured: the pool kernel is ~84
         * registers, which allows 20 blocks/SM, while the captured 20 suggests
         * 32). A conservative hint only lowers the worker count.
         */
        int registerHint{0};
        /** Startup rendezvous deadline in device cycles (CUDA). */
        std::uint64_t initDeadlineCycles{10'000'000'000u};
        /** Startup rendezvous deadline in spin iterations (CPU). */
        std::uint64_t initDeadlineIterations{50'000'000u};
        /**
         * Device pointer passed to every functor through the reserved control
         * port (`maxPacketPorts - 1`). The pool never dereferences it.
         */
        void* controlBlock{nullptr};
        /**
         * Test-only: cap the persistent worker warps (0 = use the full launch
         * geometry). Not part of the public API. It exists so the GPU
         * correctness ladder can force a reproducible worker count per rung;
         * the value only ever reduces the geometry, never raises it.
         */
        std::uint32_t testWorkerLimit{0u};
        /** Device buffer exposed through `copyResultsToHost`. */
        void* resultDevice{nullptr};
        /** Byte size of `resultDevice`. */
        std::uint64_t resultBytes{0u};
    };

    /** Host validation of @p config against the concrete launch geometry.
     *
     * Returns an empty string on success and a message naming the violated
     * contract otherwise. The check is deliberately available without a device
     * so the host-only test can exercise it.
     */
    ALPAKA_FN_HOST inline std::string validatePoolConfig(
        PoolConfig const& config,
        LaunchGeometry const& geometry,
        std::uint32_t packetSlots = static_cast<std::uint32_t>(poolPacketSlotCapacity),
        std::uint32_t packetBytes = static_cast<std::uint32_t>(poolPacketElementBytes))
    {
        if(!isValid(config.packetSchema) || config.packetSchema.rank == 0u)
            return "graph: pool packet schema must be a valid non-empty rank-1 schema";
        if(config.packetSchema.elementSize == 0u)
            return "graph: pool packet element size must be positive";
        if(config.chunkCapacity == 0u)
            return "graph: pool chunk capacity must be at least one element";
        if(config.chunkCapacity > config.packetSchema.extents[0])
            return "graph: pool chunk capacity exceeds the schema extent";
        if(static_cast<std::uint64_t>(config.chunkCapacity) * config.packetSchema.elementSize
           > static_cast<std::uint64_t>(packetBytes))
            return "graph: pool chunk capacity times element size exceeds the packet slot capacity in bytes";
        if(!config.sourceOperation.isValid())
            return "graph: pool source operation id is invalid";
        if(config.maxImportBatch == 0u || config.maxImportBatch > poolStagingCapacity)
            return "graph: pool import batch must be in [1, poolStagingCapacity]";
        if(config.importPerIteration == 0u)
            return "graph: pool importPerIteration must be at least one";
        if(config.idleNanosleep == 0u)
            return "graph: pool idleNanosleep must be positive";
        if(geometry.workerCount == 0u || geometry.warpsPerBlock != 1u)
            return "graph: pool requires the one-warp-per-block launch geometry";
        if(geometry.workerCount > poolMaxWorkers)
            return "graph: workerCapacityExceeded: geometry worker count " + std::to_string(geometry.workerCount)
                   + " exceeds the pool tracking capacity " + std::to_string(poolMaxWorkers);

        PacketPoolBudget budget = config.budget;
        if(budget.workers == 0u)
            budget.workers = geometry.workerCount;
        if(budget.workers > poolMaxWorkers)
            return "graph: workerCapacityExceeded: budget worker count " + std::to_string(budget.workers)
                   + " exceeds the pool tracking capacity " + std::to_string(poolMaxWorkers);
        std::uint64_t const required = requiredSlots(budget);
        if(required > static_cast<std::uint64_t>(packetSlots))
            return "graph: packet pool capacity (" + std::to_string(packetSlots)
                   + " slots) is undersized for the "
                     "budget formula P + 2B + W*R + S = "
                   + std::to_string(required);
        if(required == 0u)
            return "graph: packet pool budget is empty";

        // Internal publications need a reserved slice of the ready ring so they
        // cannot deadlock behind importer work, and every in-flight packet must
        // fit the packet pool.
        std::uint64_t const reserved = 3u * static_cast<std::uint64_t>(budget.workers);
        if(reserved >= static_cast<std::uint64_t>(packetSlots))
            return "graph: the geometry's worker count leaves no packet slots for the ready ring";
        std::uint64_t const derivedLimit = static_cast<std::uint64_t>(packetSlots) - reserved
                                           - static_cast<std::uint64_t>(poolStagingCapacity) - 64u;
        if(static_cast<std::uint64_t>(poolReadyCapacity) <= reserved)
            return "graph: ready ring capacity is too small for the reserved scheduler capacity";
        std::uint64_t const hardLimit = static_cast<std::uint64_t>(poolReadyCapacity) - reserved;
        std::uint64_t const limit = config.maxReadyInFlight == 0u
                                        ? (derivedLimit < hardLimit ? derivedLimit : hardLimit)
                                        : static_cast<std::uint64_t>(config.maxReadyInFlight);
        if(limit == 0u || limit > hardLimit)
            return "graph: maxReadyInFlight leaves no reserved ready-ring capacity";
        return {};
    }

    /** Distinct stop reasons reported by `Pool::wait()`. */
    enum class PoolStopReason : std::uint32_t
    {
        /** No kernel was ever started for this handle. */
        neverStarted = 0u,
        /** Ordered close, complete drain, and clean exit. */
        stoppedByDrain = 1u,
        /** Diagnostic abort (host or device) or a device error. */
        stoppedByAbort = 2u
    };

    /** Accounting of the scheduler state, readable after `wait()`. */
    struct QuiescenceSnapshot
    {
        std::uint64_t accepted{0u};
        std::uint64_t imported{0u};
        std::uint64_t queued{0u};
        std::uint64_t active{0u};
        std::uint64_t inFlightReservations{0u};
        std::uint64_t pendingSources{0u};
        std::uint64_t partialBuffers{0u};
        std::uint64_t publications{0u};
        std::uint64_t outstanding{0u};
        std::uint64_t completedJobs{0u};
        std::uint64_t claimAttempts{0u};
        std::uint64_t distinctWorkers{0u};
        std::uint64_t launchCount{0u};
        std::uint64_t readyViolations{0u};
        /** Jobs retired by a node without outgoing edges. */
        std::uint64_t retiredPackets{0u};
        /** Importer loop iterations that polled for ingress. */
        std::uint64_t importPolls{0u};
        /** Importer poll phases skipped because the ready ring/pool had no room. */
        std::uint64_t importBlockedPolls{0u};
        /** Importer polls that found no new descriptor. */
        std::uint64_t importEmptyPolls{0u};
        std::uint32_t errorWord{0u};
        std::uint32_t stopReason{0u};
        std::uint32_t initFailures{0u};
        std::uint32_t initWinner{0xffff'ffffu};
        std::uint64_t initMaxSpin{0u};
        std::uint64_t execCounts[8]{};
        std::uint32_t debugBlockThreads{0u};
        std::uint32_t debugWarpWidth{0u};
        std::uint32_t debugRequiredSlots{0u};
        std::uint32_t debugBudgetWorkers{0u};
        std::uint32_t debugConfigured{0u};
        std::uint32_t debugStopStage{0u};
        std::uint32_t initLockValue{0xffff'ffffu};
        std::uint32_t readyCapacity{0u};
        std::uint32_t packetCapacity{0u};
        std::uint32_t workerCount{0u};
        /** Blocks that took the initializer branch. Must be exactly one. */
        std::uint64_t initializerClaims{0u};
        /** Peak number of packet slots held at once (recycling diagnostic). */
        std::uint64_t peakHeldSlots{0u};
        /** Packet slots returned to the pool (a long run must recycle). */
        std::uint64_t slotsReturned{0u};
        /** Non-empty partial buffers released by the drain protocol. */
        std::uint64_t drainPublications{0u};
        /** Quiescence/drain rounds executed. */
        std::uint64_t drainRounds{0u};
    };

    /** Result of one whole pool lifetime. */
    struct PoolResult
    {
        PoolStopReason reason{PoolStopReason::neverStarted};
        std::uint32_t errorWord{0u};
        QuiescenceSnapshot counters{};

        bool stopped() const noexcept
        {
            return reason == PoolStopReason::stoppedByDrain || reason == PoolStopReason::stoppedByAbort;
        }
    };

    /** Status of one host submission batch. */
    enum class PoolEnqueueStatus : std::uint32_t
    {
        accepted = 0u,
        full,
        closed,
        error
    };

    /** Result of one host submission batch. */
    struct PoolEnqueueResult
    {
        PoolEnqueueStatus status{PoolEnqueueStatus::error};
        std::uint32_t accepted{0u};
        std::uint32_t requested{0u};

        bool allAccepted() const noexcept
        {
            return status == PoolEnqueueStatus::accepted;
        }
    };

    namespace detail
    {
        /** Cache-line aligned device scheduler state.
         *
         * Every field that is contended by different warps owns its own cache
         * line. The block is a plain trivially copyable aggregate; the host
         * zeroes one instance and copies it to device memory before launch.
         */
        struct alignas(128) PoolDeviceState
        {
            alignas(128) std::uint32_t initLock{0u};
            alignas(128) std::uint32_t initPad{0u};
            alignas(128) std::uint64_t initDone{0u};
            alignas(128) std::uint64_t launchCount{0u};
            alignas(128) std::uint64_t stopReason{0u};
            alignas(128) std::uint64_t closedSeen{0u};
            alignas(128) std::uint64_t outstanding{0u};
            alignas(128) std::uint64_t pendingSources{0u};
            alignas(128) std::uint64_t partialBuffers{0u};
            alignas(128) std::uint64_t inFlightReservations{0u};
            alignas(128) std::uint64_t active{0u};
            alignas(128) std::uint64_t claimAttempts{0u};
            alignas(128) std::uint64_t publications{0u};
            alignas(128) std::uint64_t importedJobs{0u};
            alignas(128) std::uint64_t completedJobs{0u};
            alignas(128) std::uint64_t publicationSeq{0u};
            alignas(128) std::uint64_t readyViolations{0u};
            alignas(128) std::uint32_t errorWord{0u};
            alignas(128) std::uint32_t stagingHead{0u};
            alignas(128) std::uint32_t stagingCount{0u};
            alignas(128) std::uint32_t initFailures{0u};
            alignas(128) std::uint32_t initWinner{0u};
            alignas(128) std::uint64_t initMaxSpin{0u};
            /**
             * Number of blocks that took the initializer branch. Exactly one
             * block may initialize the ring and the packet pool; a higher value
             * means the pool was re-configured underneath live owners.
             */
            alignas(128) std::uint64_t initializerClaims{0u};
            alignas(128) std::uint64_t execCounts[8]{};
            alignas(128) std::uint64_t retiredPackets{0u};
            alignas(128) std::uint64_t debugPolls{0u};
            alignas(128) std::uint64_t debugDispatches{0u};
            alignas(128) std::uint64_t debugImportBlocked{0u};
            alignas(128) std::uint64_t debugImportEmpty{0u};
            alignas(128) std::uint32_t blockPhase[poolMaxWorkers]{};
            alignas(128) std::uint64_t workerRetired[poolMaxWorkers]{};
            alignas(128) std::uint32_t debugBlockThreads{0u};
            alignas(128) std::uint32_t debugWarpWidth{0u};
            alignas(128) std::uint32_t debugRequiredSlots{0u};
            alignas(128) std::uint32_t debugBudgetWorkers{0u};
            alignas(128) std::uint32_t debugConfigured{0u};
            alignas(128) std::uint32_t debugStopStage{0u};
            alignas(128) std::uint64_t workerBits[poolMaxWorkers / 64u]{};
            /** Peak number of packet slots held at once (diagnostic). */
            alignas(128) std::uint64_t peakHeldSlots{0u};
            /** Number of packet slots returned to the pool during the run. */
            alignas(128) std::uint64_t slotsReturned{0u};
            /** Number of non-empty partial buffers released during a drain. */
            alignas(128) std::uint64_t drainPublications{0u};
            /** Quiescence/drain rounds executed (repeat-detection count). */
            alignas(128) std::uint64_t drainRounds{0u};
        };

        static_assert(std::is_trivially_copyable_v<PoolDeviceState>);
        static_assert(std::is_standard_layout_v<PoolDeviceState>);
        static_assert(alignof(PoolDeviceState) == 128u);

        /** Mapped host<->device control words.
         *
         * Only pure acquire loads and release stores touch these fields.
         */
        struct alignas(128) PoolHostControl
        {
            alignas(128) std::uint64_t abortRequest{0u};
            alignas(128) std::uint64_t imported{0u};
            alignas(128) std::uint64_t initDone{0u};
            alignas(128) std::uint64_t polls{0u};
            alignas(128) std::uint64_t dispatches{0u};
            alignas(128) std::uint32_t stopReason{0u};
            alignas(128) std::uint32_t reserved{0u};
            /**
             * Monotonic per-worker retirement count. Each worker publishes only
             * its own word with a release store, so the host-side sum is a
             * monotonic progress signal without any mapped-memory RMW.
             */
            alignas(128) std::uint64_t retiredPackets{0u};
            alignas(128) std::uint64_t debugPolls{0u};
            alignas(128) std::uint64_t debugDispatches{0u};
            alignas(128) std::uint32_t blockPhase[poolMaxWorkers]{};
            alignas(128) std::uint64_t workerRetired[poolMaxWorkers]{};
            alignas(128) std::uint32_t debugBlockThreads{0u};
            alignas(128) std::uint32_t debugWarpWidth{0u};
            alignas(128) std::uint32_t debugRequiredSlots{0u};
            alignas(128) std::uint32_t debugBudgetWorkers{0u};
            alignas(128) std::uint32_t debugConfigured{0u};
            alignas(128) std::uint32_t debugStopStage{0u};
        };

        static_assert(std::is_trivially_copyable_v<PoolHostControl>);
        static_assert(std::is_standard_layout_v<PoolHostControl>);
        static_assert(alignof(PoolHostControl) == 128u);

        /** Host mirror of `ReadyRing::RingHeader`.
         *
         * The host copies this header once after the worker kernel stopped to
         * report the in-flight ready-entry count. It is layout-identical to the
         * ring's header; the kernel asserts the size equality where the
         * accelerator type is known.
         */
        struct alignas(64) ReadyRingLayout
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

        static_assert(std::is_trivially_copyable_v<ReadyRingLayout>);
        static_assert(std::is_standard_layout_v<ReadyRingLayout>);

        /** Exact device storage sizes of the ring and pool, independent of the
         *  accelerator type (their layouts only depend on the payload types). */
        using HostReadyRing = ReadyRing<void*, poolReadyCapacity>;
        using HostPacketPool = PacketPool<void*, poolPacketSlotCapacity, poolPacketElementBytes>;
        inline constexpr std::size_t hostReadyStorageBytes = HostReadyRing::storageBytes;
        inline constexpr std::size_t hostPacketStorageBytes = HostPacketPool::storageBytes;

        /** Host mirror of `OperationTable<Acc>`.
         *
         * The host fills operation ids and schemas and leaves every function
         * pointer null; device code replaces them with device addresses. A
         * function pointer has the same size as `void*` on every supported
         * backend, which the kernel asserts together with the size equality.
         */
        struct OperationTableLayout
        {
            struct Entry
            {
                OperationId id{};
                ElementSchema inputSchema{};
                ElementSchema outputSchema{};
                void* function{nullptr};
            };

            Entry entries[PacketPoolTraits::maxOperations]{};
            std::uint32_t count{0u};
        };

        static_assert(std::is_trivially_copyable_v<OperationTableLayout>);
        static_assert(std::is_standard_layout_v<OperationTableLayout>);

        /** Reason values stored in `PoolDeviceState::stopReason`. */
        inline constexpr std::uint64_t poolStoppingNone = 0u;
        inline constexpr std::uint64_t poolStoppingDrained = 1u;
        inline constexpr std::uint64_t poolStoppingAborted = 2u;

        /** Device-scope atomic add. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline T addDevice(T_Acc const& acc, T* pointer, T value) noexcept
        {
            return alpaka::onAcc::atomicAdd(acc, pointer, value, alpaka::onAcc::scope::device);
        }

        /** Device-scope atomic exchange. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline T exchangeDevice(T_Acc const& acc, T* pointer, T value) noexcept
        {
            return alpaka::onAcc::atomicExch(acc, pointer, value, alpaka::onAcc::scope::device);
        }

        /** Device-scope atomic compare-exchange. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline T casDevice(T_Acc const& acc, T* pointer, T expected, T desired) noexcept
        {
            return alpaka::onAcc::atomicCas(acc, pointer, expected, desired, alpaka::onAcc::scope::device);
        }

        /** Device-scope atomic OR. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline T orDevice(T_Acc const& acc, T* pointer, T value) noexcept
        {
            return alpaka::onAcc::atomicOr(acc, pointer, value, alpaka::onAcc::scope::device);
        }

        /** Device pause used by idle workers and the importer.
         *
         * The measured wake-up cost on the HAL A30 is a 5.05 us round trip with
         * a busy spin; `__nanosleep(64..256)` is below the noise floor. The
         * CPU backend has no sleep intrinsic in this path, so it falls through
         * to an empty pause and the loop's iteration budget provides the
         * backoff.
         */
        ALPAKA_FN_ACC inline void devicePause(std::uint32_t nanos) noexcept
        {
#if defined(__CUDA_ARCH__)
            __nanosleep(nanos);
#else
            (void) nanos;
#endif
        }
    } // namespace detail
} // namespace graph
