// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/chunk_assembly.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/pool_control.hpp>
#include <graph/detail/rendezvous.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/detail/warp_handoff.hpp>
#include <graph/graph_model.hpp>
#include <graph/ingress_ring.hpp>
#include <graph/packet_pool.hpp>
#include <graph/ready_ring.hpp>

#include <chrono>
#include <cstdint>
#include <type_traits>

/** @file
 *
 * One persistent worker kernel and the scheduling loop it runs.
 *
 * # Worker shape
 *
 * One warp per block. Every lane of a worker executes the same uniform path:
 * the leader claims from the ready ring, the descriptor is broadcast, every
 * lane enters the same indirect `OperationFn` call, and the leader publishes the
 * successors and retires the input exactly once. There is no block barrier
 * inside the loop except the warp barrier that orders the leader's claim against
 * the lanes' packet reads and the lanes' writes against the leader's
 * publication. With one warp per block that barrier is `syncBlockThreads`,
 * which is the only sanctioned mapping in `warp_barrier.hpp`.
 *
 * # Work accounting (the acyclic stop rule)
 *
 * `outstanding` counts every job that exists: in the ready ring, executing on a
 * worker, staged by the importer, or held while it is being published. A job is
 * counted before it becomes visible and a parent is retired only after every
 * successor has been counted. `pendingSources` counts accepted source id ranges
 * whose last chunk has not executed yet. The acyclic stop rule is
 *
 *     closedSeen && pendingSources == 0 && outstanding == 0
 *         && partialBuffers == 0 && inFlightReservations == 0
 *
 * with a drained stop reason. No job can appear afterwards: the end marker is
 * imported only after every earlier descriptor, and a successor of a live job
 * is counted before its parent leaves `outstanding`.
 *
 * # Startup operation table
 *
 * The host publishes the operation ids and a placeholder table. The first warp
 * to take the startup lock binds every id to a **device** function pointer
 * (`T_Binder::bind`), publishes `initDone` with a device-scope release, and
 * every worker waits for it with an acquire load inside a deadline-bounded
 * rendezvous. A host function address is never stored in the table.
 *
 * # Importer
 *
 * Exactly one block (block zero) imports. Its leader polls the mapped ingress
 * ring at a bounded cadence, imports at most `maxImportBatch` descriptors only
 * while the ready ring is below the soft limit, resolves operation ids through
 * the device table, acquires one packet per source job, and dispatches at most
 * `importPerIteration` jobs per loop iteration. The block therefore gives
 * ingress a bounded opportunity even while ready work exists.
 */
namespace graph
{
    namespace detail
    {
        /** All arguments of the persistent launch.
         *
         * A single trivially copyable value; no host object is referenced from
         * device code.
         */
        struct PoolKernelArgs
        {
            void* state{nullptr};
            void* readyStorage{nullptr};
            void* packetStorage{nullptr};
            void* operationTable{nullptr};
            void* graph{nullptr};
            PoolHostControl* hostControl{nullptr};
            IngressControl* ingressControl{nullptr};
            JobDescriptor* ingressSlots{nullptr};
            JobDescriptor* staging{nullptr};
            std::uint32_t ingressSlotCount{0u};
            std::uint32_t ingressSystemScope{0u};
            std::uint32_t stagingCapacity{0u};
            std::uint32_t maxImportBatch{0u};
            std::uint32_t importPerIteration{0u};
            std::uint32_t idleNanosleep{0u};
            std::uint32_t chunkCapacity{0u};
            std::uint32_t readySoftLimit{0u};
            std::uint32_t workers{0u};
            std::uint32_t idLimit{0u};
            std::uint64_t initDeadlineCycles{0u};
            std::uint64_t initDeadlineIterations{0u};
            ElementSchema packetSchema{};
            PacketPoolBudget budget{};
            void* controlBlock{nullptr};
            // Real per-destination chunk assembly (design section 6). The
            // assembler state (stream table + chunk arena) is preallocated by the
            // host, copied to device storage, and owned by the pool for its
            // lifetime. A job that assembles chunks routes its outputs through
            // the reservation/commit protocol against this state.
            void* assemblyStorage{nullptr};
            std::uint32_t assemblyStreamCount{0u};
            std::uint32_t assemblySlotCount{0u};
        };

        static_assert(std::is_trivially_copyable_v<PoolKernelArgs>);
        static_assert(std::is_standard_layout_v<PoolKernelArgs>);

        /** One destination stream's state, as the scheduler's assembler sees it.
         *
         * This is `chunk_assembly.hpp`'s `AssemblyStream<T>` with the element
         * type erased: the element type is fixed for one pool lifetime by the
         * schema (`ElementSchema::elementSize`), so the scheduler can drive the
         * two-chunk layout and rotation through `elementSize` bytes without a
         * C++ type. The layout is byte-identical to `AssemblyStream<T>` for a
         * rank-1 contiguous schema, which the host validates before launch.
         */
        struct AssemblyStreamState
        {
            StreamId id{};
            std::uint32_t capacity{0u};
            std::uint32_t count{0u};
            std::uint32_t firstSlot{invalidChunkSlot};
            std::uint32_t firstHandleIndex{0u};
            std::uint32_t firstHandleGeneration{0u};
            std::uint32_t secondSlot{invalidChunkSlot};
            std::uint32_t secondHandleIndex{0u};
            std::uint32_t secondHandleGeneration{0u};
            std::uint32_t firstCount{0u};
            std::uint32_t claim{0u};
            std::uint32_t version{0u};
            bool drained{false};
            bool valid{false};
        };

        static_assert(std::is_trivially_copyable_v<AssemblyStreamState>);
        static_assert(std::is_standard_layout_v<AssemblyStreamState>);
        // Byte-identical to AssemblyStream<T> for a rank-1 contiguous schema.
        static_assert(
            sizeof(AssemblyStreamState) == sizeof(AssemblyStream<std::int8_t>),
            "graph: the erased assembly state must match AssemblyStream<T>");

        /** Device-side view over the pool's assembly storage.
         *
         * Storage layout: `[AssemblyStreamState streams][N * elementSize chunk
         * arena]`, where `N` is the number of physical chunk slots (arena slots).
         * `find()` is a linear scan over the stream ids (the stream count is
         * small; a graph has at most `GraphLimits::maxStreams` streams).
         */
        template<typename T_Acc>
        struct AssemblyView
        {
            Acc const& acc;
            AssemblyStreamState* streams{nullptr};
            std::uint32_t streamCount{0u};
            std::uint8_t* arena{nullptr};
            std::uint32_t elementSize{0u};
            std::uint32_t chunkCapacity{0u};
            std::uint32_t slotCount{0u};

            ALPAKA_FN_ACC constexpr std::uint8_t* chunkData(std::uint32_t slot) const noexcept
            {
                return arena + static_cast<std::uint64_t>(slot) * static_cast<std::uint64_t>(chunkCapacity)
                       * static_cast<std::uint64_t>(elementSize);
            }

            ALPAKA_FN_ACC constexpr std::uint32_t find(StreamId id) const noexcept
            {
                if(!id.isValid())
                    return invalidChunkSlot;
                for(std::uint32_t i = 0u; i < streamCount; ++i)
                    if(streams[i].valid && streams[i].id == id)
                        return i;
                return invalidChunkSlot;
            }
        };

        /** Outcome of planning one input job's assembly. */
        enum class PlanStatus : std::uint32_t
        {
            Ok = 0u,
            None = 1u,
            StreamBusy = 2u,
            StreamDrained = 3u,
            OccupancyOverflow = 4u,
            NotEnoughReplacements = 5u,
            NoCapacity = 6u
        };

        static_assert(std::is_trivially_copyable_v<PlanStatus>);

        /** One planned destination: the stream and the committed append. */
        struct PlanTarget
        {
            std::uint32_t streamIndex{0u};
            std::uint32_t itemCount{0u};
            std::uint32_t baseCount{0u};
            bool rotate{false};
            std::uint32_t replacement{invalidChunkSlot};
        };

        static_assert(std::is_trivially_copyable_v<PlanTarget>);

        /** Plan of one input job's output routing (the scheduler's side of the
         *  reservation protocol). `targetCount == 0` means the job produces no
         *  assembled output (a source, or a node without an outgoing edge). */
        struct Plan
        {
            PlanStatus status{PlanStatus::None};
            std::uint32_t targetCount{0u};
            PlanTarget targets[maxAssemblyTargets]{};
        };

        static_assert(std::is_trivially_copyable_v<Plan>);

        template<typename T_Acc>
        struct SchedulerRuntime
        {
            using Acc = std::remove_cvref_t<T_Acc>;
            using Ring = ReadyRing<Acc, poolReadyCapacity>;
            using Pool = PacketPool<Acc, poolPacketSlotCapacity, poolPacketElementBytes>;
            using Table = OperationTable<Acc>;

            Acc const& acc;
            WarpGroup<Acc> const& warp;
            WarpBarrier<Acc> const& barrier;
            Ring& ring;
            Pool& pool;
            Table& table;
            CompiledGraph const& graph;
            PoolDeviceState& state;
            PoolHostControl& host;
            PoolKernelArgs const& args;
            /**
             * Threads per block of this launch. The block index is *not* a
             * member: it is passed by value into the functions that need it so
             * every use reads the value the kernel derived from the
             * accelerator, never a copy carried inside this aggregate.
             */
            std::uint32_t blockThreads;
        };

        /** Find the compiled node that owns an operation id.
         *
         * The scheduler uses one node per operation id (the node that produced
         * the job). Two nodes must not share an id in a pool-core graph.
         */
        ALPAKA_FN_HOST_ACC inline CompiledNode const* findNode(CompiledGraph const& graph, OperationId id) noexcept
        {
            for(std::uint32_t index = 0u; index < graph.nodeCount; ++index)
            {
                if(graph.nodes[index].operation == id)
                    return &graph.nodes[index];
            }
            return nullptr;
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC inline void recordError(SchedulerRuntime<T_Acc> const& runtime, PoolError error) noexcept
        {
            orDevice(runtime.acc, &runtime.state.errorWord, poolErrorBits(error));
        }

        /** Leader-published state of the importer's dispatch decision.
         *
         * Distributed through `WarpHandoff`, not a shuffle: the decision
         * follows a leader-only read of the staging state, and the pinned
         * alpaka's shuffle does not reconverge a warp that a leader-only region
         * split.
         */
        struct ImportDispatch
        {
            std::uint32_t stagingCount{0u};
            std::uint32_t stagingHead{0u};
            std::uint32_t hasRoom{0u};
        };

        static_assert(std::is_trivially_copyable_v<ImportDispatch>);
        static_assert(std::is_standard_layout_v<ImportDispatch>);

        /** Leader-published outcome of one ingress poll. */
        struct ImportPoll
        {
            ImportStatus status{ImportStatus::nothing_new};
            std::uint32_t imported{0u};
        };

        static_assert(std::is_trivially_copyable_v<ImportPoll>);
        static_assert(std::is_standard_layout_v<ImportPoll>);

        /** Leader-published outcome of the startup configuration. */
        struct StartupState
        {
            std::uint32_t configured{1u};
            std::uint32_t bound{1u};
            std::uint32_t ready{1u};
        };

        static_assert(std::is_trivially_copyable_v<StartupState>);
        static_assert(std::is_standard_layout_v<StartupState>);

        /** Overwrite the packet-pool record of a freshly acquired slot.
         *
         * `PacketPool::acquire()` documents that the caller fills in the
         * schema and element count; the pool exposes the record read-only, so
         * the scheduler writes it through a cast. The slot is exclusively owned
         * by the caller at this point.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void configureRecord(
            SchedulerRuntime<T_Acc>& runtime,
            PacketHandle const& handle,
            std::uint32_t elementCount) noexcept
        {
            PacketRecord& record = const_cast<PacketRecord&>(runtime.pool.slotRecord(handle));
            record.handle = handle;
            record.schema = runtime.args.packetSchema;
            record.capacity = runtime.args.chunkCapacity;
            record.elementCount = elementCount;
        }

        /** Build the ready entry for a successor job.
         *
         * Deterministic on every lane, so the ring's warp-uniform `enqueue`
         * receives identical values without a second broadcast.
         */
        ALPAKA_FN_ACC inline JobDescriptor makeSuccessor(
            CompiledEdge const& edge,
            CompiledNode const& destination,
            JobDescriptor const& parent) noexcept
        {
            JobDescriptor next{};
            next.operation = destination.operation;
            next.kind = JobKind::OPERATION;
            next.packet = parent.packet;
            next.routing.inputPort = edge.inputPort;
            next.routing.outputPort = 0u;
            next.routing.outputPortCount = destination.outputPortCount;
            next.request = RequestSpec{};
            next.sequenceNumber = parent.sequenceNumber + 1u;
            return next;
        }

        /** True when the mapped host abort word is set. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool hostAbortRequested(SchedulerRuntime<T_Acc> const& runtime) noexcept
        {
            return graph::systemLoadAcquire(runtime.acc, &runtime.host.abortRequest) != 0u;
        }

        /** Set the stop flag once and report the reason. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void requestStop(
            SchedulerRuntime<T_Acc>& runtime,
            std::uint64_t reason,
            std::uint32_t stage = 0u) noexcept
        {
            if(runtime.warp.isLeader())
            {
                if(stage != 0u)
                    runtime.state.debugStopStage = stage;
                if(reason == poolStoppingAborted)
                    recordError(runtime, PoolError::aborted);
            }
            (void) exchangeDevice(runtime.acc, &runtime.state.stopReason, reason);
        }

        /** Acquire a packet slot, retrying a transient pool-lock contention.
         *
         * `PacketPool::acquire()` reports both "no free slot" and "locked" as
         * `false`. Under the validated budget the free list cannot be empty for
         * a scheduler acquisition, so a bounded retry separates the two: an
         * exhausted retry budget is a hard resource failure.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool acquireWithRetry(SchedulerRuntime<T_Acc>& runtime, PacketHandle& handle) noexcept
        {
            constexpr std::uint32_t maxAttempts = 1u << 20u;
            for(std::uint32_t attempt = 0u; attempt < maxAttempts; ++attempt)
            {
                if(runtime.pool.acquire(runtime.acc, runtime.warp, handle))
                    return true;
                devicePause(runtime.args.idleNanosleep);
            }
            return false;
        }

        /** True when the handle broadcast with a ready entry still names a
         *  slot the pool considers owned: the slot is active and its
         *  generation matches the handle's.
         *
         * The handle is the contract of the ready entry (its index and
         * generation were broadcast with the descriptor). A mismatch means the
         * slot was recycled or the descriptor is stale, which is a protocol
         * violation: it is recorded as an ownership violation and the job is
         * dropped rather than executed against a packet it does not own. The
         * check runs on every lane (warp-uniform), so the failure is
         * deterministic for the warp instead of a lane-by-lane data race.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool packetHandleValid(
            SchedulerRuntime<T_Acc>& runtime,
            PacketHandle const& handle) noexcept
        {
            if(handle.index >= static_cast<std::uint32_t>(poolPacketSlotCapacity))
            {
                if(runtime.warp.isLeader())
                    recordError(runtime, PoolError::internalState);
                return false;
            }
            // Ownership is carried by the generation: the slot is owned by this
            // handle iff the pool's current generation for that index equals the
            // generation carried by the descriptor. `elementCount` is not an
            // ownership signal: under a deliberately small pool a valid in-flight
            // job can be claimed while its payload is not yet fully configured,
            // so a zero elementCount must not drop the job. The read is an
            // acquire load because another block's claim published the record.
            bool const matches = runtime.pool.observedGeneration(runtime.acc, handle.index) == handle.generation;
            if(!matches && runtime.warp.isLeader())
                (void) addDevice(runtime.acc, &runtime.state.readyViolations, std::uint64_t{1u});
            return matches;
        }

        /** Publish one ready entry, retrying a transient lock contention.
         *
         * The ring lock is held only for a short critical section and is never
         * held across a collective, so `Contended` is always retryable: the
         * warp that owns the lock finishes and releases it. Only `Full` is a
         * hard capacity failure.
         *
         * The attempt budget must cover a whole launch wave publishing at once
         * (~900 warps on the A30). A smaller budget (4096) was exhausted under
         * that contention and the caller reported a capacity error for a ring
         * that was almost empty, which aborted the pool. The backoff runs every
         * `pauseInterval` failures: pausing after *every* failure re-synchronizes
         * the fleet into the same thundering herd that starved it.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline RingStatus enqueueWithRetry(
            SchedulerRuntime<T_Acc>& runtime,
            JobDescriptor const& entry) noexcept
        {
            constexpr std::uint32_t maxAttempts = 1u << 20u;
            constexpr std::uint32_t pauseInterval = 64u;
            RingStatus status = RingStatus::Contended;
            for(std::uint32_t attempt = 0u; attempt < maxAttempts; ++attempt)
            {
                status = runtime.ring.enqueue(runtime.acc, runtime.warp, entry);
                if(status != RingStatus::Contended)
                    return status;
                if(attempt % pauseInterval == 0u)
                    devicePause(runtime.args.idleNanosleep);
            }
            return status;
        }

        /** Error bit for a failed ready-ring publication.
         *
         * `Full` is the reserved-capacity failure the budget is sized against;
         * an exhausted `Contended` budget is a scheduler livelock, which is a
         * different, louder contract violation. Reporting contention as
         * `readyRingFull` hides it behind a capacity error that never happened.
         */
        ALPAKA_FN_ACC inline std::uint32_t publicationError(RingStatus status) noexcept
        {
            return poolErrorBits(status == RingStatus::Full ? PoolError::readyRingFull : PoolError::internalState);
        }

        /** Acyclic quiescence check. Every worker may run it; the first
         *  observer publishes the drained stop flag. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void tryQuiesce(SchedulerRuntime<T_Acc>& runtime) noexcept
        {
            if(!runtime.warp.isLeader())
                return;
            std::uint64_t const closed = loadDeviceAcquire(runtime.acc, &runtime.state.closedSeen);
            if(closed == 0u)
                return;
            std::uint64_t const pendingSources = loadDeviceRelaxed(runtime.acc, &runtime.state.pendingSources);
            if(pendingSources != 0u)
                return;
            std::uint64_t const outstanding = loadDeviceRelaxed(runtime.acc, &runtime.state.outstanding);
            if(outstanding != 0u)
                return;
            if(loadDeviceRelaxed(runtime.acc, &runtime.state.partialBuffers) != 0u)
                return;
            if(loadDeviceRelaxed(runtime.acc, &runtime.state.inFlightReservations) != 0u)
                return;
            if(loadDeviceRelaxed(runtime.acc, &runtime.state.stagingCount) != 0u)
                return;
            std::uint64_t const violations = runtime.ring.ownershipViolations();
            if(violations != 0u)
            {
                runtime.state.readyViolations = violations;
                recordError(runtime, PoolError::readyRingViolation);
                requestStop(runtime, poolStoppingAborted);
                return;
            }
            (void) exchangeDevice(runtime.acc, &runtime.state.stopReason, poolStoppingDrained);
        }

        /** Bind the device operation table and publish the init flag.
         *
         * The whole routine is warp-uniform: only the leader takes the startup
         * lock, and every control value is broadcast back to the lanes. Error
         * bits are written by the leader only, so no lane can diverge on a
         * shared word.
         */
        template<typename T_Binder, typename T_Acc>
        ALPAKA_FN_ACC inline bool startupRendezvous(
            SchedulerRuntime<T_Acc>& runtime,
            std::uint32_t blockIndex,
            std::uint64_t deadlineCycles,
            std::uint64_t /*deadlineIterations*/) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            // Exactly one block initializes the ready ring and the packet pool.
            // The role is decided from the block index the kernel derived from
            // the accelerator and passes in *by value*; it must never be read
            // back out of the `SchedulerRuntime` aggregate here. When the
            // election is wrong every block runs `configure()`, whose per-slot
            // record rewrite clobbers a packet record another block already
            // handed out (the `{next, 0}` generation-0 abort) and relinks the
            // free list underneath live owners. `initializerClaims` is a hard
            // invariant: exactly one block may take this branch, and the
            // ladder asserts it on every rung so a regression cannot hide as a
            // silent re-configuration.
            bool const winner = blockIndex == 0u;
            if(winner && runtime.warp.isLeader())
            {
                runtime.state.initLock = 1u;
                (void) addDevice(runtime.acc, &runtime.state.initializerClaims, std::uint64_t{1u});
            }

            bool ready = true;
            if(winner)
            {
                runtime.ring.initialize(runtime.acc, runtime.warp);
                if(runtime.warp.isLeader())
                {
                    runtime.state.debugRequiredSlots = requiredSlots(runtime.args.budget);
                    runtime.state.debugBudgetWorkers = runtime.args.budget.workers;
                }
                bool const configured = runtime.pool.configure(runtime.acc, runtime.warp, runtime.args.budget);
                // The pool configure already distributed `configured` through
                // its own handoff; the remaining startup flags follow through a
                // second handoff so no lane depends on a shuffle.
                WarpHandoff<Acc, StartupState, handoff_id::schedulerStartup> handoff;
                StartupState local{};
                if(runtime.warp.isLeader())
                {
                    runtime.state.debugConfigured = configured ? 1u : 0u;
                    // One initializer writes the launch marker exactly once.
                    runtime.state.launchCount = 1u;
                    runtime.state.initWinner = blockIndex;
                    local.configured = configured ? 1u : 0u;
                    local.bound = T_Binder::bind(runtime.acc, runtime.table) ? 1u : 0u;
                    local.ready = (local.configured != 0u && local.bound != 0u) ? 1u : 0u;
                    if(local.configured == 0u)
                        recordError(runtime, PoolError::internalState);
                    if(local.bound == 0u)
                        recordError(runtime, PoolError::unknownOperation);
                }
                handoff.publish(runtime.acc, runtime.warp, local);
                handoff.sync(runtime.acc);
                ready = handoff.get(runtime.acc).ready != 0u;
                // Publish every table and ring store before any worker reads.
                // A device-scope release fence followed by a volatile store is
                // the publication pattern validated by the ready-ring and
                // chunk-assembly cross-warp tests.
                alpaka::onAcc::memFence(runtime.acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
                if(runtime.warp.isLeader())
                {
                    *reinterpret_cast<std::uint64_t volatile*>(&runtime.state.initDone) = 1u;
                    graph::systemStoreRelease(runtime.acc, &runtime.host.initDone, std::uint64_t{1u});
                }
            }

            // Wait warp-uniformly: every lane of the warp polls the flag, so
            // the leader-only publication section is never followed by a
            // collective from a divergent active mask.
            bool observed = winner;
            if(!winner)
            {
                using Api = ALPAKA_TYPEOF(runtime.acc.getApi());
                constexpr bool isCuda = std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Cuda>;
                (void) deadlineCycles;
                auto* initFlag = reinterpret_cast<std::uint64_t volatile*>(&runtime.state.initDone);
                // The host deadline must never be *formed* on the device: a
                // `__host__`-only clock read inside a device function is
                // illegal, and the pinned toolchain only warns about it. Every
                // non-initializing block takes this path, so it has to be
                // device-legal, not merely unreached by the initializer.
                std::chrono::steady_clock::time_point cpuDeadline{};
                (void) cpuDeadline;
#if !defined(__CUDA_ARCH__)
                if constexpr(!isCuda)
                    cpuDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{5000};
#endif
                constexpr std::uint64_t maxSpinIterations = isCuda ? 20'000'000u : 200'000'000u;
                std::uint64_t iterations = 0u;
                while(*initFlag == 0u)
                {
                    if(++iterations > maxSpinIterations)
                        break;
#if !defined(__CUDA_ARCH__)
                    if constexpr(!isCuda)
                    {
                        if(std::chrono::steady_clock::now() > cpuDeadline)
                            break;
                    }
#endif
                    devicePause(runtime.args.idleNanosleep);
                }
                runtime.barrier.barrier(runtime.acc, runtime.blockThreads, runtime.warp);
                alpaka::onAcc::memFence(runtime.acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::acquire);
                observed = *initFlag != 0u;
                if(runtime.warp.isLeader())
                    runtime.state.initMaxSpin
                        = iterations > runtime.state.initMaxSpin ? iterations : runtime.state.initMaxSpin;
            }

            // `ready` and `observed` are warp-uniform values; compute the fatal
            // decision without another shuffle.
            bool const fatal = !observed || !ready;
            if(fatal)
            {
                if(runtime.warp.isLeader())
                {
                    (void) addDevice(runtime.acc, &runtime.state.initFailures, std::uint32_t{1u});
                    if(!observed)
                    {
                        recordError(runtime, PoolError::startupTimeout);
                        runtime.state.debugStopStage = 5u;
                    }
                    requestStop(runtime, poolStoppingAborted, runtime.state.debugStopStage);
                }
                return false;
            }
            return true;
        }

        /** Give ingress a bounded opportunity and dispatch already staged jobs.
         *
         * Every decision the other lanes need is published by the importer
         * leader through a block-shared handoff and read after a block barrier.
         * There is **no** leader-only `return` in this function: a lane that
         * returned while the rest of the warp still had to reach a barrier
         * desynchronized the block's barrier accounting and stalled the
         * importer after a few jobs.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void importerStep(SchedulerRuntime<T_Acc>& runtime) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            using DispatchHandoff = WarpHandoff<Acc, ImportDispatch, handoff_id::schedulerImport>;
            using PollHandoff = WarpHandoff<Acc, ImportPoll, handoff_id::schedulerImport + 1u>;

            if(runtime.warp.isLeader())
            {
                std::uint64_t const polls = addDevice(runtime.acc, &runtime.state.debugPolls, std::uint64_t{1u}) + 1u;
                graph::systemStoreRelease(runtime.acc, &runtime.host.polls, polls);
            }

            // Dispatch staged descriptors first: the importer acknowledged them
            // already, so they are accounted as outstanding.
            for(std::uint32_t dispatched = 0u; dispatched < runtime.args.importPerIteration; ++dispatched)
            {
                ImportDispatch local{};
                if(runtime.warp.isLeader())
                {
                    local.stagingCount = runtime.state.stagingCount;
                    local.stagingHead = runtime.state.stagingHead;
                    local.hasRoom = (runtime.ring.inFlightCount() < runtime.args.readySoftLimit
                                     && runtime.pool.freeCount() > runtime.args.workers)
                                        ? 1u
                                        : 0u;
                }
                DispatchHandoff handoff;
                handoff.publish(runtime.acc, runtime.warp, local);
                handoff.sync(runtime.acc);
                ImportDispatch const state = handoff.get(runtime.acc);
                if(state.stagingCount == 0u || state.hasRoom == 0u)
                    break;

                // Only the importer leader mutates the staging area, and it is
                // not writing during the dispatch phase, so the descriptor is
                // stable while every lane reads it.
                JobDescriptor const staged = runtime.args.staging[state.stagingHead];
                PacketHandle handle{};
                if(!acquireWithRetry(runtime, handle))
                {
                    if(runtime.warp.isLeader())
                    {
                        recordError(runtime, PoolError::packetExhausted);
                        requestStop(runtime, poolStoppingAborted);
                    }
                    return;
                }
                if(runtime.warp.isLeader())
                {
                    std::uint32_t const emitted = staged.request.count < runtime.args.chunkCapacity
                                                      ? static_cast<std::uint32_t>(staged.request.count)
                                                      : runtime.args.chunkCapacity;
                    configureRecord(runtime, handle, emitted);
                    // The importer already counted this job as outstanding.
                    (void) addDevice(runtime.acc, &runtime.state.publications, std::uint64_t{1u});
                    std::uint64_t const dispatches
                        = addDevice(runtime.acc, &runtime.state.debugDispatches, std::uint64_t{1u}) + 1u;
                    graph::systemStoreRelease(runtime.acc, &runtime.host.dispatches, dispatches);
                }
                JobDescriptor job = staged;
                job.packet = handle;
                RingStatus const status = enqueueWithRetry(runtime, job);
                if(status != RingStatus::Accepted)
                {
                    if(runtime.warp.isLeader())
                    {
                        orDevice(runtime.acc, &runtime.state.errorWord, publicationError(status));
                        requestStop(runtime, poolStoppingAborted);
                    }
                    return;
                }
                if(runtime.warp.isLeader())
                {
                    if(runtime.state.stagingHead + 1u >= runtime.state.stagingCount)
                    {
                        runtime.state.stagingHead = 0u;
                        runtime.state.stagingCount = 0u;
                    }
                    else
                    {
                        runtime.state.stagingHead += 1u;
                    }
                }
            }

            // Re-publish the staging state and the room decision for the poll
            // phase; the barrier also orders the leader's staging writes above
            // against any lane that reads staging in the next iteration.
            {
                ImportDispatch local{};
                if(runtime.warp.isLeader())
                {
                    local.stagingCount = runtime.state.stagingCount;
                    local.hasRoom = (runtime.ring.inFlightCount() < runtime.args.readySoftLimit
                                     && runtime.pool.freeCount() > runtime.args.workers)
                                        ? 1u
                                        : 0u;
                }
                DispatchHandoff handoff;
                handoff.publish(runtime.acc, runtime.warp, local);
                handoff.sync(runtime.acc);
                ImportDispatch const state = handoff.get(runtime.acc);
                if(state.stagingCount != 0u)
                    return;
                if(state.hasRoom == 0u)
                {
                    if(runtime.warp.isLeader())
                        (void) addDevice(runtime.acc, &runtime.state.debugImportBlocked, std::uint64_t{1u});
                    return;
                }
            }

            // One importer lane polls; the resolver maps ids to device
            // pointers. Every exit decision is published and taken uniformly.
            ImportPoll poll{};
            if(runtime.warp.isLeader())
            {
                IngressRing<JobDescriptor, poolIngressCapacity>::View view{};
                view.control = runtime.args.ingressControl;
                view.slots = runtime.args.ingressSlots;
                view.slotCount = runtime.args.ingressSlotCount;
                view.systemScopeHostAtomics = runtime.args.ingressSystemScope;
                auto resolver = [&runtime](JobDescriptor const& descriptor) noexcept
                {
                    if(descriptor.kind == JobKind::END_OF_INPUT)
                        return true;
                    return runtime.table.resolve(descriptor.operation) != nullptr;
                };
                ImportBatch const batch = view.pollAndImport(
                    runtime.acc,
                    runtime.args.staging,
                    runtime.args.stagingCapacity,
                    runtime.args.maxImportBatch,
                    resolver);
                poll.status = batch.status;
                poll.imported = batch.imported;
                if(batch.status != ImportStatus::error && batch.imported != 0u)
                {
                    std::uint32_t jobs = 0u;
                    for(std::uint32_t index = 0u; index < batch.imported; ++index)
                    {
                        JobDescriptor const& descriptor = runtime.args.staging[index];
                        if(descriptor.kind == JobKind::END_OF_INPUT)
                            continue;
                        ++jobs;
                        bool const overflow
                            = descriptor.request.count > ~std::uint64_t{0u} - descriptor.request.firstId;
                        if(overflow)
                            recordError(runtime, PoolError::requestOverflow);
                        if(runtime.args.idLimit != 0u && !overflow
                           && descriptor.request.firstId + descriptor.request.count
                                  > static_cast<std::uint64_t>(runtime.args.idLimit))
                            recordError(runtime, PoolError::idOutOfRange);
                    }
                    if(loadDeviceRelaxed(runtime.acc, &runtime.state.errorWord) != 0u)
                    {
                        poll.status = ImportStatus::error;
                    }
                    else
                    {
                        runtime.state.stagingHead = 0u;
                        runtime.state.stagingCount = jobs;
                        (void) addDevice(runtime.acc, &runtime.state.importedJobs, static_cast<std::uint64_t>(jobs));
                        (void) addDevice(runtime.acc, &runtime.state.pendingSources, static_cast<std::uint64_t>(jobs));
                        (void) addDevice(runtime.acc, &runtime.state.outstanding, static_cast<std::uint64_t>(jobs));
                        if(batch.endMarkerSeen)
                            storeDeviceRelease(runtime.acc, &runtime.state.closedSeen, std::uint64_t{1u});
                        std::uint64_t const imported = loadDeviceRelaxed(runtime.acc, &runtime.state.importedJobs);
                        graph::systemStoreRelease(runtime.acc, &runtime.host.imported, imported);
                    }
                }
            }
            PollHandoff handoff;
            handoff.publish(runtime.acc, runtime.warp, poll);
            handoff.sync(runtime.acc);
            ImportPoll const result = handoff.get(runtime.acc);
            if(result.imported == 0u)
            {
                if(runtime.warp.isLeader())
                    (void) addDevice(runtime.acc, &runtime.state.debugImportEmpty, std::uint64_t{1u});
            }
            if(result.status == ImportStatus::error)
            {
                if(runtime.warp.isLeader())
                {
                    recordError(runtime, PoolError::unknownOperation);
                    requestStop(runtime, poolStoppingAborted);
                }
                return;
            }
            if(result.imported == 0u)
                return;
        }

        /** The owner token a plan's claims were taken with.
         *
         * Derived from the input packet handle (index xor generation) so it is
         * warp-uniform, nonzero, and unique among concurrently in-flight writer
         * warps. A released stream has `claim == 0`, which no token equals.
         */
        ALPAKA_FN_ACC inline std::uint32_t ownerTokenOf(PacketHandle const& handle) noexcept
        {
            std::uint32_t const token = (handle.index ^ (handle.generation + 1u)) + 1u;
            return token == 0u ? 1u : token;
        }

        /** Build the assembly view over the pool's preallocated storage. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline AssemblyView<typename SchedulerRuntime<T_Acc>::Acc> makeAssemblyView(
            SchedulerRuntime<T_Acc> const& runtime) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            AssemblyView<Acc> view{};
            view.acc = runtime.acc;
            view.elementSize = runtime.args.packetSchema.elementSize;
            view.chunkCapacity = runtime.args.chunkCapacity;
            view.streamCount = runtime.args.assemblyStreamCount;
            view.slotCount = runtime.args.assemblySlotCount;
            if(runtime.args.assemblyStorage == nullptr)
                return view;
            char* base = static_cast<char*>(runtime.args.assemblyStorage);
            view.streams = reinterpret_cast<AssemblyStreamState*>(base);
            view.arena
                = reinterpret_cast<std::uint8_t*>(base)
                  + static_cast<std::size_t>(runtime.args.assemblyStreamCount) * sizeof(AssemblyStreamState);
            return view;
        }

        /** True when the job's node writes at least one assembled stream. */
        ALPAKA_FN_ACC inline bool nodeAssembles(CompiledNode const& node) noexcept
        {
            return !node.isSource && !node.isSink && node.outputEdgeCount != 0u;
        }

        /** Publish a filled assembly chunk as a real packet and hand it to the
         *  ready ring as a successor job.
         *
         * The caller must hold the stream claim and have finished every lane's
         * element stores (the commit barrier ordered them). This runs only on
         * the leader. The published chunk's slot becomes a packet owned by the
         * downstream consumer; its record carries the true element count.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void publishPublishedChunk(
            SchedulerRuntime<T_Acc>& runtime,
            CompiledEdge const& edge,
            JobDescriptor const& parent,
            std::uint32_t slot,
            std::uint32_t count) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            // The filled chunk slot becomes the downstream packet. It stays a
            // pool slot (it was acquired at init and is owned by the stream
            // pair); we do not release/reacquire it. We only set its record to
            // describe the published count and hand the handle downstream.
            PacketHandle handle{slot, 0u};
            // The pool's record for this slot is updated with the true count.
            // `configureRecord` writes through the pool's read-only view and
            // publishes with a release so the consumer observes the count.
            PacketRecord& record = const_cast<PacketRecord&>(runtime.pool.slotRecord(handle));
            record.handle = handle;
            record.schema = runtime.args.packetSchema;
            record.capacity = runtime.args.chunkCapacity;
            record.elementCount = count;
            (void) addDevice(runtime.acc, &runtime.state.publications, std::uint64_t{1u});

            CompiledNode const& destination = runtime.graph.nodes[edge.to.value];
            JobDescriptor next = makeSuccessor(edge, destination, parent);
            next.packet = handle;
            if(runtime.warp.isLeader())
                (void) addDevice(runtime.acc, &runtime.state.outstanding, std::uint64_t{1u});
            (void) enqueueWithRetry(runtime, next);
        }

        /** Initialize the assembly state from the compiled graph.
         *
         * Runs once, on the initializer block, after the pool is configured.
         * For every compiled stream that assembles chunks (capacity `Y`, two
         * physical chunks), it claims two chunk slots from the pool, records
         * them as the stream's `first`/`second`, and marks the stream valid.
         * Terminal streams (one chunk) are not assembled; they consume their
         * terminal batch directly.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void initAssembly(
            SchedulerRuntime<T_Acc>& runtime,
            CompiledGraph const& graph,
            std::uint32_t chunkCapacity) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            if(!runtime.warp.isLeader() || runtime.args.assemblyStorage == nullptr)
                return;
            auto const view = makeAssemblyView(runtime);
            std::uint32_t used = 0u;
            for(std::uint32_t s = 0u; s < graph.streamCount; ++s)
            {
                CompiledStream const& stream = graph.streams[s];
                // Only double-buffered assembly streams (chunkCount == 2) keep
                // the two-chunk layout. A terminal stream (chunkCount == 1) is
                // consumed directly and needs no assembly state.
                if(stream.chunkCount != 2u || used + 2u > view.slotCount)
                    continue;
                PacketHandle firstHandle{};
                PacketHandle secondHandle{};
                if(runtime.pool.acquire(runtime.acc, runtime.warp, firstHandle)
                   && runtime.pool.acquire(runtime.acc, runtime.warp, secondHandle))
                {
                    AssemblyStreamState& state = view.streams[s];
                    state.id = stream.id;
                    state.capacity = chunkCapacity;
                    state.count = 0u;
                    state.firstSlot = firstHandle.index;
                    state.firstHandleIndex = firstHandle.index;
                    state.firstHandleGeneration = firstHandle.generation;
                    state.firstCount = 0u;
                    state.secondSlot = secondHandle.index;
                    state.secondHandleIndex = secondHandle.index;
                    state.secondHandleGeneration = secondHandle.generation;
                    state.claim = 0u;
                    state.version = 1u;
                    state.drained = false;
                    state.valid = true;
                    used += 2u;
                }
            }
            alpaka::onAcc::memFence(runtime.acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
        }

        /** Compute the per-element route of one input job's outputs.
         *
         * The planning pass (design section 6.3). `edgeOf(i)` is the
         * operation's per-element routing decision: the index (in `outputEdges`
         * order) of the edge element `i` goes to. Every lane classifies the
         * elements it owns, and the ballot/rank over the per-edge groups gives
         * each element its rank within its destination edge. The result is
         * written to the pool's per-element buffer (exclusive to the in-flight
         * job) and the per-edge counts are returned so the reservation commits
         * the append sizes.
         *
         * @param perEdgeCounts receives one count per outgoing edge, in
         *        `outputEdges` order.
         * @param edgeOf per-element destination edge (a callable or pointer
         *        the scheduler invokes per input index).
         */
        template<typename T_Acc, typename T_EdgeOf>
        ALPAKA_FN_ACC inline std::uint32_t planRoutes(
            SchedulerRuntime<T_Acc>& runtime,
            JobDescriptor const& job,
            CompiledNode const& node,
            std::uint32_t* perEdgeCounts,
            T_EdgeOf const& edgeOf) noexcept
        {
            std::uint32_t const count = job.request.count;
            std::uint32_t const width = runtime.warp.workerWidth();
            for(std::uint32_t e = 0u; e < node.outputEdgeCount; ++e)
                perEdgeCounts[e] = 0u;

            // Per-edge group ranks, accumulated in input order. The ballot is
            // warp-uniform, so every lane computes the same running base and
            // the same per-group rank. The rank of element `index` in edge `e`
            // is the base (earlier elements of edge `e`) plus the ballot rank
            // within the current base's active group.
            std::uint32_t edgeBase[maxAssemblyTargets]{};
            for(std::uint32_t e = 0u; e < node.outputEdgeCount; ++e)
                edgeBase[e] = 0u;

            for(std::uint32_t base = 0u; base < count; base += width)
            {
                std::uint32_t const index = base + runtime.warp.laneIndex();
                bool const active = index < count;
                std::uint32_t const edge = active ? edgeOf(index) : 0u;
                for(std::uint32_t e = 0u; e < node.outputEdgeCount; ++e)
                {
                    auto const mask = runtime.warp.ballot(active && edge == e);
                    if(edge == e && active)
                        runtime.state.elementRank[index] = edgeBase[e] + runtime.warp.rank(mask);
                    edgeBase[e] += WarpGroup<typename SchedulerRuntime<T_Acc>::Acc>::popcount(mask);
                }
                if(active)
                    runtime.state.elementEdge[index] = edge;
            }
            // Publish the per-edge counts (the final group bases).
            for(std::uint32_t e = 0u; e < node.outputEdgeCount; ++e)
                perEdgeCounts[e] = edgeBase[e];
            return count;
        }

        /** Try to reserve every assembled destination of one input job.
         *
         * All-or-nothing (design section 6.5): every target stream's writer
         * claim is taken together with its committed append size and (when the
         * append reaches `capacity`) one replacement chunk. A failure at any
         * target releases every claim already taken and leaves every counter
         * unchanged, so the job may be deferred and retried with no mutation.
         *
         * `perEdgeCounts[e]` is the committed element count of edge `e` (in
         * `outputEdges` order); its sum must not exceed the input's count. The
         * plan is broadcast to every lane so the write phase is warp-uniform.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline Plan reserveAssembly(
            SchedulerRuntime<T_Acc>& runtime,
            JobDescriptor const& job,
            CompiledNode const& node,
            std::uint32_t const* perEdgeCounts,
            PacketHandle const& newHandle) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            auto const view = makeAssemblyView(runtime);
            Plan plan{};
            if(view.streams == nullptr || !nodeAssembles(node) || node.outputEdgeCount > maxAssemblyTargets)
            {
                plan.status = PlanStatus::None;
                return plan;
            }

            std::uint32_t const ownerToken = ownerTokenOf(newHandle);

            if(runtime.warp.isLeader())
            {
                std::uint32_t acquired = 0u;
                std::uint32_t replacementCursor = 0u;
                for(std::uint32_t e = 0u; e < node.outputEdgeCount; ++e)
                {
                    std::uint32_t const edgeIndex
                        = runtime.graph.outputEdges[node.firstOutputEdge + e];
                    CompiledEdge const& edge = runtime.graph.edges[edgeIndex];
                    std::uint32_t const index = view.find(edge.stream);
                    if(index == invalidChunkSlot)
                    {
                        plan.status = PlanStatus::None;
                        break;
                    }
                    auto& state = view.streams[index];
                    if(state.drained)
                    {
                        plan.status = PlanStatus::StreamDrained;
                        break;
                    }
                    if(!StreamAssembler<Acc>::tryClaim(runtime.acc, &state.claim, ownerToken))
                    {
                        plan.status = PlanStatus::StreamBusy;
                        break;
                    }
                    std::uint32_t const k = perEdgeCounts[e];
                    std::uint64_t const sum
                        = static_cast<std::uint64_t>(state.count) + static_cast<std::uint64_t>(k);
                    if(sum > StreamAssembler<Acc>::maxAppendOccupancy(state.capacity))
                    {
                        plan.status = PlanStatus::OccupancyOverflow;
                        break;
                    }
                    bool const rotate = sum >= state.capacity;
                    if(rotate && replacementCursor >= view.slotCount)
                    {
                        plan.status = PlanStatus::NotEnoughReplacements;
                        break;
                    }
                    plan.targets[acquired].streamIndex = index;
                    plan.targets[acquired].itemCount = k;
                    plan.targets[acquired].baseCount = state.count;
                    plan.targets[acquired].rotate = rotate;
                    plan.targets[acquired].replacement = rotate ? replacementCursor : invalidChunkSlot;
                    if(rotate)
                        ++replacementCursor;
                    ++acquired;
                }
                if(plan.status == PlanStatus::None)
                    plan.status = PlanStatus::Ok;
                if(plan.status != PlanStatus::Ok)
                {
                    for(std::uint32_t i = 0u; i < acquired; ++i)
                    {
                        auto& state = view.streams[plan.targets[i].streamIndex];
                        StreamAssembler<Acc>::releaseClaim(runtime.acc, &state.claim, ownerToken);
                    }
                    plan.targets = {};
                    plan.targetCount = 0u;
                }
                else
                {
                    plan.targetCount = acquired;
                }
            }

            std::uint32_t const status
                = runtime.warp.template broadcast<std::uint32_t>(static_cast<std::uint32_t>(plan.status), 0u);
            plan.status = static_cast<PlanStatus>(status);
            std::uint32_t const count = runtime.warp.template broadcast<std::uint32_t>(plan.targetCount, 0u);
            plan.targetCount = count;
            for(std::uint32_t i = 0u; i < count; ++i)
            {
                plan.targets[i].streamIndex
                    = runtime.warp.template broadcast<std::uint32_t>(plan.targets[i].streamIndex, 0u);
                plan.targets[i].itemCount = runtime.warp.template broadcast<std::uint32_t>(plan.targets[i].itemCount, 0u);
                plan.targets[i].baseCount = runtime.warp.template broadcast<std::uint32_t>(plan.targets[i].baseCount, 0u);
                plan.targets[i].rotate = runtime.warp.template broadcast<std::uint32_t>(
                                             plan.targets[i].rotate ? 1u : 0u, 0u)
                                         != 0u;
                plan.targets[i].replacement
                    = runtime.warp.template broadcast<std::uint32_t>(plan.targets[i].replacement, 0u);
            }
            (void) job;
            return plan;
        }

        /** Element write pointer for logical rank `rank` of `target`.
         *
         * Mirrors `StreamReservation::writePointer` over the erased arena. The
         * caller holds the reservation (the stream is claimed), so the base
         * pointer is stable for the write phase.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void* assemblyWritePointer(
            AssemblyView<typename SchedulerRuntime<T_Acc>::Acc> const& view,
            PlanTarget const& target,
            std::uint32_t rank) noexcept
        {
            auto const& state = view.streams[target.streamIndex];
            std::uint64_t const logical = static_cast<std::uint64_t>(target.baseCount) + rank;
            if(logical < state.capacity)
                return view.chunkData(state.firstSlot) + logical * view.elementSize;
            auto const offset = logical - state.capacity;
            return view.chunkData(state.secondSlot) + offset * view.elementSize;
        }

        /** Commit a reserved append: apply the counters, rotate, and hand the
         *  filled chunk downstream as a real packet with its true count.
         *
         * Only the leader mutates. A `WarpBarrier` orders the lanes' element
         * stores against the leader's metadata publication, and the leader's
         * release fence publishes the slot moves and the publication descriptor.
         * The input packet is released exactly once after the commit (design
         * section 6.5).
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool commitAssembly(
            SchedulerRuntime<T_Acc>& runtime,
            Plan const& plan,
            PacketHandle const& inputPacket) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            auto const view = makeAssemblyView(runtime);
            auto const blockThreads = static_cast<std::uint32_t>(
                runtime.acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
            WarpBarrier<Acc> barrier{};
            barrier.barrier(runtime.acc, blockThreads, runtime.warp);
            if(!runtime.warp.isLeader())
                return true;

            auto const ownerToken = ownerTokenOf(inputPacket);
            bool ok = true;
            for(std::uint32_t i = 0u; i < plan.targetCount; ++i)
            {
                auto const& state = view.streams[plan.targets[i].streamIndex];
                if(state.claim != ownerToken || state.count != plan.targets[i].baseCount)
                    ok = false;
            }
            if(!ok)
            {
                for(std::uint32_t i = 0u; i < plan.targetCount; ++i)
                    StreamAssembler<Acc>::releaseClaim(
                        runtime.acc, &view.streams[plan.targets[i].streamIndex].claim, ownerToken);
                return false;
            }
            alpaka::onAcc::memFence(runtime.acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);

            for(std::uint32_t i = 0u; i < plan.targetCount; ++i)
            {
                auto& state = view.streams[plan.targets[i].streamIndex];
                auto const& target = plan.targets[i];
                std::uint64_t const sum = static_cast<std::uint64_t>(target.baseCount) + target.itemCount;
                if(target.rotate)
                {
                    // Hand the filled first chunk downstream as a real packet
                    // with its full count. The publication is counted before it
                    // is released, so it is always observable.
                    publishPublishedChunk(runtime, state.firstSlot, state.capacity);
                    state.firstSlot = state.secondSlot;
                    state.firstHandleIndex = state.secondHandleIndex;
                    state.firstHandleGeneration = state.secondHandleGeneration;
                    state.firstCount = static_cast<std::uint32_t>(sum - state.capacity);
                    state.secondSlot = target.replacement;
                    state.count = static_cast<std::uint32_t>(sum - state.capacity);
                }
                else
                {
                    state.firstCount = static_cast<std::uint32_t>(sum);
                    state.count = static_cast<std::uint32_t>(sum);
                }
                StreamAssembler<Acc>::releaseClaim(runtime.acc, &state.claim, ownerToken);
                state.version += 1u;
            }
            if(inputPacket.index < static_cast<std::uint32_t>(poolPacketSlotCapacity))
            {
                runtime.pool.release(runtime.acc, runtime.warp, inputPacket);
                (void) addDevice(runtime.acc, &runtime.state.slotsReturned, std::uint64_t{1u});
            }
            return true;
        }

        /** Publish successors, the source continuation, and retire the input. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool publishSuccessors(
            SchedulerRuntime<T_Acc>& runtime,
            JobDescriptor const& job) noexcept
        {
            CompiledNode const* node = findNode(runtime.graph, job.operation);
            if(node == nullptr)
            {
                if(runtime.warp.isLeader())
                    recordError(runtime, PoolError::unknownOperation);
                return false;
            }

            // The handle broadcast with the ready entry (scalar index and
            // generation, then the whole descriptor through the leader) names
            // the packet that was acquired for this job. It is the authority
            // for the successor: re-deriving it from the pool slot would alias
            // whichever packet the slot currently holds and hide a stale or
            // lost descriptor.
            PacketHandle const ownedPacket = job.packet;

            bool ok = true;
            // Resumable source range: emit at most one chunk per invocation and
            // re-queue the remainder as a new source job.
            if(node->isSource)
            {
                std::uint32_t const emitted = job.request.count < runtime.args.chunkCapacity
                                                  ? static_cast<std::uint32_t>(job.request.count)
                                                  : runtime.args.chunkCapacity;
                if(job.request.count > static_cast<std::uint64_t>(emitted))
                {
                    PacketHandle continuation{};
                    bool acquired = acquireWithRetry(runtime, continuation);
                    if(acquired)
                    {
                        if(runtime.warp.isLeader())
                        {
                            std::uint64_t const remaining = job.request.count - emitted;
                            std::uint32_t const nextCount = remaining < runtime.args.chunkCapacity
                                                                ? static_cast<std::uint32_t>(remaining)
                                                                : runtime.args.chunkCapacity;
                            configureRecord(runtime, continuation, nextCount);
                            (void) addDevice(runtime.acc, &runtime.state.outstanding, std::uint64_t{1u});
                            (void) addDevice(runtime.acc, &runtime.state.publications, std::uint64_t{1u});
                        }
                        JobDescriptor next = job;
                        // The acquire above broadcast the fresh handle to every
                        // lane, so the continuation descriptor carries the exact
                        // index/generation the pool just handed out.
                        next.packet = continuation;
                        next.request.firstId = job.request.firstId + emitted;
                        next.request.count = job.request.count - emitted;
                        RingStatus const status = enqueueWithRetry(runtime, next);
                        if(status != RingStatus::Accepted)
                        {
                            if(runtime.warp.isLeader())
                                orDevice(runtime.acc, &runtime.state.errorWord, publicationError(status));
                            ok = false;
                        }
                    }
                    else
                    {
                        if(runtime.warp.isLeader())
                            recordError(runtime, PoolError::packetExhausted);
                        ok = false;
                    }
                }
                else if(runtime.warp.isLeader())
                {
                    // Last chunk of this accepted range: the source is done.
                    (void) addDevice(runtime.acc, &runtime.state.pendingSources, std::uint64_t{0u} - 1u);
                }
            }

            // One successor per outgoing edge. Pool-core forwards the same
            // packet in place; per-destination output sizes are the chunk
            // assembly milestone.
            for(std::uint32_t index = 0u; ok && index < node->outputEdgeCount; ++index)
            {
                std::uint32_t const edgeIndex = runtime.graph.outputEdges[node->firstOutputEdge + index];
                CompiledEdge const& edge = runtime.graph.edges[edgeIndex];
                CompiledNode const& destination = runtime.graph.nodes[edge.to.value];
                JobDescriptor next = makeSuccessor(edge, destination, job);
                next.packet = ownedPacket;
                if(runtime.warp.isLeader())
                {
                    (void) addDevice(runtime.acc, &runtime.state.outstanding, std::uint64_t{1u});
                    (void) addDevice(runtime.acc, &runtime.state.publications, std::uint64_t{1u});
                }
                RingStatus const status = enqueueWithRetry(runtime, next);
                if(status != RingStatus::Accepted)
                {
                    if(runtime.warp.isLeader())
                        orDevice(runtime.acc, &runtime.state.errorWord, publicationError(status));
                    ok = false;
                }
            }

            if(node->outputEdgeCount == 0u)
            {
                // Pool-core retirement is accounting only: the packet pool's
                // slots are not recycled inside one pool lifetime. Acyclic runs
                // are bounded by the validated slot budget, and per-stream slot
                // recycling needs the chunk-assembly ownership protocol (next
                // milestone). Teardown releases the whole allocation.
                if(runtime.warp.isLeader())
                    (void) addDevice(runtime.acc, &runtime.state.retiredPackets, std::uint64_t{1u});
            }
            return ok;
        }

        /** Execute one claimed job with every lane of the warp.
         *
         * For a node that assembles chunks, the job's outputs are routed
         * through the reservation/commit protocol against the pool's assembly
         * state. The operation's functor writes its output elements to the
         * committed addresses (via the per-element route the planning pass
         * computed), and the scheduler commits the append, rotates the chunk
         * pair, and hands the filled chunk downstream as a real packet with its
         * true element count.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool executeJob(SchedulerRuntime<T_Acc>& runtime, JobDescriptor const& job) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            CompiledNode const* node = findNode(runtime.graph, job.operation);
            bool const isSource = node != nullptr && node->isSource;
            // A non-source job names the packet it reads. Reject a handle whose
            // generation no longer matches the slot: executing it would read a
            // packet the job does not own.
            if(!isSource && !packetHandleValid(runtime, job.packet))
                return false;
            PacketAccessor accessor{};
            if(job.packet.index < static_cast<std::uint32_t>(poolPacketSlotCapacity))
            {
                PacketRecord const record = runtime.pool.observedRecord(runtime.acc, job.packet);
                void* data = runtime.pool.slotData(job.packet);
                if(!isSource)
                    accessor.setInput(job.routing.inputPort, &record, data);
                for(std::uint32_t port = 0u; port < job.routing.outputPortCount && port + 1u < maxPacketPorts; ++port)
                    accessor.setOutput(port, &record, data);
            }
            // Reserved control port: never a stream, never part of routing.
            accessor.inputData[maxPacketPorts - 1u] = runtime.args.controlBlock;

            OperationFn<Acc> const function = runtime.table.resolve(job.operation);
            if(function == nullptr)
            {
                if(runtime.warp.isLeader())
                    recordError(runtime, PoolError::unknownOperation);
                return false;
            }

            bool const assembles = node != nullptr && nodeAssembles(*node);
            if(!assembles)
            {
                // Terminal or source node: the operation reads its input and
                // writes its output slot(s) directly (no assembly). The
                // successor publication and input retirement happen in
                // `publishSuccessors` (source continuation, terminal retire).
                function(runtime.acc, job, accessor);
                if(runtime.warp.isLeader() && job.operation.value < 8u)
                    (void) addDevice(runtime.acc, &runtime.state.execCounts[job.operation.value], std::uint64_t{1u});
                return true;
            }

            // Assembly node. The operation is a pure classifier: it reads the
            // input packet and writes, for every input element, the destination
            // edge and its rank within that edge's group, into the pool's
            // per-element route buffer. It returns the per-edge counts. The
            // scheduler then reserves the append, copies each element to its
            // committed address, and commits the rotation. This keeps the
            // per-particle transformation in the operation (where it belongs)
            // and the reservation/rotation in the scheduler (where the design
            // puts it), with no element-type erasure in the scheduler.
            auto const view = makeAssemblyView(runtime);
            std::uint32_t perEdge[maxAssemblyTargets]{};
            classifyAssembly(runtime, job, accessor, *node, function, perEdge);
            Plan plan = reserveAssembly(runtime, job, *node, perEdge, job.packet);
            if(plan.status != PlanStatus::Ok)
            {
                if(plan.status == PlanStatus::StreamBusy)
                {
                    // Deferred: another writer holds a target stream. No
                    // mutation occurred; the job is retried (still outstanding).
                    return true;
                }
                if(runtime.warp.isLeader())
                    recordError(runtime, PoolError::internalState);
                return false;
            }
            writeAssembly(runtime, view, plan, job, accessor);
            bool const committed = commitAssembly(runtime, plan, job.packet);
            if(runtime.warp.isLeader() && job.operation.value < 8u)
                (void) addDevice(runtime.acc, &runtime.state.execCounts[job.operation.value], std::uint64_t{1u});
            return committed;
        }

        /** The persistent scheduling loop of one worker warp. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void schedulerLoop(SchedulerRuntime<T_Acc>& runtime, std::uint32_t blockIndex) noexcept
        {
            std::uint64_t idleIterations = 0u;
            while(true)
            {
                if(loadDeviceAcquire(runtime.acc, &runtime.state.stopReason) != 0u)
                    return;

                bool didWork = false;
                if(blockIndex == 0u)
                {
                    importerStep(runtime);
                    // Uniform across the warp: every lane observes the mapped
                    // abort word, so the branch cannot diverge before the next
                    // collective.
                    bool const hostAborted = hostAbortRequested(runtime);
                    if(hostAborted)
                    {
                        if(runtime.warp.isLeader())
                        {
                            recordError(runtime, PoolError::aborted);
                            requestStop(runtime, poolStoppingAborted, 6u);
                        }
                    }
                    if(loadDeviceAcquire(runtime.acc, &runtime.state.stopReason) != 0u)
                        return;
                }

                JobDescriptor job{};
                RingStatus const status = runtime.ring.dequeue(runtime.acc, runtime.warp, job);
                if(status == RingStatus::Accepted)
                {
                    idleIterations = 0u;
                    if(runtime.warp.isLeader())
                    {
                        (void) addDevice(runtime.acc, &runtime.state.active, std::uint64_t{1u});
                        (void) addDevice(runtime.acc, &runtime.state.claimAttempts, std::uint64_t{1u});
                        std::uint32_t const word = blockIndex / 64u;
                        std::uint64_t const bit = std::uint64_t{1u} << (blockIndex % 64u);
                        (void) orDevice(runtime.acc, &runtime.state.workerBits[word], bit);
                    }
                    // Order the leader's acquire before any lane reads packet data.
                    runtime.barrier.barrier(runtime.acc, runtime.blockThreads, runtime.warp);
                    bool const executed = executeJob(runtime, job);
                    // Order every lane's stores before the leader publishes or retires.
                    runtime.barrier.barrier(runtime.acc, runtime.blockThreads, runtime.warp);
                    bool published = false;
                    if(executed)
                        published = publishSuccessors(runtime, job);
                    if(runtime.warp.isLeader())
                    {
                        (void) addDevice(runtime.acc, &runtime.state.active, std::uint64_t{0u} - 1u);
                        if(executed && published)
                        {
                            (void) addDevice(runtime.acc, &runtime.state.completedJobs, std::uint64_t{1u});
                            std::uint64_t const local = ++runtime.state.workerRetired[blockIndex];
                            graph::systemStoreRelease(runtime.acc, &runtime.host.workerRetired[blockIndex], local);
                            (void) addDevice(runtime.acc, &runtime.state.outstanding, std::uint64_t{0u} - 1u);
                        }
                        else
                        {
                            recordError(runtime, PoolError::internalState);
                            requestStop(runtime, poolStoppingAborted, executed ? 3u : 2u);
                        }
                    }
                    didWork = true;
                }
                else if(status == RingStatus::Contended)
                {
                    devicePause(runtime.args.idleNanosleep);
                }
                else
                {
                    ++idleIterations;
                    if(idleIterations >= 4u)
                        tryQuiesce(runtime);
                    devicePause(runtime.args.idleNanosleep);
                }
            }
        }

        /** One persistent worker kernel. */
        template<typename T_Binder>
        struct PoolKernel
        {
            template<typename T_Acc>
            ALPAKA_FN_ACC void operator()(T_Acc const& acc, PoolKernelArgs args) const
            {
                using Acc = std::remove_cvref_t<T_Acc>;
                using Ring = ReadyRing<Acc, poolReadyCapacity>;
                using Pool = PacketPool<Acc, poolPacketSlotCapacity, poolPacketElementBytes>;
                using Table = OperationTable<Acc>;
                using Runtime = SchedulerRuntime<Acc>;

                static_assert(
                    sizeof(typename Ring::RingHeader) == sizeof(ReadyRingLayout),
                    "graph: the host ready-ring header mirror does not match the device layout");
                static_assert(
                    sizeof(Table) == sizeof(OperationTableLayout),
                    "graph: the host operation-table mirror does not match the device layout");
                static_assert(
                    sizeof(void*) == sizeof(OperationFn<Acc>),
                    "graph: a device operation pointer must have the size of a void pointer");

                WarpGroup<Acc> const warp{acc};
                WarpBarrier<Acc> const barrier{};
                auto const blockThreads = static_cast<std::uint32_t>(
                    acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
                auto const blockIndex = static_cast<std::uint32_t>(
                    acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)[0]);

                // Loud capacity guard: every per-worker array in the device
                // state (`blockPhase`, `workerRetired`, `workerBits`) is sized
                // by `poolMaxWorkers` and indexed by the block index. A geometry
                // beyond that capacity is rejected here, before any index is
                // formed, so it is never accepted and then indexed out of range.
                if(blockIndex >= static_cast<std::uint32_t>(poolMaxWorkers))
                {
                    if(warp.isLeader())
                    {
                        orDevice(
                            acc,
                            &static_cast<PoolDeviceState*>(args.state)->errorWord,
                            poolErrorBits(PoolError::workerCapacityExceeded));
                        (void) exchangeDevice(
                            acc,
                            &static_cast<PoolDeviceState*>(args.state)->stopReason,
                            poolStoppingAborted);
                    }
                    return;
                }

                if(warp.isLeader())
                {
                    static_cast<PoolDeviceState*>(args.state)->debugBlockThreads = blockThreads;
                    static_cast<PoolDeviceState*>(args.state)->debugWarpWidth = warp.workerWidth();
                }
                if(blockThreads != warp.workerWidth())
                {
                    if(warp.isLeader())
                        orDevice(
                            acc,
                            &static_cast<PoolDeviceState*>(args.state)->errorWord,
                            poolErrorBits(PoolError::internalState));
                    return;
                }

                Ring ring{args.readyStorage};
                Pool pool{args.packetStorage};
                Table& table = *static_cast<Table*>(args.operationTable);

                Runtime runtime{
                    acc,
                    warp,
                    barrier,
                    ring,
                    pool,
                    table,
                    *static_cast<CompiledGraph const*>(args.graph),
                    *static_cast<PoolDeviceState*>(args.state),
                    *args.hostControl,
                    args,
                    blockThreads};

                bool const started = startupRendezvous<T_Binder>(
                    runtime,
                    blockIndex,
                    args.initDeadlineCycles,
                    args.initDeadlineIterations);
                if(!started)
                    return;
                schedulerLoop(runtime, blockIndex);
            }
        };
    } // namespace detail
} // namespace graph
