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
            // Barrier divergence diagnostics (see
            // graph/detail/barrier_diagnostics.hpp). `barrierDiagnostics` is
            // the launch's diagnostics state; the two booleans mirror the host
            // flags of `PoolConfig`.
            void* barrierDiagnostics{nullptr};
            std::uint32_t barrierDiagnosticsEnabled{0u};
            std::uint32_t barrierSkipOnDivergence{0u};
        };

        static_assert(std::is_trivially_copyable_v<PoolKernelArgs>);
        static_assert(std::is_standard_layout_v<PoolKernelArgs>);

        template<typename T_Acc>
        struct SchedulerRuntime
        {
            using Acc = std::remove_cvref_t<T_Acc>;
            using Ring = ReadyRing<Acc, poolReadyCapacity>;
            using Pool = PacketPool<Acc, poolPacketSlotCapacity, poolPacketElementBytes>;
            using Table = OperationTable<Acc>;

            Acc const& acc;
            WarpGroup<Acc> const& warp;
            /** Divergence diagnostics shared by every barrier and handoff. */
            WarpBarrierDiagnostics<Acc> const& diagnostics;
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

        /** Byte-level assembly view used by the scheduler.
         *
         * The scheduler does not know the functor element type. It copies
         * `count * elementSize` bytes from an output span into the destination
         * stream's reserved logical range and converts back to element counts
         * at publication. Chunk capacities in the assembler are therefore byte
         * capacities (`Y * elementSize`).
         */
        using AssemblyElement = std::uint8_t;
        using AssemblyStreamState = AssemblyStream<AssemblyElement>;
        using AssemblySlot = ChunkSlot<AssemblyElement>;
        using AssemblyReservation = StreamReservation<AssemblyElement>;
        using Assembly = StreamAssembler<AssemblyElement>;

        /** Build the assembler over the preallocated stream table and the
         *  packet pool's payload area.
         *
         * The arena overlays the packet pool: slot `i` of the arena is slot
         * `i` of the pool, so a chunk slot's handle travels downstream with
         * the published packet. The stride between slots is the pool's slot
         * stride, while the logical capacity is one chunk.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline Assembly makeAssembly(SchedulerRuntime<T_Acc> const& runtime) noexcept
        {
            using Pool = typename SchedulerRuntime<T_Acc>::Pool;
            ChunkArena<AssemblyElement> arena{};
            if(runtime.args.packetStorage != nullptr && runtime.args.chunkCapacity != 0u)
            {
                arena.storage = static_cast<AssemblyElement*>(runtime.args.packetStorage) + Pool::headerBytes;
                arena.slotCount = runtime.args.assemblySlotCount;
                arena.elementCapacity = runtime.args.chunkCapacity * runtime.args.packetSchema.elementSize;
                arena.elementStride = static_cast<std::uint32_t>(Pool::slotBytes);
            }
            return Assembly(
                static_cast<AssemblyStreamState*>(runtime.args.assemblyStorage),
                runtime.args.assemblyStreamCount,
                arena);
        }

        /** Physical chunk descriptor of a packet-pool slot in the arena. */
        ALPAKA_FN_ACC inline AssemblySlot makeAssemblySlot(
            PacketHandle const& handle,
            std::uint32_t capacityBytes) noexcept
        {
            return AssemblySlot{handle.index, capacityBytes, 0u, handle};
        }

        /** Find the compiled stream entry for a stream id. */
        ALPAKA_FN_ACC inline CompiledStream const* findCompiledStream(CompiledGraph const& graph, StreamId id) noexcept
        {
            for(std::uint32_t index = 0u; index < graph.streamCount; ++index)
            {
                if(graph.streams[index].id == id)
                    return &graph.streams[index];
            }
            return nullptr;
        }

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
            std::uint32_t assembled{1u};
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

        /** Mark a freshly acquired slot as an empty assembly chunk. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void configureChunkRecord(
            SchedulerRuntime<T_Acc>& runtime,
            PacketHandle const& handle) noexcept
        {
            configureRecord(runtime, handle, 0u);
        }

        /** Mark a freshly acquired slot as a full-capacity output scratch span. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void configureOutputRecord(
            SchedulerRuntime<T_Acc>& runtime,
            PacketHandle const& handle) noexcept
        {
            configureRecord(runtime, handle, runtime.args.chunkCapacity);
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

        // ------------------------------------------------------------------
        // Exit-path diagnostics (clearly marked; cheap relaxed stores).
        //
        // Every worker warp writes its current loop `stage`, an iteration
        // counter, and a `last action` word at each loop transition. The host
        // copies these words back once the kernel has stopped and prints every
        // worker that is not `exiting`, which names the loop stage of a worker
        // that never left the scheduling loop after a stop request. The stores
        // are relaxed: they carry no synchronization meaning for the pool
        // protocol, and a reader that observes a stage value also observes no
        // more than the counter values of that same transition.
        // ------------------------------------------------------------------
        /** Loop stages of one worker warp, written by the scheduling loop. */
        enum class WorkerStage : std::uint32_t
        {
            idle = 0u,
            startup = 1u,
            waitInit = 2u,
            pollReady = 3u,
            importerStep = 4u,
            backoff = 5u,
            executing = 6u,
            publishing = 7u,
            retiring = 8u,
            waitingLock = 9u,
            exiting = 10u
        };

        /** Last successful action of one worker warp, written with the stage. */
        enum class WorkerLastAction : std::uint32_t
        {
            none = 0u,
            claim = 1u,
            execute = 2u,
            publish = 3u,
            retire = 4u,
            import = 5u,
            quiescence = 6u
        };

        static_assert(std::is_trivially_copyable_v<WorkerStage>);
        static_assert(std::is_trivially_copyable_v<WorkerLastAction>);

        ALPAKA_FN_HOST_ACC inline char const* toString(WorkerStage stage) noexcept
        {
            switch(stage)
            {
            case WorkerStage::idle:
                return "idle";
            case WorkerStage::startup:
                return "startup";
            case WorkerStage::waitInit:
                return "waitInit";
            case WorkerStage::pollReady:
                return "pollReady";
            case WorkerStage::importerStep:
                return "importerStep";
            case WorkerStage::backoff:
                return "backoff";
            case WorkerStage::executing:
                return "executing";
            case WorkerStage::publishing:
                return "publishing";
            case WorkerStage::retiring:
                return "retiring";
            case WorkerStage::waitingLock:
                return "waitingLock";
            case WorkerStage::exiting:
                return "exiting";
            }
            return "?";
        }

        ALPAKA_FN_HOST_ACC inline char const* toString(WorkerLastAction action) noexcept
        {
            switch(action)
            {
            case WorkerLastAction::none:
                return "none";
            case WorkerLastAction::claim:
                return "claim";
            case WorkerLastAction::execute:
                return "execute";
            case WorkerLastAction::publish:
                return "publish";
            case WorkerLastAction::retire:
                return "retire";
            case WorkerLastAction::import:
                return "import";
            case WorkerLastAction::quiescence:
                return "quiescence";
            }
            return "?";
        }

        /** Record the current loop stage for exit-path diagnostics. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void recordStage(
            SchedulerRuntime<T_Acc> const& runtime,
            std::uint32_t blockIndex,
            WorkerStage stage,
            WorkerLastAction action = WorkerLastAction::none) noexcept
        {
            if(runtime.warp.isLeader())
            {
                auto* state = static_cast<PoolDeviceState*>(runtime.args.state);
                dev::storeRelease(&state->workerStage[blockIndex], static_cast<std::uint32_t>(stage));
                dev::storeRelease(
                    &state->workerIterations[blockIndex],
                    dev::loadRelaxed(&state->workerIterations[blockIndex]) + std::uint64_t{1u});
                dev::storeRelease(&state->workerLastAction[blockIndex], static_cast<std::uint32_t>(action));
            }
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
                if(runtime.pool.acquire(runtime.acc, runtime.warp, handle, runtime.diagnostics))
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
                status = runtime.ring.enqueue(runtime.acc, runtime.warp, entry, runtime.diagnostics);
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

        /** Publish one assembled chunk to every consumer of its stream.
         *
         * The chunk's packet record must already carry its real element count.
         * A job is counted as outstanding before it becomes visible, so the
         * acyclic stop rule can never observe the chunk missing. One job is
         * published per distinct consumer binding, not per edge: several
         * producers may feed one input port through the same stream, and the
         * compiled consumer table has already deduplicated them.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool publishChunk(
            SchedulerRuntime<T_Acc>& runtime,
            StreamId streamId,
            PacketHandle const& handle,
            std::uint32_t elementCount,
            std::uint64_t seed,
            std::uint64_t sequenceNumber) noexcept
        {
            CompiledStream const* stream = findCompiledStream(runtime.graph, streamId);
            if(stream == nullptr || stream->consumerCount == 0u)
            {
                if(runtime.warp.isLeader())
                    recordError(runtime, PoolError::internalState);
                return false;
            }
            configureRecord(runtime, handle, elementCount);
            bool ok = true;
            for(std::uint32_t index = 0u; ok && index < stream->consumerCount; ++index)
            {
                CompiledPortBinding const& binding = runtime.graph.streamConsumers[stream->firstConsumer + index];
                if(binding.node.value >= runtime.graph.nodeCount)
                {
                    if(runtime.warp.isLeader())
                        recordError(runtime, PoolError::internalState);
                    return false;
                }
                CompiledNode const& destination = runtime.graph.nodes[binding.node.value];
                JobDescriptor next{};
                next.operation = destination.operation;
                next.kind = JobKind::OPERATION;
                next.packet = handle;
                next.routing.inputPort = binding.port;
                next.routing.outputPort = 0u;
                next.routing.outputPortCount = destination.outputPortCount;
                next.request = RequestSpec{0u, 0u, seed};
                next.sequenceNumber = sequenceNumber;
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
            return ok;
        }

        /** Acquire a packet slot with a bounded attempt budget.
         *
         * Unlike `acquireWithRetry`, a failure is reported instead of spinning
         * forever. The drain path uses it so a transiently exhausted pool
         * makes the worker retry the quiescence step instead of wedging the
         * persistent launch. Warp-uniform.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool acquireBounded(
            SchedulerRuntime<T_Acc>& runtime,
            PacketHandle& handle,
            std::uint32_t attempts) noexcept
        {
            for(std::uint32_t attempt = 0u; attempt < attempts; ++attempt)
            {
                if(runtime.pool.acquire(runtime.acc, runtime.warp, handle, runtime.diagnostics))
                    return true;
                devicePause(runtime.args.idleNanosleep);
            }
            return false;
        }

        /** Initialize one A/B chunk pair per compiled stream.
         *
         * The two slots come from the packet pool's free list, so a published
         * chunk keeps a real packet handle and travels downstream without a
         * copy. Runs in the initializer block before `initDone` is published.
         * Warp-uniform.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool initializeAssembly(SchedulerRuntime<T_Acc>& runtime) noexcept
        {
            if(runtime.args.assemblyStorage == nullptr)
                return runtime.graph.streamCount == 0u;
            Assembly assembler = makeAssembly(runtime);
            if(assembler.streamCount() != runtime.graph.streamCount)
                return false;
            std::uint32_t const capacityBytes = runtime.args.chunkCapacity * runtime.args.packetSchema.elementSize;
            bool ok = true;
            for(std::uint32_t index = 0u; ok && index < runtime.graph.streamCount; ++index)
            {
                PacketHandle first{};
                PacketHandle second{};
                if(!acquireWithRetry(runtime, first))
                {
                    if(runtime.warp.isLeader())
                        recordError(runtime, PoolError::packetExhausted);
                    return false;
                }
                configureChunkRecord(runtime, first);
                if(!acquireWithRetry(runtime, second))
                {
                    if(runtime.warp.isLeader())
                        recordError(runtime, PoolError::packetExhausted);
                    return false;
                }
                configureChunkRecord(runtime, second);
                ok = assembler.initialize(
                    runtime.acc,
                    runtime.warp,
                    index,
                    runtime.graph.streams[index].id,
                    makeAssemblySlot(first, capacityBytes),
                    makeAssemblySlot(second, capacityBytes),
                    runtime.diagnostics);
            }
            return ok;
        }

        /** One quiescence observation, with an optional drain step.
         *
         * Warp-uniform. A single leader takes the drain lock (the
         * `inFlightReservations` control word), so the observation and any
         * drain publication form one transaction: no other worker can publish a
         * chunk or observe the same inconsistent state in between. If a
         * non-empty partial buffer exists, one chunk is detached with its real
         * length, released as an ordinary job with a fresh replacement chunk
         * attached to the stream, and the step returns without stopping. A
         * particle that returns during the drain therefore creates a new
         * partial buffer and a later round sees it. The drained stop flag is
         * set only when one observation shows no pending sources, no outstanding
         * jobs, no staged imports, and no non-empty partial buffer.
         *
         * @return true when the worker must leave the scheduling loop.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool quiescenceStep(SchedulerRuntime<T_Acc>& runtime, std::uint32_t blockIndex) noexcept
        {
            auto const& warp = runtime.warp;
            bool const leader = warp.isLeader();

            std::uint32_t locked = 0u;
            if(leader)
            {
                std::uint64_t const closed = loadDeviceAcquire(runtime.acc, &runtime.state.closedSeen);
                std::uint64_t const pendingSources = loadDeviceAcquire(runtime.acc, &runtime.state.pendingSources);
                std::uint64_t const outstanding = loadDeviceAcquire(runtime.acc, &runtime.state.outstanding);
                std::uint64_t const staging = loadDeviceAcquire(runtime.acc, &runtime.state.stagingCount);
                if(closed != 0u && pendingSources == 0u && outstanding == 0u && staging == 0u)
                {
                    locked = alpaka::onAcc::atomicCas(
                                 runtime.acc,
                                 &runtime.state.inFlightReservations,
                                 std::uint64_t{0u},
                                 std::uint64_t{1u},
                                 alpaka::onAcc::scope::device)
                                     == 0u
                                 ? 1u
                                 : 0u;
                }
            }
            // Reconverge after the leader-only CAS before the shuffle.
            runtime.barrier.barrier(
                runtime.acc,
                runtime.blockThreads,
                warp,
                barrier_id::quiescenceCas,
                static_cast<std::uint32_t>(WorkerStage::retiring),
                0u);
            locked = warp.template broadcast<std::uint32_t>(locked, 0u);
            if(locked == 0u)
                return false;

            std::uint32_t action = 0u; // 0 none, 1 stop, 2 drain, 3 abort
            std::uint32_t drainStream = 0u;
            if(leader)
            {
                std::uint64_t const closed = loadDeviceAcquire(runtime.acc, &runtime.state.closedSeen);
                std::uint64_t const pendingSources = loadDeviceAcquire(runtime.acc, &runtime.state.pendingSources);
                std::uint64_t const outstanding = loadDeviceAcquire(runtime.acc, &runtime.state.outstanding);
                std::uint64_t const staging = loadDeviceAcquire(runtime.acc, &runtime.state.stagingCount);
                if(closed != 0u && pendingSources == 0u && outstanding == 0u && staging == 0u)
                {
                    std::uint64_t const packetViolations = runtime.pool.ownershipViolations();
                    std::uint64_t const violations = runtime.ring.ownershipViolations();
                    if(packetViolations != 0u)
                    {
                        recordError(runtime, PoolError::internalState);
                        action = 3u;
                    }
                    else if(violations != 0u)
                    {
                        runtime.state.readyViolations = violations;
                        recordError(runtime, PoolError::readyRingViolation);
                        action = 3u;
                    }
                    else
                    {
                        // Pair with the release fence of the last commit before
                        // reading the stream counters. Without it a drain scan
                        // can miss a just-committed partial buffer and stop the
                        // pool with data still buffered.
                        alpaka::onAcc::memFence(
                            runtime.acc,
                            alpaka::onAcc::scope::device,
                            alpaka::onAcc::order::acquire);
                        Assembly assembler = makeAssembly(runtime);
                        std::uint64_t partials = 0u;
                        for(std::uint32_t index = 0u; index < assembler.streamCount(); ++index)
                        {
                            if(assembler.stream(index).count != 0u)
                            {
                                if(partials == 0u)
                                    drainStream = index;
                                ++partials;
                            }
                        }
                        runtime.state.partialBuffers = partials;
                        action = partials == 0u ? 1u : 2u;
                    }
                }
            }
            // Reconverge after the leader-only observation before distributing
            // the decision through shuffles.
            runtime.barrier.barrier(
                runtime.acc,
                runtime.blockThreads,
                warp,
                barrier_id::quiescenceDecision,
                static_cast<std::uint32_t>(WorkerStage::retiring),
                0u);
            action = warp.template broadcast<std::uint32_t>(action, 0u);
            drainStream = warp.template broadcast<std::uint32_t>(drainStream, 0u);

            if(action == 3u)
            {
                if(leader)
                    requestStop(runtime, poolStoppingAborted);
                return true;
            }
            if(action == 1u)
            {
                if(leader)
                {
                    (void) exchangeDevice(runtime.acc, &runtime.state.inFlightReservations, std::uint64_t{0u});
                    (void) exchangeDevice(runtime.acc, &runtime.state.stopReason, poolStoppingDrained);
                }
                return true;
            }
            if(action == 2u)
            {
                Assembly assembler = makeAssembly(runtime);
                std::uint32_t const capacityBytes = runtime.args.chunkCapacity * runtime.args.packetSchema.elementSize;
                PacketHandle replacement{};
                bool const acquired = acquireBounded(runtime, replacement, 64u);
                if(!acquired)
                {
                    if(leader)
                        (void) exchangeDevice(runtime.acc, &runtime.state.inFlightReservations, std::uint64_t{0u});
                    return false;
                }
                configureChunkRecord(runtime, replacement);
                AssemblySlot published{};
                FlushStatus const status = assembler.flushPartial(
                    runtime.acc,
                    warp,
                    drainStream,
                    blockIndex + 1u,
                    makeAssemblySlot(replacement, capacityBytes),
                    published,
                    runtime.diagnostics);
                if(status == FlushStatus::Published)
                {
                    std::uint64_t seed = 0u;
                    if(leader)
                        seed = assembler.stream(drainStream).seed;
                    seed = warp.template broadcast<std::uint64_t>(seed, 0u);
                    std::uint32_t const elementCount = published.count / runtime.args.packetSchema.elementSize;
                    if(leader)
                    {
                        (void) addDevice(runtime.acc, &runtime.state.drainPublications, std::uint64_t{1u});
                        (void) addDevice(runtime.acc, &runtime.state.drainRounds, std::uint64_t{1u});
                    }
                    bool const ok = publishChunk(
                        runtime,
                        assembler.stream(drainStream).id,
                        published.handle,
                        elementCount,
                        seed,
                        static_cast<std::uint64_t>(drainStream) + 1u);
                    if(!ok)
                    {
                        if(leader)
                        {
                            recordError(runtime, PoolError::internalState);
                            requestStop(runtime, poolStoppingAborted);
                        }
                        return true;
                    }
                }
                else
                {
                    // Nothing was detached (count raced to zero) or the claim
                    // was busy: return the replacement to the pool.
                    bool const released = runtime.pool.release(runtime.acc, warp, replacement, runtime.diagnostics);
                    if(leader)
                    {
                        if(!released)
                            recordError(runtime, PoolError::internalState);
                        (void) addDevice(runtime.acc, &runtime.state.slotsReturned, std::uint64_t{1u});
                    }
                }
                if(leader)
                    (void) exchangeDevice(runtime.acc, &runtime.state.inFlightReservations, std::uint64_t{0u});
                return false;
            }
            if(leader)
                (void) exchangeDevice(runtime.acc, &runtime.state.inFlightReservations, std::uint64_t{0u});
            return false;
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
            std::uint64_t /*deadlineIterations*/,
            std::uint64_t iteration) noexcept
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
                bool const configured = runtime.pool.configure(runtime.acc, runtime.warp, runtime.args.budget, runtime.diagnostics);
                // The A/B chunk pairs come from the pool's free list, so this
                // runs after configure() and before any worker can claim work.
                bool const assembled = configured ? initializeAssembly(runtime) : false;
                // The pool configure already distributed `configured` through
                // its own handoff; the remaining startup flags follow through a
                // second handoff so no lane depends on a shuffle.
                WarpHandoff<Acc, StartupState, handoff_id::schedulerStartup> handoff{runtime.diagnostics};
                StartupState local{};
                if(runtime.warp.isLeader())
                {
                    runtime.state.debugConfigured = configured ? 1u : 0u;
                    // One initializer writes the launch marker exactly once.
                    runtime.state.launchCount = 1u;
                    runtime.state.initWinner = blockIndex;
                    local.configured = configured ? 1u : 0u;
                    local.bound = T_Binder::bind(runtime.acc, runtime.table) ? 1u : 0u;
                    local.assembled = assembled ? 1u : 0u;
                    local.ready = (local.configured != 0u && local.bound != 0u && local.assembled != 0u) ? 1u : 0u;
                    if(local.configured == 0u)
                        recordError(runtime, PoolError::internalState);
                    if(local.assembled == 0u)
                        recordError(runtime, PoolError::internalState);
                    if(local.bound == 0u)
                        recordError(runtime, PoolError::unknownOperation);
                }
                handoff.publish(runtime.acc, runtime.warp, local);
                handoff.sync(runtime.acc, handoff_id::schedulerStartup);
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
                runtime.barrier.barrier(
                    runtime.acc,
                    runtime.blockThreads,
                    runtime.warp,
                    barrier_id::schedulerStartupInit,
                    static_cast<std::uint32_t>(WorkerStage::waitInit),
                    iteration);
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
        ALPAKA_FN_ACC inline void importerStep(SchedulerRuntime<T_Acc>& runtime, std::uint64_t iteration) noexcept
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
                DispatchHandoff handoff{runtime.diagnostics};
                handoff.publish(runtime.acc, runtime.warp, local);
                handoff.sync(runtime.acc, handoff_id::schedulerImport);
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
                DispatchHandoff handoff{runtime.diagnostics};
                handoff.publish(runtime.acc, runtime.warp, local);
                handoff.sync(runtime.acc, handoff_id::schedulerImport);
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
            PollHandoff handoff{runtime.diagnostics};
            handoff.publish(runtime.acc, runtime.warp, poll);
            handoff.sync(runtime.acc, handoff_id::schedulerImport + 1u);
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

            // Outgoing data is not forwarded in place. `executeJob()` owns the
            // per-destination assembly: it appends the reported elements to
            // each stream and publishes every full chunk itself. Only the
            // resumable source range is handled here.
            return ok;
        }

        /** Execute one claimed job with every lane of the warp.
         *
         * The funnel ABI stays unchanged: the functor writes into the output
         * spans the accessor resolves and reports one warp-uniform element
         * count per port. The scheduler then owns per-destination chunk
         * assembly (design section 6):
         *
         * 1. read the reported count of every outgoing edge,
         * 2. reserve all destination ranges or none,
         * 3. copy the produced elements into the reserved logical range,
         * 4. commit the counter and rotate the A/B pair, publishing every full
         *    chunk as an ordinary ready job with its real element count,
         * 5. retire the input packet exactly once and return unused scratch
         *    slots to the pool.
         *
         * A failed reservation leaves every stream and counter unchanged and
         * the step retries without re-running the functor. A count that
         * exceeds the port capacity or names a port without storage is the
         * named `packetExceedsCapacity` error, never a silent over-read.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool executeJob(
            SchedulerRuntime<T_Acc>& runtime,
            JobDescriptor const& job,
            std::uint32_t blockIndex) noexcept
        {
            using Acc = typename SchedulerRuntime<T_Acc>::Acc;
            auto const& warp = runtime.warp;
            bool const leader = warp.isLeader();
            std::uint32_t const elementSize = runtime.args.packetSchema.elementSize;
            std::uint32_t const capacityBytes = runtime.args.chunkCapacity * elementSize;

            CompiledNode const* node = findNode(runtime.graph, job.operation);
            if(node == nullptr)
            {
                if(leader)
                    recordError(runtime, PoolError::unknownOperation);
                return false;
            }
            bool const isSource = node->isSource;
            if(!isSource && !packetHandleValid(runtime, job.packet))
                return false;

            std::uint32_t const ports
                = node->outputPortCount < maxPacketPorts - 1u ? node->outputPortCount : maxPacketPorts - 1u;
            PacketAccessor accessor{};
            accessor.clear();
            PacketHandle scratch[maxPacketPorts]{};
            bool hasScratch[maxPacketPorts]{};
            if(job.packet.index < static_cast<std::uint32_t>(poolPacketSlotCapacity))
            {
                PacketRecord const record = runtime.pool.observedRecord(runtime.acc, job.packet);
                void* data = runtime.pool.slotData(job.packet);
                if(isSource)
                {
                    // A source writes into the packet the importer acquired for
                    // its request range; that slot is its output scratch.
                    if(ports != 0u)
                    {
                        accessor.setOutput(0u, &record, data);
                        scratch[0] = job.packet;
                        hasScratch[0] = true;
                    }
                }
                else
                {
                    accessor.setInput(job.routing.inputPort, &record, data);
                }
            }
            else
            {
                if(leader)
                    recordError(runtime, PoolError::internalState);
                return false;
            }
            if(!isSource)
            {
                for(std::uint32_t port = 0u; port < ports; ++port)
                {
                    PacketHandle handle{};
                    if(!acquireWithRetry(runtime, handle))
                    {
                        if(leader)
                            recordError(runtime, PoolError::packetExhausted);
                        return false;
                    }
                    configureOutputRecord(runtime, handle);
                    scratch[port] = handle;
                    hasScratch[port] = true;
                    accessor.setOutput(port, &runtime.pool.slotRecord(handle), runtime.pool.slotData(handle));
                }
            }
            // Reserved control port: never a stream, never part of routing.
            accessor.inputData[maxPacketPorts - 1u] = runtime.args.controlBlock;

            OperationFn<Acc> const function = runtime.table.resolve(job.operation);
            if(function == nullptr)
            {
                if(leader)
                    recordError(runtime, PoolError::unknownOperation);
                return false;
            }
            function(runtime.acc, job, accessor);
            // The functor contains leader-only regions and lane-strided loops;
            // under CUDA's independent thread scheduling the warp can return
            // from the indirect call split. Converge before any lane reads a
            // per-lane count or runs a count-dependent loop, or the leader can
            // reach the next block barrier while other lanes are still in a
            // count-dependent loop.
            runtime.barrier.barrier(
                runtime.acc,
                runtime.blockThreads,
                warp,
                barrier_id::schedulerExecute,
                static_cast<std::uint32_t>(WorkerStage::executing),
                0u);
            if(leader && job.operation.value < 8u)
                (void) addDevice(runtime.acc, &runtime.state.execCounts[job.operation.value], std::uint64_t{1u});

            // Read the reported counts and build one append request per
            // outgoing edge. Each output port maps to exactly one destination
            // stream; a duplicate would append the same bytes twice.
            Assembly assembler = makeAssembly(runtime);
            AppendRequest requests[maxAssemblyTargets]{};
            std::uint32_t requestPorts[maxAssemblyTargets]{};
            AssemblySlot replacements[maxAssemblyTargets]{};
            std::uint32_t requestCount = 0u;
            for(std::uint32_t index = 0u; index < node->outputEdgeCount; ++index)
            {
                std::uint32_t const edgeIndex = runtime.graph.outputEdges[node->firstOutputEdge + index];
                CompiledEdge const& edge = runtime.graph.edges[edgeIndex];
                std::uint32_t const port = edge.outputPort;
                if(port >= ports || !hasScratch[port] || accessor.outputRecord(port) == nullptr)
                {
                    if(leader)
                        recordError(runtime, PoolError::packetExceedsCapacity);
                    return false;
                }
                // The functor reports a warp-uniform count; broadcast it so
                // every lane uses the same value even if a backend keeps the
                // mutable accessor copy in per-lane local memory.
                std::uint32_t const count = warp.template broadcast<std::uint32_t>(accessor.outputCount(port), 0u);
                if(count == 0u)
                    continue;
                if(count > runtime.args.chunkCapacity || requestCount >= maxAssemblyTargets)
                {
                    if(leader)
                        recordError(runtime, PoolError::packetExceedsCapacity);
                    return false;
                }
                for(std::uint32_t other = 0u; other < requestCount; ++other)
                {
                    if(requests[other].stream == edge.stream)
                    {
                        if(leader)
                            recordError(runtime, PoolError::internalState);
                        return false;
                    }
                }
                requests[requestCount] = AppendRequest{edge.stream, count * elementSize};
                requestPorts[requestCount] = port;
                replacements[requestCount] = makeAssemblySlot(scratch[port], capacityBytes);
                ++requestCount;
            }

            // Dedicated replacement chunks: one per destination target. The
            // functor output scratch is copied into the reserved range and is
            // then always returned to the pool; it never becomes a stream
            // chunk. This keeps the two ownership roles (functor scratch vs.
            // stream chunk) disjoint, so a released scratch can never be a
            // chunk another job already owns.
            for(std::uint32_t request = 0u; request < requestCount; ++request)
            {
                PacketHandle replacement{};
                if(!acquireWithRetry(runtime, replacement))
                {
                    if(leader)
                        recordError(runtime, PoolError::packetExhausted);
                    return false;
                }
                configureChunkRecord(runtime, replacement);
                replacements[request] = makeAssemblySlot(replacement, capacityBytes);
            }
            AssemblySlot publications[maxAssemblyTargets]{};
            AssemblyReservation reservation{};
            bool committed = false;
            constexpr std::uint32_t maxReserveAttempts = 1u << 16u;
            for(std::uint32_t attempt = 0u; !committed && attempt < maxReserveAttempts; ++attempt)
            {
                reservation = AssemblyReservation{};
                AppendStatus const status = reservation.tryReserve(
                    runtime.acc,
                    warp,
                    blockIndex + 1u,
                    assembler,
                    requests,
                    requestCount,
                    replacements,
                    requestCount,
                    publications,
                    maxAssemblyTargets,
                    runtime.diagnostics);
                if(status == AppendStatus::StreamBusy)
                {
                    devicePause(runtime.args.idleNanosleep);
                    continue;
                }
                if(status != AppendStatus::Ok)
                {
                    if(leader)
                        recordError(runtime, PoolError::internalState);
                    return false;
                }

                bool copyOk = true;
                std::uint32_t const width = warp.workerWidth();
                for(std::uint32_t target = 0u; target < requestCount; ++target)
                {
                    auto* source = static_cast<AssemblyElement*>(accessor.outputData[requestPorts[target]]);
                    // A uniform round count keeps every lane in the same loop
                    // iteration. A lane-strided `byte < itemCount` loop can
                    // leave lanes at different program points under CUDA's
                    // independent thread scheduling, and the following
                    // block barrier would then wait for lanes that never
                    // arrive.
                    std::uint32_t const rounds = (requests[target].itemCount + width - 1u) / width;
                    for(std::uint32_t round = 0u; round < rounds; ++round)
                    {
                        std::uint32_t const byte = round * width + warp.laneIndex();
                        if(byte >= requests[target].itemCount)
                            continue;
                        AssemblyElement* destination = reservation.writePointer(target, byte);
                        if(destination == nullptr)
                            copyOk = false;
                        else
                            *destination = source[byte];
                    }
                }
                if(!copyOk)
                {
                    reservation.cancel(runtime.acc, warp, runtime.diagnostics);
                    if(leader)
                        recordError(runtime, PoolError::packetExceedsCapacity);
                    return false;
                }
                if(!reservation.commit(runtime.acc, warp, runtime.diagnostics))
                {
                    // A competing writer committed in between; the reservation
                    // released every claim and mutated nothing. Retry.
                    devicePause(runtime.args.idleNanosleep);
                    continue;
                }
                committed = true;

                std::uint32_t publicationCount = 0u;
                for(std::uint32_t target = 0u; target < requestCount; ++target)
                    publicationCount += reservation.rotates(target) ? 1u : 0u;
                std::uint32_t cursor = 0u;
                for(std::uint32_t target = 0u; target < requestCount; ++target)
                {
                    if(!reservation.rotates(target))
                        continue;
                    std::uint32_t slot = 0u;
                    std::uint32_t count = 0u;
                    std::uint32_t handleIndex = 0u;
                    std::uint32_t handleGeneration = 0u;
                    if(leader && cursor < publicationCount)
                    {
                        slot = publications[cursor].slot;
                        count = publications[cursor].count;
                        handleIndex = publications[cursor].handle.index;
                        handleGeneration = publications[cursor].handle.generation;
                    }
                    slot = warp.template broadcast<std::uint32_t>(slot, 0u);
                    count = warp.template broadcast<std::uint32_t>(count, 0u);
                    handleIndex = warp.template broadcast<std::uint32_t>(handleIndex, 0u);
                    handleGeneration = warp.template broadcast<std::uint32_t>(handleGeneration, 0u);
                    ++cursor;
                    PacketHandle const publishedHandle{handleIndex, handleGeneration};
                    std::uint32_t const elementCount = count / elementSize;
                    if(!publishChunk(
                           runtime,
                           requests[target].stream,
                           publishedHandle,
                           elementCount,
                           job.request.seed,
                           job.sequenceNumber + 1u))
                    {
                        if(leader)
                            recordError(runtime, PoolError::internalState);
                        return false;
                    }
                }
                // Remember the request seed of every destination stream so a
                // drain-published partial chunk keeps the deterministic stream
                // key of the run.
                if(leader)
                {
                    for(std::uint32_t target = 0u; target < requestCount; ++target)
                        assembler.stream(reservation.target(target).streamIndex).seed = job.request.seed;
                }
            }
            if(!committed)
            {
                if(leader)
                    recordError(runtime, PoolError::internalState);
                return false;
            }

            // Retire exactly once. All functor scratch slots are released;
            // the replacement chunks consumed by a rotation stay with their
            // stream, and unused replacements return to the pool. The input
            // packet of a non-source job is released separately; for a source
            // its output scratch is that same packet.
            std::uint32_t rotates = 0u;
            for(std::uint32_t target = 0u; target < requestCount; ++target)
                rotates += reservation.rotates(target) ? 1u : 0u;
            for(std::uint32_t port = 0u; port < ports; ++port)
            {
                if(!hasScratch[port])
                    continue;
                bool const released = runtime.pool.release(runtime.acc, warp, scratch[port], runtime.diagnostics);
                if(leader)
                {
                    if(!released)
                        recordError(runtime, PoolError::internalState);
                    (void) addDevice(runtime.acc, &runtime.state.slotsReturned, std::uint64_t{1u});
                    (void) addDevice(runtime.acc, &runtime.state.retiredPackets, std::uint64_t{1u});
                }
            }
            for(std::uint32_t replacement = rotates; replacement < requestCount; ++replacement)
            {
                bool const released = runtime.pool.release(runtime.acc, warp, replacements[replacement].handle, runtime.diagnostics);
                if(leader)
                {
                    if(!released)
                        recordError(runtime, PoolError::internalState);
                    (void) addDevice(runtime.acc, &runtime.state.slotsReturned, std::uint64_t{1u});
                }
            }
            if(!isSource)
            {
                bool const released = runtime.pool.release(runtime.acc, warp, job.packet, runtime.diagnostics);
                if(leader)
                {
                    if(!released)
                        recordError(runtime, PoolError::internalState);
                    (void) addDevice(runtime.acc, &runtime.state.slotsReturned, std::uint64_t{1u});
                    (void) addDevice(runtime.acc, &runtime.state.retiredPackets, std::uint64_t{1u});
                }
            }
            return true;
        }

        /** The persistent scheduling loop of one worker warp. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void schedulerLoop(SchedulerRuntime<T_Acc>& runtime, std::uint32_t blockIndex) noexcept
        {
            std::uint64_t idleIterations = 0u;
            std::uint64_t iteration = 0u;
            while(true)
            {
                ++iteration;
                recordStage(runtime, blockIndex, WorkerStage::pollReady);
                if(loadDeviceAcquire(runtime.acc, &runtime.state.stopReason) != 0u)
                {
                    recordStage(runtime, blockIndex, WorkerStage::exiting);
                    // All lanes exit together: the stop reason is warp-uniform
                    // (device-scope read, same value for every lane in the
                    // warp). Exiting only the leader would leave the other
                    // lanes in the loop, and their next barrier (quiescence,
                    // assembly, or handoff) would see a divergent warp under
                    // CUDA's independent thread scheduling.
                    return;
                }

                // Every worker honors the mapped host abort word, not only the
                // importer block. A diagnostic abort must stop a stuck idle
                // worker as well, and the read is warp-uniform on every lane.
                if(hostAbortRequested(runtime))
                {
                    recordStage(runtime, blockIndex, WorkerStage::exiting);
                    if(runtime.warp.isLeader())
                    {
                        recordError(runtime, PoolError::aborted);
                        requestStop(runtime, poolStoppingAborted, 6u);
                    }
                    return;
                }

                bool didWork = false;
                if(blockIndex == 0u)
                {
                    recordStage(runtime, blockIndex, WorkerStage::importerStep);
                    importerStep(runtime, iteration);
                    if(loadDeviceAcquire(runtime.acc, &runtime.state.stopReason) != 0u)
                    {
                        recordStage(runtime, blockIndex, WorkerStage::exiting);
                        return;
                    }
                }

                JobDescriptor job{};
                RingStatus const status = runtime.ring.dequeue(runtime.acc, runtime.warp, job, runtime.diagnostics);
                if(status == RingStatus::Accepted)
                {
                    idleIterations = 0u;
                    recordStage(runtime, blockIndex, WorkerStage::executing, WorkerLastAction::claim);
                    if(runtime.warp.isLeader())
                    {
                        (void) addDevice(runtime.acc, &runtime.state.active, std::uint64_t{1u});
                        (void) addDevice(runtime.acc, &runtime.state.claimAttempts, std::uint64_t{1u});
                        std::uint32_t const word = blockIndex / 64u;
                        std::uint64_t const bit = std::uint64_t{1u} << (blockIndex % 64u);
                        (void) orDevice(runtime.acc, &runtime.state.workerBits[word], bit);
                    }
                    // Order the leader's acquire before any lane reads packet data.
                    runtime.barrier.barrier(
                        runtime.acc,
                        runtime.blockThreads,
                        runtime.warp,
                        barrier_id::schedulerClaim,
                        static_cast<std::uint32_t>(WorkerStage::pollReady),
                        iteration);
                    bool const executed = executeJob(runtime, job, blockIndex);
                    recordStage(
                        runtime,
                        blockIndex,
                        WorkerStage::publishing,
                        executed ? WorkerLastAction::execute : WorkerLastAction::none);
                    // Order every lane's stores before the leader publishes or retires.
                    runtime.barrier.barrier(
                        runtime.acc,
                        runtime.blockThreads,
                        runtime.warp,
                        barrier_id::schedulerPublish,
                        static_cast<std::uint32_t>(WorkerStage::publishing),
                        iteration);
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
                    recordStage(runtime, blockIndex, WorkerStage::backoff);
                    devicePause(runtime.args.idleNanosleep);
                }
                else
                {
                    ++idleIterations;
                    if(idleIterations >= 4u)
                    {
                        recordStage(runtime, blockIndex, WorkerStage::retiring, WorkerLastAction::quiescence);
                        if(quiescenceStep(runtime, blockIndex))
                        {
                            recordStage(runtime, blockIndex, WorkerStage::exiting);
                            return;
                        }
                    }
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
                // Barrier divergence diagnostics for this launch. The state
                // block is owned by the pool and zeroed (with sentinel first
                // fields) before the launch; the flags mirror PoolConfig.
                auto* diagState = static_cast<BarrierDiagnosticsState*>(args.barrierDiagnostics);
                BarrierDiagnosticsConfig diagConfig;
                diagConfig.enabled = args.barrierDiagnosticsEnabled != 0u;
                diagConfig.skipOnDivergence = args.barrierSkipOnDivergence != 0u;
                auto const diagnostics = WarpBarrierDiagnostics<Acc>::make(diagState, diagConfig);
                WarpBarrier<Acc> const barrier{diagnostics};
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
                    diagnostics,
                    barrier,
                    ring,
                    pool,
                    table,
                    *static_cast<CompiledGraph const*>(args.graph),
                    *static_cast<PoolDeviceState*>(args.state),
                    *args.hostControl,
                    args,
                    blockThreads};

                // `iteration` counts the scheduling-loop iterations of this
                // warp; it is published with every stage record and barrier
                // divergence record, so a first divergence names the loop
                // position of the warp that first reached the barrier
                // divergent.
                std::uint64_t iteration = 0u;
                bool const started = startupRendezvous<T_Binder>(
                    runtime,
                    blockIndex,
                    args.initDeadlineCycles,
                    args.initDeadlineIterations,
                    iteration);
                if(!started)
                    return;
                schedulerLoop(runtime, blockIndex);
            }
        };
    } // namespace detail
} // namespace graph
