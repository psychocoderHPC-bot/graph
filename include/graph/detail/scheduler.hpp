// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/pool_control.hpp>
#include <graph/detail/rendezvous.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
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
            WarpBarrier<Acc> const& barrier;
            Ring& ring;
            Pool& pool;
            Table& table;
            CompiledGraph const& graph;
            PoolDeviceState& state;
            PoolHostControl& host;
            PoolKernelArgs const& args;
            std::uint32_t blockIndex;
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

        /** Warp-uniform broadcast of a 32-bit value from the leader. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline std::uint32_t broadcastU32(
            SchedulerRuntime<T_Acc> const& runtime,
            std::uint32_t value) noexcept
        {
            return runtime.warp.template broadcast<std::uint32_t>(value, 0u);
        }

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
            PacketRecord const& record = runtime.pool.slotRecord(handle);
            // Ownership is carried by the generation: the slot is owned by this
            // handle iff the pool's current generation for that index equals the
            // generation broadcast with the descriptor. `elementCount` is not an
            // ownership signal: under a deliberately small pool a valid in-flight
            // job can be claimed while its payload is not yet fully configured,
            // so a zero elementCount must not drop the job.
            bool const matches = record.handle.generation == handle.generation;
            if(!matches && runtime.warp.isLeader())
                (void) addDevice(runtime.acc, &runtime.state.readyViolations, std::uint64_t{1u});
            return matches;
        }

        /** Publish one ready entry, retrying a transient lock contention.
         *
         * The ring lock is held only for a short critical section and is never
         * held across a collective, so retrying the warp-uniform attempt is
         * safe. A `Full` result is a hard capacity failure.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline RingStatus enqueueWithRetry(
            SchedulerRuntime<T_Acc>& runtime,
            JobDescriptor const& entry) noexcept
        {
            constexpr std::uint32_t maxAttempts = 4096u;
            RingStatus status = RingStatus::Contended;
            for(std::uint32_t attempt = 0u; attempt < maxAttempts; ++attempt)
            {
                status = runtime.ring.enqueue(runtime.acc, runtime.warp, entry);
                if(status != RingStatus::Contended)
                    return status;
                devicePause(runtime.args.idleNanosleep);
            }
            return status;
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
            std::uint64_t deadlineCycles,
            std::uint64_t /*deadlineIterations*/) noexcept
        {
            // Block zero is the initializer. The launch is sized to one
            // resident wave, and the deadline below turns a non-resident
            // initializer into a reported startup failure instead of a hang.
            bool const winner = runtime.blockIndex == 0u;

            bool ready = true;
            if(winner)
            {
                if(runtime.warp.isLeader())
                    runtime.state.initLock = 1u;
                runtime.ring.initialize(runtime.acc, runtime.warp);
                if(runtime.warp.isLeader())
                {
                    runtime.state.debugRequiredSlots = requiredSlots(runtime.args.budget);
                    runtime.state.debugBudgetWorkers = runtime.args.budget.workers;
                }
                bool configured = runtime.pool.configure(runtime.acc, runtime.warp, runtime.args.budget);
                bool bound = false;
                if(runtime.warp.isLeader())
                {
                    runtime.state.debugConfigured = configured ? 1u : 0u;
                    // One initializer writes the launch marker exactly once.
                    runtime.state.launchCount = 1u;
                    runtime.state.initWinner = runtime.blockIndex;
                    bound = T_Binder::bind(runtime.acc, runtime.table);
                }
                configured = broadcastU32(runtime, configured ? 1u : 0u) != 0u;
                bound = broadcastU32(runtime, bound ? 1u : 0u) != 0u;
                if(runtime.warp.isLeader())
                {
                    if(!configured)
                        recordError(runtime, PoolError::internalState);
                    if(!bound)
                        recordError(runtime, PoolError::unknownOperation);
                    ready = configured && bound;
                }
                ready = broadcastU32(runtime, ready ? 1u : 0u) != 0u;
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
                (void) deadlineCycles;
                auto* initFlag = reinterpret_cast<std::uint64_t volatile*>(&runtime.state.initDone);
                auto const cpuDeadline
                    = std::chrono::steady_clock::now() + std::chrono::milliseconds{static_cast<std::int64_t>(5000)};
                constexpr std::uint64_t maxCudaSpinIterations = 20'000'000u;
                constexpr std::uint64_t maxCpuSpinIterations = 200'000'000u;
                std::uint64_t iterations = 0u;
                while(*initFlag == 0u)
                {
                    if constexpr(std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Cuda>)
                    {
                        if(++iterations > maxCudaSpinIterations)
                            break;
                    }
                    else
                    {
                        if(std::chrono::steady_clock::now() > cpuDeadline || ++iterations > maxCpuSpinIterations)
                            break;
                    }
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

        /** Give ingress a bounded opportunity and dispatch already staged jobs. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void importerStep(SchedulerRuntime<T_Acc>& runtime) noexcept
        {
            if(runtime.warp.isLeader())
            {
                std::uint64_t const polls = addDevice(runtime.acc, &runtime.state.debugPolls, std::uint64_t{1u}) + 1u;
                graph::systemStoreRelease(runtime.acc, &runtime.host.polls, polls);
            }
            // Dispatch staged descriptors first: the importer acknowledged them
            // already, so they are accounted as outstanding.
            for(std::uint32_t dispatched = 0u; dispatched < runtime.args.importPerIteration; ++dispatched)
            {
                std::uint32_t count = 0u;
                std::uint32_t head = 0u;
                if(runtime.warp.isLeader())
                {
                    count = runtime.state.stagingCount;
                    head = runtime.state.stagingHead;
                }
                count = broadcastU32(runtime, count);
                head = broadcastU32(runtime, head);
                if(count == 0u)
                    break;

                bool hasRoom = false;
                if(runtime.warp.isLeader())
                    hasRoom = runtime.ring.inFlightCount() < runtime.args.readySoftLimit;
                hasRoom = broadcastU32(runtime, hasRoom ? 1u : 0u) != 0u;
                if(!hasRoom)
                    break;

                JobDescriptor const staged = runtime.args.staging[head];
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
                        recordError(runtime, PoolError::readyRingFull);
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

            std::uint32_t staged = 0u;
            if(runtime.warp.isLeader())
                staged = runtime.state.stagingCount;
            staged = broadcastU32(runtime, staged);
            if(staged != 0u)
                return;

            bool hasRoom = false;
            if(runtime.warp.isLeader())
                hasRoom = runtime.ring.inFlightCount() < runtime.args.readySoftLimit;
            hasRoom = broadcastU32(runtime, hasRoom ? 1u : 0u) != 0u;
            if(!hasRoom)
                return;

            // One importer lane polls; the resolver maps ids to device pointers.
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
                if(batch.status == ImportStatus::error)
                {
                    recordError(runtime, PoolError::unknownOperation);
                    requestStop(runtime, poolStoppingAborted);
                    return;
                }
                if(batch.imported == 0u)
                    return;

                std::uint32_t jobs = 0u;
                for(std::uint32_t index = 0u; index < batch.imported; ++index)
                {
                    JobDescriptor const& descriptor = runtime.args.staging[index];
                    if(descriptor.kind == JobKind::END_OF_INPUT)
                        continue;
                    ++jobs;
                    bool const overflow = descriptor.request.count > ~std::uint64_t{0u} - descriptor.request.firstId;
                    if(overflow)
                        recordError(runtime, PoolError::requestOverflow);
                    if(runtime.args.idLimit != 0u && !overflow
                       && descriptor.request.firstId + descriptor.request.count
                              > static_cast<std::uint64_t>(runtime.args.idLimit))
                        recordError(runtime, PoolError::idOutOfRange);
                }
                if(loadDeviceRelaxed(runtime.acc, &runtime.state.errorWord) != 0u)
                {
                    requestStop(runtime, poolStoppingAborted);
                    return;
                }
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
            runtime.barrier.barrier(runtime.acc, runtime.blockThreads, runtime.warp);
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
                                recordError(runtime, PoolError::readyRingFull);
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
                        recordError(runtime, PoolError::readyRingFull);
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

        /** Execute one claimed job with every lane of the warp. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline bool executeJob(SchedulerRuntime<T_Acc>& runtime, JobDescriptor const& job) noexcept
        {
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
                PacketRecord const& record = runtime.pool.slotRecord(job.packet);
                void* data = runtime.pool.slotData(job.packet);
                if(!isSource)
                    accessor.setInput(job.routing.inputPort, &record, data);
                for(std::uint32_t port = 0u; port < job.routing.outputPortCount && port + 1u < maxPacketPorts; ++port)
                    accessor.setOutput(port, &record, data);
            }
            // Reserved control port: never a stream, never part of routing.
            accessor.inputData[maxPacketPorts - 1u] = runtime.args.controlBlock;

            OperationFn<typename SchedulerRuntime<T_Acc>::Acc> const function = runtime.table.resolve(job.operation);
            if(function == nullptr)
            {
                if(runtime.warp.isLeader())
                    recordError(runtime, PoolError::unknownOperation);
                return false;
            }
            function(runtime.acc, job, accessor);
            if(runtime.warp.isLeader() && job.operation.value < 8u)
                (void) addDevice(runtime.acc, &runtime.state.execCounts[job.operation.value], std::uint64_t{1u});
            return true;
        }

        /** The persistent scheduling loop of one worker warp. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline void schedulerLoop(SchedulerRuntime<T_Acc>& runtime) noexcept
        {
            std::uint64_t idleIterations = 0u;
            while(true)
            {
                if(loadDeviceAcquire(runtime.acc, &runtime.state.stopReason) != 0u)
                    return;

                bool didWork = false;
                if(runtime.blockIndex == 0u)
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
                        std::uint32_t const word = runtime.blockIndex / 64u;
                        std::uint64_t const bit = std::uint64_t{1u} << (runtime.blockIndex % 64u);
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
                            std::uint64_t const local = ++runtime.state.workerRetired[runtime.blockIndex];
                            graph::systemStoreRelease(
                                runtime.acc,
                                &runtime.host.workerRetired[runtime.blockIndex],
                                local);
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
                    blockIndex,
                    blockThreads};

                bool const started
                    = startupRendezvous<T_Binder>(runtime, args.initDeadlineCycles, args.initDeadlineIterations);
                if(!started)
                    return;
                schedulerLoop(runtime);
            }
        };
    } // namespace detail
} // namespace graph
