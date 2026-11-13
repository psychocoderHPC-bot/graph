// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>
#include <graph/detail/copy_queue.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/launch.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/pool_control.hpp>
#include <graph/detail/scheduler.hpp>
#include <graph/graph_model.hpp>
#include <graph/ingress_ring.hpp>
#include <graph/packet_pool.hpp>
#include <graph/ready_ring.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

/** @file
 *
 * Host-facing persistent warp pool.
 *
 * `startPool()` returns a `Pool` that owns the worker queue, the dedicated copy
 * queue, every allocation (ingress ring, ready ring, packet pool, device
 * scheduler state, operation table, graph tables, staging), and the lifetime of
 * the single persistent worker kernel. All device-side work is described in
 * `graph/detail/scheduler.hpp`; this header is the orchestration layer.
 *
 * # Threading
 *
 * The input side is single-producer. One host thread at a time may call
 * `tryEnqueue`, `enqueue`, `closeInput`, and `abort`. `wait`, `waitFor`,
 * `completedJobs`, `quiescenceSnapshot`, and the copy helpers may be called from
 * another thread.
 *
 * # Never started
 *
 * A default-constructed `Pool` owns no backend. Its `wait()` reports
 * `PoolStopReason::neverStarted` and submissions report `closed`.
 */
namespace graph
{
    namespace detail
    {
        /** Type-erased host backend of a pool. */
        class PoolBackend
        {
        public:
            virtual ~PoolBackend() = default;

            virtual PoolEnqueueResult tryEnqueue(std::span<RequestSpec const> requests) = 0;
            virtual PoolEnqueueResult enqueueBlocking(
                std::span<RequestSpec const> requests,
                std::chrono::nanoseconds timeout)
                = 0;
            virtual PoolEnqueueStatus closeInput(std::chrono::nanoseconds timeout) = 0;
            virtual void abort() = 0;
            virtual PoolResult wait() = 0;
            virtual std::optional<PoolResult> waitFor(std::chrono::nanoseconds timeout) = 0;
            virtual LaunchGeometry geometry() const = 0;
            virtual QuiescenceSnapshot quiescenceSnapshot() = 0;
            virtual void copyResultsToHost(void* pinnedDestination, std::size_t bytes) = 0;
            virtual void waitCopies() = 0;
            virtual std::uint64_t completedJobs() const = 0;
            virtual PoolConfig const& config() const = 0;
        };

        /** Binder used when the caller does not register device operations.
         *
         * A pool started with it can be shut down but cannot execute jobs.
         */
        struct NullBinder
        {
            template<typename T_Acc>
            ALPAKA_FN_ACC static bool bind(T_Acc const&, OperationTable<T_Acc>&) noexcept
            {
                return true;
            }
        };

        /** Build the host descriptor of one source request. */
        ALPAKA_FN_HOST inline JobDescriptor makeRequestJob(
            RequestSpec const& request,
            OperationId sourceOperation,
            std::uint64_t sequenceNumber) noexcept
        {
            JobDescriptor job{};
            job.operation = sourceOperation;
            job.kind = JobKind::OPERATION;
            job.packet.index = invalidSlotIndex;
            job.routing.inputPort = 0u;
            job.routing.outputPort = 0u;
            job.routing.outputPortCount = 1u;
            job.request = request;
            job.sequenceNumber = sequenceNumber;
            return job;
        }

        /** Effective soft cap of importer-published ready entries.
         *
         * The cap never exceeds the hard ring capacity minus the per-worker
         * reservation (`3 * workerCount`), so the importer cannot fill the ring
         * past the space the workers need for their own publications. The
         * per-worker reservation covers at most one in-flight job, one
         * continuation, and one successor.
         */
        ALPAKA_FN_HOST inline std::uint32_t derivedReadySoftLimit(
            PoolConfig const& config,
            LaunchGeometry const& geometry) noexcept
        {
            std::uint64_t const hard = static_cast<std::uint64_t>(poolReadyCapacity);
            std::uint64_t const reserved = 3u * static_cast<std::uint64_t>(geometry.workerCount);
            std::uint64_t const readyLimit = hard > reserved ? hard - reserved : 0u;
            std::uint64_t const limit = config.maxReadyInFlight == 0u
                                            ? readyLimit
                                            : (static_cast<std::uint64_t>(config.maxReadyInFlight) < readyLimit
                                                   ? static_cast<std::uint64_t>(config.maxReadyInFlight)
                                                   : readyLimit);
            return static_cast<std::uint32_t>(limit);
        }

        /** Detached, kernel-free backend for host API validation.
         *
         * It owns a CPU ingress ring and a tiny host state machine. It exists so
         * the host-only test can exercise submission status, ordered close, and
         * the abort/drain result distinction without launching a worker kernel.
         */
        class DetachedPoolBackend final : public PoolBackend
        {
        public:
            explicit DetachedPoolBackend(PoolConfig config) : m_config(std::move(config))
            {
                auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
                auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
                if(selector.getDeviceCount() == 0u)
                    throw std::runtime_error("graph: no host device for the detached pool backend");
                auto device = selector.makeDevice(0);
                m_ingress.emplace(IngressRing<JobDescriptor, poolIngressCapacity>::create(device));
            }

            PoolEnqueueResult tryEnqueue(std::span<RequestSpec const> requests) override
            {
                PoolEnqueueResult result;
                result.requested = static_cast<std::uint32_t>(requests.size());
                if(m_aborted)
                {
                    result.status = PoolEnqueueStatus::closed;
                    return result;
                }
                std::vector<JobDescriptor> descriptors;
                descriptors.reserve(requests.size());
                for(auto const& request : requests)
                    descriptors.push_back(makeRequestJob(request, m_config.sourceOperation, m_sequence++));
                auto const submitted = m_ingress->tryEnqueue(descriptors);
                result.accepted = submitted.accepted;
                switch(submitted.status)
                {
                case EnqueueStatus::accepted:
                    result.status = PoolEnqueueStatus::accepted;
                    break;
                case EnqueueStatus::ring_full:
                    result.status = PoolEnqueueStatus::full;
                    break;
                case EnqueueStatus::closed:
                    result.status = PoolEnqueueStatus::closed;
                    break;
                case EnqueueStatus::error:
                    result.status = PoolEnqueueStatus::error;
                    break;
                }
                m_accepted += submitted.accepted;
                return result;
            }

            PoolEnqueueResult enqueueBlocking(std::span<RequestSpec const> requests, std::chrono::nanoseconds timeout)
                override
            {
                auto const deadline = std::chrono::steady_clock::now() + timeout;
                PoolEnqueueResult total;
                total.requested = static_cast<std::uint32_t>(requests.size());
                std::size_t offset = 0u;
                while(offset < requests.size())
                {
                    auto const part = tryEnqueue(requests.subspan(offset));
                    if(part.status == PoolEnqueueStatus::closed || part.status == PoolEnqueueStatus::error)
                    {
                        total.status = part.status;
                        return total;
                    }
                    total.accepted += part.accepted;
                    offset += part.accepted;
                    if(offset >= requests.size())
                        break;
                    if(std::chrono::steady_clock::now() >= deadline)
                    {
                        total.status = PoolEnqueueStatus::full;
                        return total;
                    }
                    std::this_thread::yield();
                }
                total.status = PoolEnqueueStatus::accepted;
                return total;
            }

            PoolEnqueueStatus closeInput(std::chrono::nanoseconds timeout) override
            {
                auto const deadline = std::chrono::steady_clock::now() + timeout;
                while(true)
                {
                    auto const status = m_ingress->closeInput();
                    if(status.status == EnqueueStatus::accepted)
                    {
                        m_closed = true;
                        return PoolEnqueueStatus::accepted;
                    }
                    if(status.status == EnqueueStatus::closed)
                    {
                        m_closed = true;
                        return PoolEnqueueStatus::closed;
                    }
                    if(status.status == EnqueueStatus::error)
                        return PoolEnqueueStatus::error;
                    if(std::chrono::steady_clock::now() >= deadline)
                        return PoolEnqueueStatus::full;
                    std::this_thread::yield();
                }
            }

            void abort() override
            {
                m_aborted = true;
            }

            PoolResult wait() override
            {
                return result();
            }

            std::optional<PoolResult> waitFor(std::chrono::nanoseconds) override
            {
                return result();
            }

            LaunchGeometry geometry() const override
            {
                return LaunchGeometry{};
            }

            QuiescenceSnapshot quiescenceSnapshot() override
            {
                QuiescenceSnapshot snapshot;
                snapshot.accepted = m_accepted;
                snapshot.imported = m_ingress->consumed();
                snapshot.readyCapacity = static_cast<std::uint32_t>(poolIngressCapacity);
                return snapshot;
            }

            void copyResultsToHost(void*, std::size_t) override
            {
                throw std::logic_error("graph: the detached pool backend has no result buffer");
            }

            void waitCopies() override
            {
            }

            std::uint64_t completedJobs() const override
            {
                return m_ingress->consumed();
            }

            PoolConfig const& config() const override
            {
                return m_config;
            }

        private:
            PoolResult result() const
            {
                PoolResult res;
                res.reason = m_aborted ? PoolStopReason::stoppedByAbort
                                       : (m_closed ? PoolStopReason::stoppedByDrain : PoolStopReason::neverStarted);
                if(m_aborted)
                    res.errorWord = poolErrorBits(PoolError::aborted);
                res.counters.accepted = m_accepted;
                res.counters.imported = m_ingress->consumed();
                return res;
            }

            PoolConfig m_config;
            std::optional<IngressRing<JobDescriptor, poolIngressCapacity>> m_ingress;
            std::uint64_t m_sequence{0u};
            std::uint64_t m_accepted{0u};
            bool m_closed{false};
            bool m_aborted{false};
        };

        template<typename T_Device, typename T_Executor, typename T_Binder>
        class PersistentPoolBackend final : public PoolBackend
        {
        public:
            using Device = std::remove_cvref_t<T_Device>;
            using QueueType = decltype(std::declval<Device&>().makeQueue());
            using EventType = decltype(std::declval<QueueType const&>().makeEvent());
            using CopyToken = typename CopyQueue<Device>::Token;
            using ByteBuffer = decltype(alpaka::onHost::alloc<std::uint8_t>(std::declval<Device>(), std::size_t{1}));
            using HostByteBuffer = decltype(alpaka::onHost::allocHost<std::uint8_t>(std::size_t{1}));
            using StateBuffer = decltype(alpaka::onHost::alloc<PoolDeviceState>(std::declval<Device>(), 1u));
            using HostStateBuffer = decltype(alpaka::onHost::allocHost<PoolDeviceState>(1u));
            using DiagBuffer = decltype(alpaka::onHost::alloc<BarrierDiagnosticsState>(std::declval<Device>(), 1u));
            using HostDiagBuffer = decltype(alpaka::onHost::allocHost<BarrierDiagnosticsState>(1u));
            using HostControlBuffer
                = decltype(alpaka::onHost::allocMapped<PoolHostControl>(std::declval<Device>(), 1u));
            using TableBuffer = decltype(alpaka::onHost::alloc<OperationTableLayout>(std::declval<Device>(), 1u));
            using GraphBuffer = decltype(alpaka::onHost::alloc<CompiledGraph>(std::declval<Device>(), 1u));
            using HostGraphBuffer = decltype(alpaka::onHost::allocHost<CompiledGraph>(1u));
            using HostTableBuffer = decltype(alpaka::onHost::allocHost<OperationTableLayout>(1u));
            using StagingBuffer = decltype(alpaka::onHost::alloc<JobDescriptor>(
                std::declval<Device>(),
                static_cast<std::uint32_t>(poolStagingCapacity)));
            using AssemblyState = AssemblyStream<std::uint8_t>;

            PersistentPoolBackend(
                Device device,
                T_Executor executor,
                CompiledGraph const& graph,
                PoolConfig config,
                T_Binder binder)
                : m_device(std::move(device))
                , m_executor(executor)
                , m_config(std::move(config))
                , m_graph(graph)
                , m_binder(binder)
                , m_copyQueue(m_device)
                , m_queue(m_device.makeQueue())
            {
                static_cast<void>(m_binder);
                // Every compiled stream is assembled by the scheduler. The
                // pool has one element schema, so all streams must share the
                // configured chunk capacity; the assembler arena is one
                // contiguous byte area with that logical capacity per chunk.
                for(std::uint32_t index = 0u; index < m_graph.streamCount; ++index)
                {
                    if(m_graph.streams[index].chunkCapacity != m_config.chunkCapacity)
                        throw std::invalid_argument(
                            "graph::startPool: compiled stream " + std::to_string(index)
                            + " has a chunk capacity different from the pool configuration");
                }
                if(m_config.budget.assemblyStreams == 0u)
                    m_config.budget.assemblyStreams = m_graph.streamCount;
                else if(m_config.budget.assemblyStreams < m_graph.streamCount)
                    throw std::invalid_argument(
                        "graph::startPool: the packet budget reserves fewer assembly streams than the graph has");
                // The build-time capture now measures the pool's own persistent
                // kernel (`cmake/GraphRegisterProbe.cu` instantiates
                // `detail::PoolKernel` with the default `NullBinder`), so a
                // heavier pool kernel lowers the wave automatically. The
                // capture cannot see a user binder, which is compiled into the
                // same kernel and raises the register count: on the A30 the
                // default binder compiles to 48 registers/thread (32 blocks/SM,
                // the 1792-worker wave) while the test toy binder compiles to 90
                // (20 blocks/SM, 1120 blocks). The 128-register floor is kept on
                // top of the capture so an unmeasured binder cannot turn the
                // advertised single wave into a two-wave launch. A caller that
                // has measured its binder passes `PoolConfig::registerHint`.
                int const registerHint = m_config.registerHint > 0
                                             ? m_config.registerHint
                                             : std::max(defaultKernelRegisters, fallbackKernelRegisters);
                auto const environment = queryLaunchEnvironment(m_device, registerHint);
                m_geometry = computeLaunchGeometry(environment.props, LaunchPolicy{});
                // Test-only worker cap. It only lowers the geometry, so the
                // single-wave check below still holds; `PoolConfig::testWorkerLimit`
                // is documented as a test knob, not a public API.
                if(m_config.testWorkerLimit != 0u)
                {
                    if(m_config.testWorkerLimit > m_geometry.workerCount)
                        throw std::invalid_argument("graph::startPool: testWorkerLimit exceeds the launch geometry");
                    m_geometry.blocks = m_config.testWorkerLimit * m_geometry.warpsPerBlock;
                    m_geometry.workerCount = m_config.testWorkerLimit * m_geometry.warpsPerBlock;
                }
                auto const validation = validateSingleWave(m_geometry, environment.props, environment.limits);
                if(!validation.singleWave)
                    throw std::runtime_error(
                        "graph::startPool: launch geometry is not a single wave: " + validation.reason);

                auto const configError = validatePoolConfig(m_config, m_geometry);
                if(!configError.empty())
                    throw std::invalid_argument(configError);

                m_config.budget.workers = m_geometry.workerCount;
                m_readySoftLimit = derivedReadySoftLimit(m_config, m_geometry);
                if(m_readySoftLimit == 0u)
                    throw std::invalid_argument("graph::startPool: derived ready soft limit is zero");

                m_ingress.emplace(IngressRing<JobDescriptor, poolIngressCapacity>::create(m_device));

                m_readyStorage.emplace(
                    alpaka::onHost::alloc<std::uint8_t>(m_device, static_cast<std::size_t>(hostReadyStorageBytes)));
                m_packetStorage.emplace(
                    alpaka::onHost::alloc<std::uint8_t>(m_device, static_cast<std::size_t>(hostPacketStorageBytes)));
                m_table.emplace(alpaka::onHost::alloc<OperationTableLayout>(m_device, 1u));
                m_hostTable.emplace(alpaka::onHost::allocHost<OperationTableLayout>(1u));
                m_graphBuffer.emplace(alpaka::onHost::alloc<CompiledGraph>(m_device, 1u));
                m_hostGraph.emplace(alpaka::onHost::allocHost<CompiledGraph>(1u));
                m_staging.emplace(
                    alpaka::onHost::alloc<JobDescriptor>(m_device, static_cast<std::uint32_t>(poolStagingCapacity)));
                std::size_t const assemblyBytes
                    = std::max<std::size_t>(1u, m_graph.streamCount) * sizeof(AssemblyState);
                m_assemblyStorage.emplace(alpaka::onHost::alloc<std::uint8_t>(m_device, assemblyBytes));
                m_deviceState.emplace(alpaka::onHost::alloc<PoolDeviceState>(m_device, 1u));
                m_hostState.emplace(alpaka::onHost::allocHost<PoolDeviceState>(1u));
                m_diagnosticsState.emplace(alpaka::onHost::alloc<BarrierDiagnosticsState>(m_device, 1u));
                m_hostDiagnostics.emplace(alpaka::onHost::allocHost<BarrierDiagnosticsState>(1u));
                m_hostControl.emplace(alpaka::onHost::allocMapped<PoolHostControl>(m_device, 1u));
                m_ringHeaderHost.emplace(
                    alpaka::onHost::allocHost<std::uint8_t>(static_cast<std::size_t>(sizeof(ReadyRingLayout))));

                if(!alpaka::onHost::isDataAccessible(m_device, *m_hostControl))
                    throw std::runtime_error(
                        "graph::startPool: missing capability 'mapped pinned host memory' - the device cannot "
                        "access the mapped ingress/control allocation, so no system-scope transport is available");

                initializeDeviceState();
                launch();
            }

            PoolEnqueueResult tryEnqueue(std::span<RequestSpec const> requests) override
            {
                PoolEnqueueResult result;
                result.requested = static_cast<std::uint32_t>(requests.size());
                if(!m_ingress)
                {
                    result.status = PoolEnqueueStatus::closed;
                    return result;
                }
                std::vector<JobDescriptor> descriptors;
                descriptors.reserve(requests.size());
                for(auto const& request : requests)
                    descriptors.push_back(makeRequestJob(request, m_config.sourceOperation, m_submitSequence++));
                auto const submitted = m_ingress->tryEnqueue(descriptors);
                result.accepted = submitted.accepted;
                switch(submitted.status)
                {
                case EnqueueStatus::accepted:
                    result.status = PoolEnqueueStatus::accepted;
                    break;
                case EnqueueStatus::ring_full:
                    result.status = PoolEnqueueStatus::full;
                    break;
                case EnqueueStatus::closed:
                    result.status = PoolEnqueueStatus::closed;
                    break;
                case EnqueueStatus::error:
                    result.status = PoolEnqueueStatus::error;
                    break;
                }
                m_accepted += submitted.accepted;
                return result;
            }

            PoolEnqueueResult enqueueBlocking(std::span<RequestSpec const> requests, std::chrono::nanoseconds timeout)
                override
            {
                auto const deadline = std::chrono::steady_clock::now() + timeout;
                PoolEnqueueResult total;
                total.requested = static_cast<std::uint32_t>(requests.size());
                std::size_t offset = 0u;
                while(offset < requests.size())
                {
                    auto const part = tryEnqueue(requests.subspan(offset));
                    if(part.status == PoolEnqueueStatus::closed || part.status == PoolEnqueueStatus::error)
                    {
                        total.status = part.status;
                        return total;
                    }
                    total.accepted += part.accepted;
                    offset += part.accepted;
                    if(offset >= requests.size())
                        break;
                    if(std::chrono::steady_clock::now() >= deadline)
                    {
                        total.status = PoolEnqueueStatus::full;
                        return total;
                    }
                    std::this_thread::yield();
                }
                total.status = PoolEnqueueStatus::accepted;
                return total;
            }

            PoolEnqueueStatus closeInput(std::chrono::nanoseconds timeout) override
            {
                if(!m_ingress)
                    return PoolEnqueueStatus::closed;
                auto const deadline = std::chrono::steady_clock::now() + timeout;
                while(true)
                {
                    auto const status = m_ingress->closeInput();
                    if(status.status == EnqueueStatus::accepted)
                        return PoolEnqueueStatus::accepted;
                    if(status.status == EnqueueStatus::closed)
                        return PoolEnqueueStatus::closed;
                    if(status.status == EnqueueStatus::error)
                        return PoolEnqueueStatus::error;
                    if(std::chrono::steady_clock::now() >= deadline)
                        return PoolEnqueueStatus::full;
                    std::this_thread::yield();
                }
            }

            void abort() override
            {
                if(!m_hostControl)
                    return;
                bool const systemScope = usesSystemScope();
                graph::detail::hostCounterStoreRelease(
                    systemScope,
                    &alpaka::onHost::data(*m_hostControl)->abortRequest,
                    std::uint64_t{1u});
            }

            PoolResult wait() override
            {
                if(!m_event)
                {
                    finalize();
                    return m_result;
                }
                alpaka::onHost::wait(*m_event);
                finalize();
                return m_result;
            }

            std::optional<PoolResult> waitFor(std::chrono::nanoseconds timeout) override
            {
                if(!m_finalized)
                {
                    auto const deadline = std::chrono::steady_clock::now() + timeout;
                    while(m_event && !m_event->isComplete())
                    {
                        if(std::chrono::steady_clock::now() >= deadline)
                            return std::nullopt;
                        std::this_thread::sleep_for(std::chrono::microseconds{200});
                    }
                    finalize();
                }
                return m_result;
            }

            LaunchGeometry geometry() const override
            {
                return m_geometry;
            }

            QuiescenceSnapshot quiescenceSnapshot() override
            {
                return m_result.counters;
            }

            void copyResultsToHost(void* pinnedDestination, std::size_t bytes) override
            {
                if(m_config.resultDevice == nullptr || m_config.resultBytes == 0u)
                    throw std::logic_error("graph::Pool::copyResultsToHost: no result device buffer configured");
                if(bytes == 0u || bytes > m_config.resultBytes)
                    throw std::invalid_argument("graph::Pool::copyResultsToHost: byte count is out of range");
                if(pinnedDestination == nullptr)
                    throw std::invalid_argument("graph::Pool::copyResultsToHost: destination is null");
                if(bytes > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
                    throw std::invalid_argument("graph::Pool::copyResultsToHost: byte count exceeds the extent type");

                auto const extents = alpaka::Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(bytes)};
                auto const source
                    = alpaka::makeView(m_device, static_cast<std::uint8_t*>(m_config.resultDevice), extents);
                auto destination = alpaka::makeView(m_device, static_cast<std::uint8_t*>(pinnedDestination), extents);
                m_copyTokens.push_back(m_copyQueue.asyncCopyToHost(destination, source));
            }

            void waitCopies() override
            {
                for(auto const& token : m_copyTokens)
                    token.wait();
                m_copyTokens.clear();
            }

            std::uint64_t completedJobs() const override
            {
                if(!m_hostControl)
                    return 0u;
                bool const systemScope = usesSystemScope();
                auto const* control = alpaka::onHost::data(*m_hostControl);
                std::uint64_t total = 0u;
                for(std::uint32_t worker = 0u; worker < m_geometry.workerCount; ++worker)
                    total += graph::detail::hostCounterLoadAcquire(systemScope, &control->workerRetired[worker]);
                return total;
            }

            PoolConfig const& config() const override
            {
                return m_config;
            }

        private:
            bool usesSystemScope() const
            {
                using Api = ALPAKA_TYPEOF(m_device.getApi());
                return std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Cuda>;
            }

            void initializeDeviceState()
            {
                std::memset(alpaka::onHost::data(*m_hostState), 0, sizeof(PoolDeviceState));
                alpaka::onHost::memcpy(m_queue, *m_deviceState, *m_hostState);

                // The divergence diagnostics state starts with `divergenceCount`
                // zero and the named fields at their sentinel values, so the
                // host can tell "never diverged" from a recorded first site.
                auto* hostDiag = alpaka::onHost::data(*m_hostDiagnostics);
                std::memset(hostDiag, 0, sizeof(BarrierDiagnosticsState));
                hostDiag->firstBarrierId = ~std::uint32_t{0u};
                hostDiag->firstWorker = ~std::uint32_t{0u};
                hostDiag->firstStage = ~std::uint32_t{0u};
                alpaka::onHost::memcpy(m_queue, *m_diagnosticsState, *m_hostDiagnostics);

                auto* control = alpaka::onHost::data(*m_hostControl);
                std::memset(control, 0, sizeof(PoolHostControl));
                control->initDone = 0u;
                control->abortRequest = 0u;
                control->imported = 0u;
                control->stopReason = 0u;
                control->reserved = 0u;

                auto* table = alpaka::onHost::data(*m_hostTable);
                std::memset(table, 0, sizeof(OperationTableLayout));
                table->count = m_graph.nodeCount;
                for(std::uint32_t index = 0u; index < m_graph.nodeCount; ++index)
                {
                    table->entries[index].id = m_graph.nodes[index].operation;
                    table->entries[index].inputSchema = m_config.packetSchema;
                    table->entries[index].outputSchema = m_config.packetSchema;
                    table->entries[index].function = nullptr;
                }
                alpaka::onHost::memcpy(m_queue, *m_table, *m_hostTable);

                auto* hostGraph = alpaka::onHost::data(*m_hostGraph);
                *hostGraph = m_graph;
                alpaka::onHost::memcpy(m_queue, *m_graphBuffer, *m_hostGraph);

                std::size_t const assemblyBytes
                    = std::max<std::size_t>(1u, m_graph.streamCount) * sizeof(AssemblyState);
                auto hostAssembly = alpaka::onHost::allocHost<std::uint8_t>(assemblyBytes);
                std::memset(alpaka::onHost::data(hostAssembly), 0, assemblyBytes);
                // Zeroed storage has StreamId value 0, which is a legal stream
                // id. Mark every slot explicitly as uninitialized so the
                // assembler's `initialize()` accepts it exactly once.
                for(std::uint32_t index = 0u; index < m_graph.streamCount; ++index)
                {
                    auto* state = reinterpret_cast<AssemblyState*>(
                        alpaka::onHost::data(hostAssembly) + static_cast<std::size_t>(index) * sizeof(AssemblyState));
                    state->id = StreamId::invalid();
                }
                alpaka::onHost::memcpy(m_queue, *m_assemblyStorage, hostAssembly);
                alpaka::onHost::wait(m_queue);
            }

            void launch()
            {
                detail::PoolKernelArgs args{};
                args.state = alpaka::onHost::data(*m_deviceState);
                args.readyStorage = alpaka::onHost::data(*m_readyStorage);
                args.packetStorage = alpaka::onHost::data(*m_packetStorage);
                args.operationTable = alpaka::onHost::data(*m_table);
                args.graph = alpaka::onHost::data(*m_graphBuffer);
                args.hostControl = alpaka::onHost::data(*m_hostControl);
                auto const ingressView = m_ingress->view();
                args.ingressControl = ingressView.control;
                args.ingressSlots = ingressView.slots;
                args.ingressSlotCount = ingressView.slotCount;
                args.ingressSystemScope = ingressView.systemScopeHostAtomics;
                args.staging = alpaka::onHost::data(*m_staging);
                args.stagingCapacity = poolStagingCapacity;
                args.maxImportBatch = m_config.maxImportBatch;
                args.importPerIteration = m_config.importPerIteration;
                args.idleNanosleep = m_config.idleNanosleep;
                args.chunkCapacity = m_config.chunkCapacity;
                args.readySoftLimit = m_readySoftLimit;
                args.workers = m_geometry.workerCount;
                args.idLimit = m_config.idLimit;
                args.initDeadlineCycles = m_config.initDeadlineCycles;
                args.initDeadlineIterations = m_config.initDeadlineIterations;
                args.packetSchema = m_config.packetSchema;
                args.budget = m_config.budget;
                args.controlBlock = m_config.controlBlock;
                args.assemblyStorage = alpaka::onHost::data(*m_assemblyStorage);
                args.assemblyStreamCount = m_graph.streamCount;
                args.assemblySlotCount = static_cast<std::uint32_t>(poolPacketSlotCapacity);
                args.barrierDiagnostics = alpaka::onHost::data(*m_diagnosticsState);
                args.barrierDiagnosticsEnabled = m_config.testEnableBarrierDiagnostics;
                args.barrierSkipOnDivergence = m_config.testBarrierSkipOnDivergence;

                auto const threadSpec
                    = alpaka::onHost::ThreadSpec{m_geometry.blocks, m_geometry.threadsPerBlock, m_executor};
                m_queue.enqueue(threadSpec, detail::PoolKernel<T_Binder>{}, args);
                m_event.emplace(m_queue.makeEvent());
                m_queue.enqueue(*m_event);
                m_launched = true;
            }

            void finalize()
            {
                if(m_finalized)
                    return;
                if(m_event && !m_event->isComplete())
                    alpaka::onHost::wait(*m_event);

                alpaka::onHost::memcpy(m_queue, *m_hostState, *m_deviceState);
                alpaka::onHost::memcpy(m_queue, *m_hostDiagnostics, *m_diagnosticsState);
                auto const headerBytes
                    = alpaka::Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(sizeof(ReadyRingLayout))};
                auto const headerView = alpaka::makeView(
                    m_device,
                    static_cast<std::uint8_t*>(alpaka::onHost::data(*m_readyStorage)),
                    headerBytes);
                alpaka::onHost::memcpy(m_queue, *m_ringHeaderHost, headerView);
                alpaka::onHost::wait(m_queue);

                PoolDeviceState const* state = alpaka::onHost::data(*m_hostState);
                ReadyRingLayout header{};
                std::memcpy(&header, alpaka::onHost::data(*m_ringHeaderHost), sizeof(ReadyRingLayout));
                BarrierDiagnosticsState const* diag = alpaka::onHost::data(*m_hostDiagnostics);

                // Barrier divergence diagnostics: report whether the check ran
                // and, when it saw a divergence, the first call site that
                // reached the block barrier with a non-full active mask. A
                // divergent arrival at a block barrier is a control-flow defect
                // (deadlock), so naming the site is the whole point of the
                // diagnostic; the per-worker histogram localizes which warps.
                PoolResult::BarrierDiagnosticsSnapshot diagSnapshot;
                diagSnapshot.enabled = enableBarrierDivergenceDiagnostics && m_config.testEnableBarrierDiagnostics;
                diagSnapshot.divergenceCount = diag->divergenceCount;
                diagSnapshot.firstBarrierId = diag->firstBarrierId;
                diagSnapshot.firstWorker = diag->firstWorker;
                diagSnapshot.firstStage = diag->firstStage;
                diagSnapshot.firstIteration = diag->firstIteration;
                diagSnapshot.firstMask = diag->firstMask;
                diagSnapshot.firstWidth = diag->firstWidth;
                if(diagSnapshot.enabled && diag->divergenceCount != 0u)
                {
                    std::printf(
                        "[pool-diag] BARRIER DIVERGENCE count=%llu first: barrierId=%u worker=%u stage=%u "
                        "iter=%llu mask=0x%x width=%u\n",
                        static_cast<unsigned long long>(diag->divergenceCount),
                        static_cast<unsigned>(diag->firstBarrierId),
                        static_cast<unsigned>(diag->firstWorker),
                        static_cast<unsigned>(diag->firstStage),
                        static_cast<unsigned long long>(diag->firstIteration),
                        static_cast<unsigned>(diag->firstMask),
                        static_cast<unsigned>(diag->firstWidth));
                    for(std::uint32_t worker = 0u; worker < m_geometry.workerCount; ++worker)
                    {
                        if(diag->perWorkerDivergences[worker] != 0u)
                        {
                            std::printf(
                                "[pool-diag]   worker=%u divergences=%u\n",
                                static_cast<unsigned>(worker),
                                static_cast<unsigned>(diag->perWorkerDivergences[worker]));
                        }
                    }
                    std::fflush(stdout);
                }

                std::uint64_t distinctWorkers = 0u;
                std::uint32_t const words = (m_geometry.workerCount + 63u) / 64u;
                for(std::uint32_t word = 0u; word < words; ++word)
                    distinctWorkers += static_cast<std::uint64_t>(alpaka::popcount(state->workerBits[word]));

                QuiescenceSnapshot snapshot;
                snapshot.accepted = m_accepted;
                snapshot.imported = state->importedJobs;
                snapshot.queued = header.pushCounter - header.claimCounter;
                snapshot.active = state->active;
                snapshot.inFlightReservations = state->inFlightReservations;
                snapshot.pendingSources = state->pendingSources;
                snapshot.partialBuffers = state->partialBuffers;
                snapshot.publications = state->publications;
                snapshot.outstanding = state->outstanding;
                snapshot.completedJobs = state->completedJobs;
                snapshot.claimAttempts = state->claimAttempts;
                snapshot.distinctWorkers = distinctWorkers;
                snapshot.launchCount = state->launchCount;
                snapshot.retiredPackets = state->retiredPackets;
                snapshot.slotsReturned = state->slotsReturned;
                snapshot.drainPublications = state->drainPublications;
                snapshot.drainRounds = state->drainRounds;
                snapshot.peakHeldSlots = state->peakHeldSlots;
                snapshot.importPolls = state->debugPolls;
                snapshot.importBlockedPolls = state->debugImportBlocked;
                snapshot.importEmptyPolls = state->debugImportEmpty;
                snapshot.readyViolations = state->readyViolations + header.violations;
                snapshot.errorWord = state->errorWord;
                snapshot.stopReason = static_cast<std::uint32_t>(state->stopReason);
                snapshot.initFailures = state->initFailures;
                snapshot.initWinner = state->initWinner;
                snapshot.initMaxSpin = state->initMaxSpin;
                for(std::uint32_t index = 0u; index < 8u; ++index)
                    snapshot.execCounts[index] = state->execCounts[index];
                snapshot.debugBlockThreads = state->debugBlockThreads;
                snapshot.debugWarpWidth = state->debugWarpWidth;
                snapshot.debugRequiredSlots = state->debugRequiredSlots;
                snapshot.debugBudgetWorkers = state->debugBudgetWorkers;
                snapshot.debugConfigured = state->debugConfigured;
                snapshot.debugStopStage = state->debugStopStage;
                snapshot.initLockValue = state->initLock;
                snapshot.readyCapacity = static_cast<std::uint32_t>(poolReadyCapacity);
                snapshot.packetCapacity = static_cast<std::uint32_t>(poolPacketSlotCapacity);
                snapshot.workerCount = m_geometry.workerCount;
                snapshot.initializerClaims = state->initializerClaims;

                // Exit-path diagnostics: dump every worker that is not `exiting`
                // after a stop was requested. This is the snapshot that names the
                // loop stage of a worker that never left the scheduling loop. The
                // words are copied from device state; the host ring-header copy
                // (above) gives the ready-ring lock owner state.
                if(state->stopReason != 0u)
                {
                    std::printf(
                        "[pool-diag] stopReason=%llu errorWord=0x%x; workers still not exiting:\n",
                        static_cast<unsigned long long>(state->stopReason),
                        state->errorWord);
                    for(std::uint32_t worker = 0u; worker < m_geometry.workerCount; ++worker)
                    {
                        std::uint32_t const stage = state->workerStage[worker];
                        if(stage != static_cast<std::uint32_t>(detail::WorkerStage::exiting))
                        {
                            std::uint32_t ringLock = 0u;
                            if(m_ringHeaderHost && alpaka::onHost::data(*m_ringHeaderHost) != nullptr)
                            {
                                ReadyRingLayout const* header
                                    = reinterpret_cast<ReadyRingLayout const*>(alpaka::onHost::data(*m_ringHeaderHost));
                                ringLock = header->lock;
                            }
                            std::printf(
                                "[pool-diag]   worker=%u stage=%s iters=%llu last=%s ringLock=%u\n",
                                static_cast<unsigned>(worker),
                                detail::toString(static_cast<detail::WorkerStage>(stage)),
                                static_cast<unsigned long long>(state->workerIterations[worker]),
                                detail::toString(static_cast<detail::WorkerLastAction>(state->workerLastAction[worker])),
                                static_cast<unsigned>(ringLock));
                        }
                    }
                    std::fflush(stdout);
                }

                PoolResult result;
                result.counters = snapshot;
                result.errorWord = state->errorWord;
                result.barrierDiagnostics = diagSnapshot;
                if(state->stopReason == poolStoppingDrained && snapshot.outstanding == 0u
                   && snapshot.pendingSources == 0u && snapshot.partialBuffers == 0u)
                {
                    result.reason = PoolStopReason::stoppedByDrain;
                }
                else if(state->stopReason == poolStoppingAborted)
                {
                    result.reason = PoolStopReason::stoppedByAbort;
                }
                else
                {
                    result.reason = PoolStopReason::stoppedByAbort;
                    result.errorWord |= poolErrorBits(PoolError::internalState);
                    result.counters.errorWord = result.errorWord;
                }
                m_result = result;
                m_finalized = true;
            }

            Device m_device;
            T_Executor m_executor;
            PoolConfig m_config;
            CompiledGraph m_graph;
            T_Binder m_binder;
            CopyQueue<Device> m_copyQueue;
            QueueType m_queue;
            LaunchGeometry m_geometry{};
            std::uint32_t m_readySoftLimit{0u};
            std::uint64_t m_submitSequence{0u};
            std::uint64_t m_accepted{0u};
            bool m_launched{false};

            std::optional<IngressRing<JobDescriptor, poolIngressCapacity>> m_ingress;
            std::optional<ByteBuffer> m_readyStorage;
            std::optional<ByteBuffer> m_packetStorage;
            std::optional<ByteBuffer> m_assemblyStorage;
            std::optional<TableBuffer> m_table;
            std::optional<HostTableBuffer> m_hostTable;
            std::optional<GraphBuffer> m_graphBuffer;
            std::optional<HostGraphBuffer> m_hostGraph;
            std::optional<StagingBuffer> m_staging;
            std::optional<StateBuffer> m_deviceState;
            std::optional<HostStateBuffer> m_hostState;
            std::optional<DiagBuffer> m_diagnosticsState;
            std::optional<HostDiagBuffer> m_hostDiagnostics;
            std::optional<HostControlBuffer> m_hostControl;
            std::optional<HostByteBuffer> m_ringHeaderHost;
            std::optional<EventType> m_event;
            std::vector<CopyToken> m_copyTokens;
            bool m_finalized{false};
            PoolResult m_result{};
        };

    } // namespace detail

    /** Persistent warp pool handle.
     *
     * The handle is cheap to copy; every copy shares one backend and therefore
     * one kernel lifetime.
     */
    class Pool
    {
    public:
        Pool() = default;

        /** Internal: wrap a backend. Not part of the user-facing constructor set. */
        explicit Pool(std::shared_ptr<detail::PoolBackend> backend) : m_backend(std::move(backend))
        {
        }

        bool started() const noexcept
        {
            return static_cast<bool>(m_backend);
        }

        PoolEnqueueResult tryEnqueue(std::span<RequestSpec const> requests)
        {
            if(!m_backend)
            {
                PoolEnqueueResult result;
                result.status = PoolEnqueueStatus::closed;
                result.requested = static_cast<std::uint32_t>(requests.size());
                return result;
            }
            return m_backend->tryEnqueue(requests);
        }

        PoolEnqueueResult enqueue(
            std::span<RequestSpec const> requests,
            std::chrono::nanoseconds timeout = std::chrono::seconds{1})
        {
            if(!m_backend)
            {
                PoolEnqueueResult result;
                result.status = PoolEnqueueStatus::closed;
                result.requested = static_cast<std::uint32_t>(requests.size());
                return result;
            }
            return m_backend->enqueueBlocking(requests, timeout);
        }

        PoolEnqueueStatus closeInput(std::chrono::nanoseconds timeout = std::chrono::seconds{1})
        {
            if(!m_backend)
                return PoolEnqueueStatus::closed;
            return m_backend->closeInput(timeout);
        }

        void abort()
        {
            if(m_backend)
                m_backend->abort();
        }

        PoolResult wait()
        {
            if(!m_backend)
                return PoolResult{};
            return m_backend->wait();
        }

        std::optional<PoolResult> waitFor(std::chrono::nanoseconds timeout)
        {
            if(!m_backend)
                return PoolResult{};
            return m_backend->waitFor(timeout);
        }

        LaunchGeometry geometry() const
        {
            return m_backend ? m_backend->geometry() : LaunchGeometry{};
        }

        QuiescenceSnapshot quiescenceSnapshot()
        {
            return m_backend ? m_backend->quiescenceSnapshot() : QuiescenceSnapshot{};
        }

        void copyResultsToHost(void* pinnedDestination, std::size_t bytes)
        {
            if(!m_backend)
                throw std::logic_error("graph::Pool::copyResultsToHost: pool was never started");
            m_backend->copyResultsToHost(pinnedDestination, bytes);
        }

        void waitCopies()
        {
            if(m_backend)
                m_backend->waitCopies();
        }

        std::uint64_t completedJobs() const
        {
            return m_backend ? m_backend->completedJobs() : 0u;
        }

        PoolConfig const& config() const
        {
            static PoolConfig const empty{};
            return m_backend ? m_backend->config() : empty;
        }

    private:
        std::shared_ptr<detail::PoolBackend> m_backend;
    };

    namespace detail
    {
        inline Pool makeDetachedPool(PoolConfig config)
        {
            return Pool{std::make_shared<DetachedPoolBackend>(std::move(config))};
        }
    } // namespace detail

    /** Start one persistent worker kernel and return the pool handle.
     *
     * @tparam T_Binder device operation binder, deduced from @p binder. See
     *         `graph/detail/scheduler.hpp`; the binder must return real device
     *         function addresses.
     */
    template<typename T_Device, typename T_Executor, typename T_Binder = detail::NullBinder>
    Pool startPool(
        T_Device device,
        T_Executor executor,
        CompiledGraph const& graph,
        PoolConfig const& config,
        T_Binder binder = {})
    {
        auto backend = std::make_shared<detail::PersistentPoolBackend<T_Device, T_Executor, T_Binder>>(
            std::move(device),
            executor,
            graph,
            config,
            binder);
        return Pool{std::move(backend)};
    }
} // namespace graph
