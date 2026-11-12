// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// End-to-end particle-feedback example: one persistent pool launch, one source
// request of N particles, ordered close, complete drain, host-side histogram.
//
// # Execution graph used here
//
// The design's five-node graph (`GenerateParticles -> FilterByProperty ->
// {ProcessA, ProcessB} -> {feedback | DestroyParticles}`) is a branching,
// cyclic dataflow. The pool at this branch ("pool-core") forwards one shared
// packet in place along every outgoing edge and has no chunk-assembly
// publication yet, so a fan-out would deliver the same packet to both bins and
// a declared feedback edge would never quiesce. Building that topology here
// would therefore either hang or silently compute the wrong result.
//
// This executable keeps the same registered operation set, the same element
// layout, and the same per-particle semantics, but linearizes the dataflow:
//
//     GenerateParticles -> FilterByProperty -> ProcessA -> ProcessB -> DestroyParticles
//
// `FilterByProperty` recomputes the visit-0 bin decision (it is a pure function
// of `(seed, id, visit)`), `ProcessA` runs the exact feedback loop for the
// particles whose visit-0 bin is Bin 0, `ProcessB` does the same for Bin 1, and
// the sink writes the unique per-id result slot. Each particle is still driven
// through every filter/A/B round of the design; only the packet routing is
// replaced by an in-packet loop. The output is the design's exact histogram and
// is independent of chunk size and backend. The real branching pipeline remains
// blocked on the pool's assembly/publication milestone.

#include <alpaka/alpaka.hpp>

// Base defect workaround: `graph/chunk_assembly.hpp` (pulled in by the functor
// header) and `graph/graph_model.hpp` both define `graph::StreamId` with the
// same layout. No translation unit may include both headers unchanged, and the
// example needs the functors and the host graph model. Only the chunk-assembly
// spelling is renamed, and only for the duration of that one include; this
// executable never uses a chunk-assembly stream id.
#define StreamId GraphChunkAssemblyStreamId
#include <graph/particle_functors.hpp>
#undef StreamId

#include "cli.hpp"

#include <graph/detail/capabilities.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/histogram.hpp>
#include <graph/particle_feedback_graph.hpp>
#include <graph/pool.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    using graph::JobDescriptor;
    using graph::PacketAccessor;
    using graph::Particle;
    using graph::particle_feedback::Backend;
    using graph::particle_feedback::Options;
    using graph::WarpGroup;

    /** Per-id final-result slot written by `DestroyParticles`.
     *
     * `visits` is the sink visit marker: the host requires exactly one sink
     * write per source id, so a lost or duplicated job cannot hide behind the
     * histogram. `weight` is only published by the first writer.
     */
    struct ResultSlot
    {
        std::int64_t weight{graph::unwrittenWeight};
        std::uint32_t visits{0u};
        std::uint32_t reserved{0u};
    };

    static_assert(std::is_trivially_copyable_v<ResultSlot>);
    static_assert(std::is_standard_layout_v<ResultSlot>);
    static_assert(alignof(ResultSlot) >= alignof(std::uint32_t));

    /** Device-visible run state shared by all registered operations.
     *
     * The functor ABI has no state parameter, so the example reserves the
     * documented control port (`maxPacketPorts - 1`) of the packet accessor for
     * this block. The pool never dereferences it.
     */
    struct RunControl
    {
        ResultSlot* results{nullptr};
        std::uint64_t particleCount{0u};
        std::uint64_t seed{0u};
        std::uint64_t maxVisits{1'000'000u};
        std::uint64_t generateElements{0u};
        std::uint64_t processElements{0u};
        std::uint64_t destroyElements{0u};
        std::uint64_t destroyMarked{0u};
        std::uint32_t errorWord{0u};
        std::uint32_t reserved{0u};
    };

    static_assert(std::is_trivially_copyable_v<RunControl>);
    static_assert(std::is_standard_layout_v<RunControl>);

    /** Sentinel visit value that marks a particle as destroyed in the packet. */
    inline constexpr std::uint32_t destroyedVisit = std::numeric_limits<std::uint32_t>::max();

    /** Bits of `RunControl::errorWord`. */
    enum ExampleError : std::uint32_t
    {
        noError = 0u,
        errorIdOutOfRange = 1u << 0u,
        errorWeightOverflow = 1u << 1u,
        errorVisitGuard = 1u << 2u,
        errorResultSlot = 1u << 3u,
        errorMissingControl = 1u << 4u
    };

    template<typename T_Acc>
    ALPAKA_FN_ACC void recordError(T_Acc const& acc, RunControl* control, std::uint32_t bit) noexcept
    {
        if(control != nullptr)
            (void) alpaka::onAcc::atomicOr(acc, &control->errorWord, bit, alpaka::onAcc::scope::device);
    }

    template<typename T_Acc>
    ALPAKA_FN_ACC RunControl* fetchControl(T_Acc const& acc, PacketAccessor const& accessor) noexcept
    {
        auto* control = static_cast<RunControl*>(accessor.inputData[graph::maxPacketPorts - 1u]);
        if(control == nullptr)
            recordError(acc, nullptr, errorMissingControl);
        return control;
    }

    ALPAKA_FN_HOST_ACC constexpr std::uint32_t elementCapacity(std::uint32_t elementSize) noexcept
    {
        return elementSize == 0u ? 0u : static_cast<std::uint32_t>(graph::poolPacketElementBytes / elementSize);
    }

    // ---------------------------------------------------------------- nodes ---

    /** Source node: assign the unique id and the property draw, weight zero. */
    template<typename T_Acc>
    struct GenerateOperation
    {
        ALPAKA_FN_ACC static void operation(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor) noexcept
        {
            auto* control = fetchControl(acc, accessor);
            auto const output = accessor.outputSpan<Particle>(job.routing.outputPort);
            auto const count = output.size();
            WarpGroup<T_Acc> const warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();

            std::uint64_t written = 0u;
            for(std::size_t i = lane; i < count; i += width)
            {
                auto const id = job.request.firstId + static_cast<std::uint64_t>(i);
                if(control != nullptr && id >= control->particleCount)
                {
                    recordError(acc, control, errorIdOutOfRange);
                    continue;
                }
                output.write(i, graph::GenerateParticles::apply(job.request.seed, id));
                ++written;
            }
            if(control != nullptr)
                (void)
                    alpaka::onAcc::atomicAdd(acc, &control->generateElements, written, alpaka::onAcc::scope::device);
        }
    };

    /** Filter stage.
     *
     * The bin decision is a pure function of `(seed, id, visit)`, so the two
     * processors recompute it from the same record; the pool-core packet has no
     * spare side channel to carry a separate bin tag.
     */
    template<typename T_Acc>
    struct FilterOperation
    {
        ALPAKA_FN_ACC static void operation(T_Acc const&, JobDescriptor const&, PacketAccessor const&) noexcept
        {
        }
    };

    /** Run the complete feedback loop for one particle.
     *
     * @return false when the loop reported an error (the particle is not marked
     *         as destroyed and the sink therefore cannot publish a slot).
     */
    template<typename T_Acc>
    ALPAKA_FN_ACC bool runFeedbackLoop(
        T_Acc const& acc,
        RunControl* control,
        graph::RngSeed seed,
        Particle& particle) noexcept
    {
        using graph::particle_feedback::Destination;

        while(true)
        {
            auto const filter = graph::FilterByProperty::apply(seed, particle);
            bool const bin1 = filter.destination == Destination::Bin1;
            auto const processed
                = bin1 ? graph::ProcessB::apply(seed, particle) : graph::ProcessA::apply(seed, particle);
            if(processed.error)
            {
                recordError(acc, control, errorWeightOverflow);
                return false;
            }
            if(processed.destination == Destination::FilterInput)
            {
                particle
                    = bin1 ? graph::ProcessB::feedbackParticle(particle) : graph::ProcessA::feedbackParticle(particle);
                if(control != nullptr && static_cast<std::uint64_t>(particle.visit) > control->maxVisits)
                {
                    recordError(acc, control, errorVisitGuard);
                    return false;
                }
                continue;
            }

            (void) graph::DestroyParticles::apply(particle);
            particle.visit = destroyedVisit;
            return true;
        }
    }

    /** Bin processor: drives the particles that start in one bin to the sink. */
    template<typename T_Acc, bool T_startInBin1>
    struct ProcessOperation
    {
        ALPAKA_FN_ACC static void operation(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor) noexcept
        {
            using graph::particle_feedback::Destination;

            auto* control = fetchControl(acc, accessor);
            if(control == nullptr)
                return;
            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            auto const output = accessor.outputSpan<Particle>(job.routing.outputPort);
            auto const count = input.size();
            auto const seed = static_cast<graph::RngSeed>(control->seed);
            WarpGroup<T_Acc> const warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();

            std::uint64_t seen = 0u;
            for(std::size_t i = lane; i < count; i += width)
            {
                auto particle = input.read(i);
                // ProcessA runs first in the chain and marks every particle it
                // finished; ProcessB must leave those records untouched.
                if(particle.visit == destroyedVisit)
                    continue;
                auto const initial = graph::FilterByProperty::apply(seed, particle);
                bool const startsInBin1 = initial.destination == Destination::Bin1;
                if(startsInBin1 != T_startInBin1)
                    continue;
                if(runFeedbackLoop(acc, control, seed, particle))
                {
                    output.write(i, particle);
                    ++seen;
                }
            }
            (void) alpaka::onAcc::atomicAdd(acc, &control->processElements, seen, alpaka::onAcc::scope::device);
        }
    };

    /** Sink: publish the unique per-id final weight and count the sink visit. */
    template<typename T_Acc>
    struct DestroyOperation
    {
        ALPAKA_FN_ACC static void operation(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor) noexcept
        {
            auto* control = fetchControl(acc, accessor);
            if(control == nullptr)
                return;
            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            auto const count = input.size();
            WarpGroup<T_Acc> const warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();

            std::uint64_t elements = 0u;
            std::uint64_t marked = 0u;
            for(std::size_t i = lane; i < count; i += width)
            {
                auto const particle = input.read(i);
                ++elements;
                if(particle.visit != destroyedVisit)
                    continue;
                ++marked;
                if(particle.id >= control->particleCount)
                {
                    recordError(acc, control, errorResultSlot);
                    continue;
                }
                auto* slot = &control->results[particle.id];
                auto const previous = alpaka::onAcc::atomicAdd(acc, &slot->visits, 1u, alpaka::onAcc::scope::device);
                if(previous == 0u)
                    slot->weight = static_cast<std::int64_t>(particle.weight);
            }
            (void) alpaka::onAcc::atomicAdd(acc, &control->destroyElements, elements, alpaka::onAcc::scope::device);
            (void) alpaka::onAcc::atomicAdd(acc, &control->destroyMarked, marked, alpaka::onAcc::scope::device);
        }
    };

    /** Bind the example operations to device function pointers. */
    struct ExampleBinder
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC static bool bind(T_Acc const&, graph::OperationTable<T_Acc>& table) noexcept
        {
            graph::particle_feedback::OperationIds const ids{};
            bool ok = true;
            for(std::uint32_t index = 0u; index < table.count; ++index)
            {
                graph::OperationId const id = table.entries[index].id;
                if(id == ids.generateParticles)
                    ok = table.bindOperation(id, &GenerateOperation<T_Acc>::operation) && ok;
                else if(id == ids.filterByProperty)
                    ok = table.bindOperation(id, &FilterOperation<T_Acc>::operation) && ok;
                else if(id == ids.processA)
                    ok = table.bindOperation(id, &ProcessOperation<T_Acc, false>::operation) && ok;
                else if(id == ids.processB)
                    ok = table.bindOperation(id, &ProcessOperation<T_Acc, true>::operation) && ok;
                else if(id == ids.destroyParticles)
                    ok = table.bindOperation(id, &DestroyOperation<T_Acc>::operation) && ok;
                else
                    ok = false;
            }
            return ok;
        }
    };

    /** Linear five-node execution graph with the design's schemas and ids. */
    inline std::optional<graph::CompiledGraph> buildExecutionGraph(std::uint32_t chunkSize, std::string& error)
    {
        using namespace graph;

        GraphModel model;
        auto const schema = makeElementSchema<Particle>(alpaka::Vec<std::size_t, 1u>{chunkSize});
        auto const streamPort = std::uint32_t{0u};
        StreamDescriptor const filterInput{particle_feedback::defaultStreamIds.filterInput, streamPort, chunkSize, 1u};
        StreamDescriptor const bin0{particle_feedback::defaultStreamIds.bin0, streamPort, chunkSize, 1u};
        StreamDescriptor const bin1{particle_feedback::defaultStreamIds.bin1, streamPort, chunkSize, 1u};
        StreamDescriptor const sink{particle_feedback::defaultStreamIds.sink, streamPort, chunkSize, 1u};
        model.addStream(filterInput);
        model.addStream(bin0);
        model.addStream(bin1);
        model.addStream(sink);

        auto const ids = particle_feedback::defaultOperationIds;

        NodeDescriptor generate;
        generate.name = "GenerateParticles";
        generate.operation = ids.generateParticles;
        generate.outputs.push_back(OutputPortDescriptor{schema, filterInput.id, streamPort});

        NodeDescriptor filter;
        filter.name = "FilterByProperty";
        filter.operation = ids.filterByProperty;
        filter.inputs.push_back(InputPortDescriptor{schema});
        filter.outputs.push_back(OutputPortDescriptor{schema, bin0.id, streamPort});

        NodeDescriptor processA;
        processA.name = "ProcessA";
        processA.operation = ids.processA;
        processA.inputs.push_back(InputPortDescriptor{schema});
        processA.outputs.push_back(OutputPortDescriptor{schema, bin1.id, streamPort});

        NodeDescriptor processB;
        processB.name = "ProcessB";
        processB.operation = ids.processB;
        processB.inputs.push_back(InputPortDescriptor{schema});
        processB.outputs.push_back(OutputPortDescriptor{schema, sink.id, streamPort});

        NodeDescriptor destroy;
        destroy.name = "DestroyParticles";
        destroy.operation = ids.destroyParticles;
        destroy.inputs.push_back(InputPortDescriptor{schema});

        auto const generateNode = model.addNode(std::move(generate));
        auto const filterNode = model.addNode(std::move(filter));
        auto const processANode = model.addNode(std::move(processA));
        auto const processBNode = model.addNode(std::move(processB));
        auto const destroyNode = model.addNode(std::move(destroy));

        bool const edgesAdded
            = model.addEdge(EdgeDescriptor{generateNode, 0u, filterNode, 0u, filterInput.id, streamPort, false})
              && model.addEdge(EdgeDescriptor{filterNode, 0u, processANode, 0u, bin0.id, streamPort, false})
              && model.addEdge(EdgeDescriptor{processANode, 0u, processBNode, 0u, bin1.id, streamPort, false})
              && model.addEdge(EdgeDescriptor{processBNode, 0u, destroyNode, 0u, sink.id, streamPort, false});
        if(!edgesAdded)
        {
            error = "graph: failed to add an execution edge";
            return std::nullopt;
        }

        auto outcome = model.buildGraph();
        if(!succeeded(outcome))
        {
            error = "graph: execution graph validation failed";
            for(auto const& graphError : std::get<std::vector<GraphError>>(outcome))
                error += "\n  " + graphError.message;
            return std::nullopt;
        }
        return std::get<CompiledGraph>(outcome);
    }

    // --------------------------------------------------------------- runner ---

    struct Outcomes
    {
        std::vector<std::int64_t> weights{};
        std::vector<std::uint32_t> visits{};
        std::uint64_t rounds{0u};
    };

    template<typename T_Device, typename T_Executor>
    int runExample(T_Device device, T_Executor executor, Options const& options, bool verbose)
    {
        std::uint64_t const particleCount = options.particles;
        std::uint32_t const maxChunk = elementCapacity(sizeof(Particle));
        std::uint32_t const chunkSize = std::min(options.chunkSize, maxChunk);
        if(chunkSize == 0u)
        {
            std::cerr << "graph_particle_feedback: the pool packet slot cannot hold one particle\n";
            return 1;
        }
        if(verbose && chunkSize != options.chunkSize)
        {
            std::cerr << "[graph_particle_feedback] requested chunk size " << options.chunkSize << " exceeds the "
                      << graph::poolPacketElementBytes << "-byte pool packet slot (" << sizeof(Particle)
                      << " bytes/particle); using " << chunkSize << "\n";
        }

        std::string graphError;
        auto const compiled = buildExecutionGraph(chunkSize, graphError);
        if(!compiled.has_value())
        {
            std::cerr << "graph_particle_feedback: " << graphError << '\n';
            return 1;
        }

        auto queue = device.makeQueue();
        std::uint32_t const slotCount = static_cast<std::uint32_t>(std::max<std::uint64_t>(particleCount, 1u));
        auto resultDevice = alpaka::onHost::alloc<ResultSlot>(device, slotCount);
        auto resultHostInit = alpaka::onHost::allocHost<ResultSlot>(slotCount);
        for(std::uint32_t index = 0u; index < slotCount; ++index)
            alpaka::onHost::data(resultHostInit)[index] = ResultSlot{};
        alpaka::onHost::memcpy(queue, resultDevice, resultHostInit);

        auto controlBuffer = alpaka::onHost::allocMapped<RunControl>(device, 1u);
        RunControl control{};
        control.results = alpaka::onHost::data(resultDevice);
        control.particleCount = particleCount;
        control.seed = options.seed;
        control.maxVisits = 1'000'000u;
        control.errorWord = 0u;
        *alpaka::onHost::data(controlBuffer) = control;
        alpaka::onHost::wait(queue);

        graph::PoolConfig config{};
        config.packetSchema = graph::makeElementSchema<Particle>(alpaka::Vec<std::size_t, 1u>{chunkSize});
        config.chunkCapacity = chunkSize;
        config.sourceOperation = graph::particle_feedback::defaultOperationIds.generateParticles;
        config.budget.liveParticles = 256u;
        config.budget.controlAllowance = 64u;
        config.maxImportBatch = 16u;
        config.importPerIteration = 1u;
        config.idLimit = particleCount <= std::numeric_limits<std::uint32_t>::max()
                             ? static_cast<std::uint32_t>(particleCount)
                             : 0u;
        config.controlBlock = alpaka::onHost::data(controlBuffer);
        config.resultDevice = alpaka::onHost::data(resultDevice);
        config.resultBytes = particleCount * sizeof(ResultSlot);

        graph::Pool pool;
        try
        {
            pool = graph::startPool(device, executor, *compiled, config, ExampleBinder{});
        }
        catch(std::exception const& exception)
        {
            std::cerr << "graph_particle_feedback: pool startup failed: " << exception.what() << '\n';
            return 1;
        }

        graph::RequestSpec const request{0u, particleCount, options.seed};
        auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{&request, 1u});
        if(submitted.status != graph::PoolEnqueueStatus::accepted || submitted.accepted != 1u)
        {
            std::cerr << "graph_particle_feedback: source request was not accepted (status "
                      << static_cast<std::uint32_t>(submitted.status) << ")\n";
            pool.abort();
            (void) pool.waitFor(std::chrono::seconds{10});
            return 1;
        }
        if(pool.closeInput() != graph::PoolEnqueueStatus::accepted)
        {
            std::cerr << "graph_particle_feedback: ordered close was not accepted\n";
            pool.abort();
            (void) pool.waitFor(std::chrono::seconds{10});
            return 1;
        }

        auto const waited = pool.waitFor(std::chrono::seconds{200});
        if(!waited.has_value())
        {
            std::cerr << "graph_particle_feedback: drain did not finish within 200 s; aborting\n";
            pool.abort();
            (void) pool.waitFor(std::chrono::seconds{10});
            return 1;
        }
        auto const result = *waited;

        Outcomes outcomes;
        auto fetchOutcomes = [&]() -> bool
        {
            if(particleCount == 0u)
                return true;
            try
            {
                auto resultHost = alpaka::onHost::allocMapped<ResultSlot>(device, slotCount);
                pool.copyResultsToHost(
                    alpaka::onHost::data(resultHost),
                    static_cast<std::size_t>(particleCount) * sizeof(ResultSlot));
                pool.waitCopies();
                outcomes.weights.resize(static_cast<std::size_t>(particleCount));
                outcomes.visits.resize(static_cast<std::size_t>(particleCount));
                for(std::uint64_t id = 0u; id < particleCount; ++id)
                {
                    auto const& slot = alpaka::onHost::data(resultHost)[id];
                    outcomes.weights[static_cast<std::size_t>(id)] = slot.weight;
                    outcomes.visits[static_cast<std::size_t>(id)] = slot.visits;
                }
            }
            catch(std::exception const& exception)
            {
                std::cerr << "graph_particle_feedback: result copy failed: " << exception.what() << '\n';
                return false;
            }
            return true;
        };

        auto reportSinkVisits = [&]()
        {
            std::uint64_t missing = 0u;
            std::uint64_t duplicate = 0u;
            std::vector<std::uint64_t> firstMissing;
            for(std::uint64_t id = 0u; id < particleCount; ++id)
            {
                auto const visits = outcomes.visits[static_cast<std::size_t>(id)];
                if(visits == 0u)
                {
                    ++missing;
                    if(firstMissing.size() < 8u)
                        firstMissing.push_back(id);
                }
                else if(visits > 1u)
                {
                    ++duplicate;
                }
            }
            std::cerr << "graph_particle_feedback: sink visits missing=" << missing << " duplicate=" << duplicate;
            if(!firstMissing.empty())
            {
                std::cerr << " firstMissingIds=";
                for(std::size_t index = 0u; index < firstMissing.size(); ++index)
                    std::cerr << (index == 0u ? "" : ",") << firstMissing[index];
            }
            std::cerr << '\n';
        };

        auto printDiagnostics = [&](std::size_t histogramRows)
        {
            auto const geometry = pool.geometry();
            std::string transport = "unknown";
            try
            {
                auto capabilities = graph::inquireVerifiedSystemMemoryCapabilities(device);
                transport = std::string{graph::toString(graph::selectIngressTransport(capabilities))};
            }
            catch(std::exception const& exception)
            {
                transport = std::string{"unavailable ("} + exception.what() + ")";
            }
            std::cerr << "[graph_particle_feedback] backend=" << graph::particle_feedback::backendName(options.backend)
                      << " chunk=" << chunkSize << " particles=" << particleCount << " seed=" << options.seed << '\n';
            std::cerr << "[graph_particle_feedback] geometry blocks=" << geometry.blocks
                      << " threadsPerBlock=" << geometry.threadsPerBlock << " warpsPerBlock=" << geometry.warpsPerBlock
                      << " workers=" << geometry.workerCount << " residentsPerSm=" << geometry.residentsPerSm
                      << " sms=" << geometry.smCount << '\n';
            std::cerr << "[graph_particle_feedback] launchCount=" << result.counters.launchCount
                      << " distinctWorkers=" << result.counters.distinctWorkers
                      << " completedJobs=" << result.counters.completedJobs
                      << " outstanding=" << result.counters.outstanding
                      << " pendingSources=" << result.counters.pendingSources << " selectedTransport=" << transport
                      << '\n';
            auto const* controlView = alpaka::onHost::data(controlBuffer);
            std::cerr << "[graph_particle_feedback] generateElements=" << controlView->generateElements
                      << " processElements=" << controlView->processElements
                      << " destroyElements=" << controlView->destroyElements
                      << " destroyMarked=" << controlView->destroyMarked << '\n';
            std::uint64_t checksum = 1'469'598'103'934'665'603ull;
            for(auto const weight : outcomes.weights)
            {
                checksum ^= static_cast<std::uint64_t>(weight);
                checksum *= 1'099'511'628'211ull;
            }
            std::cerr << "[graph_particle_feedback] fnv1a=" << std::hex << std::uppercase << checksum << std::dec
                      << " histogramRows=" << histogramRows << '\n';
        };

        bool const haveResults = fetchOutcomes();

        if(result.reason != graph::PoolStopReason::stoppedByDrain)
        {
            std::cerr << "graph_particle_feedback: pool stopped with reason "
                      << static_cast<std::uint32_t>(result.reason) << " (errorWord " << result.errorWord << ")\n";
            if(haveResults)
                reportSinkVisits();
            if(verbose)
                printDiagnostics(0u);
            return 1;
        }
        if(result.errorWord != 0u)
        {
            std::cerr << "graph_particle_feedback: pool reported error word " << result.errorWord << '\n';
            if(verbose)
                printDiagnostics(0u);
            return 1;
        }
        if(result.counters.launchCount != 1u)
        {
            std::cerr << "graph_particle_feedback: expected one persistent launch, observed "
                      << result.counters.launchCount << '\n';
            return 1;
        }
        if(result.counters.outstanding != 0u || result.counters.pendingSources != 0u)
        {
            std::cerr << "graph_particle_feedback: drain accounting is not quiescent (outstanding "
                      << result.counters.outstanding << ", pendingSources " << result.counters.pendingSources << ")\n";
            return 1;
        }
        if(!haveResults)
            return 1;

        auto const* hostControl = alpaka::onHost::data(controlBuffer);
        if(hostControl->errorWord != 0u)
        {
            std::cerr << "graph_particle_feedback: particle operations reported error word " << hostControl->errorWord
                      << '\n';
            if(verbose)
                printDiagnostics(0u);
            return 1;
        }

        auto const histogram = graph::buildHistogram(
            std::span<std::int64_t const>{outcomes.weights},
            std::span<std::uint32_t const>{outcomes.visits},
            particleCount);
        if(!histogram)
        {
            std::cerr << "graph_particle_feedback: result validation failed: " << histogram.error << '\n';
            reportSinkVisits();
            if(verbose)
                printDiagnostics(0u);
            return 1;
        }
        if(histogram.histogram.totalCount() != particleCount)
        {
            std::cerr << "graph_particle_feedback: histogram total " << histogram.histogram.totalCount()
                      << " does not equal N=" << particleCount << '\n';
            return 1;
        }

        std::cout << histogram.histogram.render();
        std::cout.flush();

        if(verbose)
            printDiagnostics(histogram.histogram.rows.size());

        return 0;
    }

    int run(Options const& options, bool verbose)
    {
        if(options.backend == Backend::Cuda)
        {
#if GRAPH_HAS_CUDA
            auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
            auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
            if(selector.getDeviceCount() == 0u)
            {
                std::cerr << "graph_particle_feedback: no NVIDIA CUDA device is visible\n";
                return 1;
            }
            auto device = selector.makeDevice(0);
            auto executor = alpaka::onHost::defaultExecutor(device);
            return runExample(device, executor, options, verbose);
#else
            std::cerr << "graph_particle_feedback: backend 'cuda' is not available in this build "
                         "(configure with -DGRAPH_ENABLE_CUDA=ON)\n";
            return 1;
#endif
        }

        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(selector.getDeviceCount() == 0u)
        {
            std::cerr << "graph_particle_feedback: no host CPU device is available\n";
            return 1;
        }
        auto device = selector.makeDevice(0);
        auto executor = alpaka::onHost::defaultExecutor(device);
        return runExample(device, executor, options, verbose);
    }
} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string_view> arguments;
    arguments.reserve(argc > 0 ? static_cast<std::size_t>(argc - 1) : 0u);
    bool verbose = false;
    for(int index = 1; index < argc; ++index)
    {
        std::string_view const argument{argv[index]};
        if(argument == "--verbose")
            verbose = true;
        else
            arguments.emplace_back(argument);
    }

    auto const parsed
        = graph::particle_feedback::parse(std::span<std::string_view const>{arguments.data(), arguments.size()});
    if(parsed.help)
    {
        std::cout << graph::particle_feedback::usageText;
        std::cout << "  --verbose            Print pool diagnostics to stderr\n";
        return 0;
    }
    if(!parsed.ok)
    {
        std::cerr << "graph_particle_feedback: " << parsed.error << '\n';
        return 1;
    }
    return run(parsed.options, verbose);
}
