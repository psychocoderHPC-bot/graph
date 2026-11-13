// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/graph_model.hpp>
#include <graph/particle_functors.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Concrete graph description of the particle-feedback example (design §8).
 *
 * This header only describes topology, schemas, routing, and stream
 * capacities. It deliberately contains no particle algorithm, RNG, queueing, or
 * scheduling code: the functors and the pool consume the description produced
 * here.
 *
 * | Node                | Inputs                        | Outputs                          |
 * | ------------------- | ----------------------------- | -------------------------------- |
 * | `GenerateParticles` | none (source request)         | filter input                     |
 * | `FilterByProperty`  | filter input                  | Bin 0, Bin 1                     |
 * | `ProcessA`          | Bin 0                         | feedback, sink                   |
 * | `ProcessB`          | Bin 1                         | feedback, sink                   |
 * | `DestroyParticles`  | sink                          | none (terminal sink)             |
 *
 * `GenerateParticles`, `ProcessA`, and `ProcessB` all write the same shared
 * filter-input stream. The edges from `ProcessA`/`ProcessB` back to
 * `FilterByProperty` are declared feedback edges, so the resulting cycle is
 * legal. `ProcessA` and `ProcessB` also write a terminal sink stream consumed by
 * `DestroyParticles`.
 */
namespace graph
{
    namespace particle_feedback
    {
        /** Element carried by every stream of the example.
         *
         * The one `graph::Particle` type of `particle_functors.hpp` is reused
         * here so the schema a stream is declared with and the element the
         * functor reads and writes are the same type. The 24-byte layout
         * (`id`, `property`, `weight`, `visit`) is kept: the deterministic RNG
         * is a pure function of `(seed, id, visit, purpose)`, so no per-particle
         * random state is needed and no second, differently sized layout can
         * drift from the functor.
         */
        using Particle = graph::Particle;

        static_assert(sizeof(Particle) == 24u);
        static_assert(PacketElement<Particle>);
        static_assert(std::is_standard_layout_v<Particle>);

        /** Operation ids used by the example graph.
         *
         * A caller may substitute its own registry values; the defaults are
         * small, stable ids that keep the header self-contained.
         */
        struct OperationIds
        {
            OperationId generateParticles{OperationId{1u}};
            OperationId filterByProperty{OperationId{2u}};
            OperationId processA{OperationId{3u}};
            OperationId processB{OperationId{4u}};
            OperationId destroyParticles{OperationId{5u}};
        };

        inline constexpr OperationIds defaultOperationIds{};

        /** Logical streams of the example graph. */
        struct StreamIds
        {
            StreamId filterInput{StreamId{1u}};
            StreamId bin0{StreamId{2u}};
            StreamId bin1{StreamId{3u}};
            StreamId sink{StreamId{4u}};
        };

        inline constexpr StreamIds defaultStreamIds{};

        /** Stream port used by every example stream. */
        inline constexpr std::uint32_t defaultStreamPort = 0u;

        /** Number of physical chunks per double-buffered assembly stream. */
        inline constexpr std::uint32_t assemblyChunkCount = 2u;

        /** Number of physical chunks of the directly consumed terminal stream. */
        inline constexpr std::uint32_t terminalChunkCount = 1u;
    } // namespace particle_feedback

    /** Build the host-side particle-feedback graph.
     *
     * @param chunkSize number of elements per physical chunk (`Y` in the
     *                  design). It is carried into `chunkCapacity` of every
     *                  stream. A zero value yields a graph without nodes, which
     *                  `buildGraph()` rejects as missing source and sink.
     * @param operations operation-id registry used for the five nodes.
     * @param streams    logical stream-id registry used for the four streams.
     *
     * The returned `GraphModel` is a description. Call `buildGraph()` to
     * validate it and obtain the compiled scheduler tables.
     */
    inline GraphModel makeParticleFeedbackGraph(
        std::uint32_t chunkSize,
        particle_feedback::OperationIds const& operations = particle_feedback::defaultOperationIds,
        particle_feedback::StreamIds const& streams = particle_feedback::defaultStreamIds)
    {
        GraphModel model;
        if(chunkSize == 0u)
            return model;

        auto const schema = makeElementSchema<particle_feedback::Particle>(
            alpaka::Vec<std::size_t, 1u>{static_cast<std::size_t>(chunkSize)});

        model.addStream(
            StreamDescriptor{
                streams.filterInput,
                particle_feedback::defaultStreamPort,
                chunkSize,
                particle_feedback::assemblyChunkCount});
        model.addStream(
            StreamDescriptor{
                streams.bin0,
                particle_feedback::defaultStreamPort,
                chunkSize,
                particle_feedback::assemblyChunkCount});
        model.addStream(
            StreamDescriptor{
                streams.bin1,
                particle_feedback::defaultStreamPort,
                chunkSize,
                particle_feedback::assemblyChunkCount});
        model.addStream(
            StreamDescriptor{
                streams.sink,
                particle_feedback::defaultStreamPort,
                chunkSize,
                particle_feedback::terminalChunkCount});

        NodeDescriptor generate;
        generate.name = "GenerateParticles";
        generate.operation = operations.generateParticles;
        generate.outputs.push_back(
            OutputPortDescriptor{schema, streams.filterInput, particle_feedback::defaultStreamPort});

        NodeDescriptor filter;
        filter.name = "FilterByProperty";
        filter.operation = operations.filterByProperty;
        filter.inputs.push_back(InputPortDescriptor{schema});
        filter.outputs.push_back(OutputPortDescriptor{schema, streams.bin0, particle_feedback::defaultStreamPort});
        filter.outputs.push_back(OutputPortDescriptor{schema, streams.bin1, particle_feedback::defaultStreamPort});

        NodeDescriptor processA;
        processA.name = "ProcessA";
        processA.operation = operations.processA;
        processA.inputs.push_back(InputPortDescriptor{schema});
        processA.outputs.push_back(
            OutputPortDescriptor{schema, streams.filterInput, particle_feedback::defaultStreamPort});
        processA.outputs.push_back(OutputPortDescriptor{schema, streams.sink, particle_feedback::defaultStreamPort});

        NodeDescriptor processB;
        processB.name = "ProcessB";
        processB.operation = operations.processB;
        processB.inputs.push_back(InputPortDescriptor{schema});
        processB.outputs.push_back(
            OutputPortDescriptor{schema, streams.filterInput, particle_feedback::defaultStreamPort});
        processB.outputs.push_back(OutputPortDescriptor{schema, streams.sink, particle_feedback::defaultStreamPort});

        NodeDescriptor destroy;
        destroy.name = "DestroyParticles";
        destroy.operation = operations.destroyParticles;
        destroy.inputs.push_back(InputPortDescriptor{schema});

        auto const generateNode = model.addNode(std::move(generate));
        auto const filterNode = model.addNode(std::move(filter));
        auto const processANode = model.addNode(std::move(processA));
        auto const processBNode = model.addNode(std::move(processB));
        auto const destroyNode = model.addNode(std::move(destroy));

        auto const port = particle_feedback::defaultStreamPort;

        // Fresh particles enter the shared filter input.
        model.addEdge(EdgeDescriptor{generateNode, 0u, filterNode, 0u, streams.filterInput, port, false});

        // The filter splits into the two bin streams.
        model.addEdge(EdgeDescriptor{filterNode, 0u, processANode, 0u, streams.bin0, port, false});
        model.addEdge(EdgeDescriptor{filterNode, 1u, processBNode, 0u, streams.bin1, port, false});

        // Declared feedback: both processors return survivors to the filter input.
        model.addEdge(EdgeDescriptor{processANode, 0u, filterNode, 0u, streams.filterInput, port, true});
        model.addEdge(EdgeDescriptor{processBNode, 0u, filterNode, 0u, streams.filterInput, port, true});

        // Terminal particles go to the single sink.
        model.addEdge(EdgeDescriptor{processANode, 1u, destroyNode, 0u, streams.sink, port, false});
        model.addEdge(EdgeDescriptor{processBNode, 1u, destroyNode, 0u, streams.sink, port, false});

        return model;
    }
} // namespace graph
