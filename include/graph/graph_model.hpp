// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/stream_id.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/** Host-side graph description, validation, and compiled scheduler tables.
 *
 * This header is host-only. It contains the *description* of a task graph and
 * the validation that turns it into a `CompiledGraph`, which is the immutable
 * table set consumed by the host scheduler.
 *
 * # What may travel to device code
 *
 * `CompiledGraph` and the plain table entries it contains (`CompiledNode`,
 * `CompiledEdge`, `CompiledStream`, `CompiledPortBinding`) are trivially
 * copyable aggregates with no owning handles. They may be copied to
 * device-visible storage. `NodeDescriptor`, `EdgeDescriptor`, `StreamDescriptor`,
 * `GraphError`, and `GraphModel` use `std::string`/`std::vector`, are build-time
 * only, and must never cross the device boundary.
 *
 * # Edge schema compatibility rule
 *
 * An edge is legal when the writer's element layout can be consumed by the
 * reader. Concretely, with `W` the producer output schema and `R` the consumer
 * input schema:
 *
 * - `W.elementType == R.elementType` (exact element type id, not merely the
 *   same size or alignment);
 * - `W.rank == R.rank`;
 * - every dimension has `W.extents[d] == R.extents[d]`;
 * - `W.pitchBytes[d] % W.elementSize == 0` and
 *   `R.pitchBytes[d] % R.elementSize == 0` for every dimension (a pitch is a
 *   whole number of elements);
 * - the innermost dimension pitches are equal:
 *   `W.pitchBytes[rank - 1] == R.pitchBytes[rank - 1]`.
 *
 * Outer pitches are allowed to differ as long as they are element-size
 * multiples; the producer may therefore hand a consumer a differently strided
 * outer layout with the same logical shape.
 *
 * # Cycles
 *
 * A cycle is legal only when at least one edge of the cycle carries
 * `EdgeDescriptor::feedback == true`. Cycle detection runs over the subgraph of
 * non-feedback edges, so any remaining cycle is an undeclared back edge and is
 * reported as `GraphErrorCode::UndeclaredCycle`. Declared feedback cycles are
 * accepted.
 */
namespace graph
{
    /** Strong identifier of a node inside one `GraphModel`. */
    struct NodeId
    {
        using ValueType = std::uint32_t;

        static constexpr ValueType invalidValue = ~ValueType{0u};

        ValueType value{invalidValue};

        constexpr NodeId() noexcept = default;

        constexpr explicit NodeId(ValueType id) noexcept : value(id)
        {
        }

        static constexpr NodeId invalid() noexcept
        {
            return NodeId{};
        }

        constexpr bool isValid() const noexcept
        {
            return value != invalidValue;
        }

        friend constexpr bool operator==(NodeId const&, NodeId const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<NodeId>);
    static_assert(std::is_standard_layout_v<NodeId>);

    /** Classification of a graph build error. */
    enum class GraphErrorCode : std::uint32_t
    {
        None = 0u,
        CapacityExceeded,
        InvalidNodeName,
        DuplicateNodeName,
        InvalidOperationId,
        InvalidSchema,
        TooManyPorts,
        DuplicatePort,
        DuplicateStream,
        InvalidStream,
        UnknownNode,
        UnknownPort,
        UnknownStream,
        StreamMismatch,
        DuplicateEdge,
        SelfEdgeWithoutFeedback,
        IncompatibleSchema,
        UndeclaredCycle,
        MissingSource,
        MissingSink,
        DanglingPort,
        UnreachableNode
    };

    /** One structured build error.
     *
     * `writerPort` and `readerPort` are filled for edge errors that involve two
     * ports. They use the labels `Node.output[N]` and `Node.input[N]` so a
     * failing build names the exact ports without requiring the caller to map
     * indices. `message` is a human-readable summary.
     *
     * The whole type is build-time only; it owns strings and never travels to
     * device code.
     */
    struct GraphError
    {
        GraphErrorCode code{GraphErrorCode::None};
        std::string writerPort;
        std::string readerPort;
        std::string message;
    };

    /** One ordered input port of a node. Input ports do not name a stream:
     * the incoming edges supply the concrete stream mapping. All edges that
     * target the same input port must agree on that mapping.
     */
    struct InputPortDescriptor
    {
        ElementSchema schema{};
    };

    /** One ordered output port of a node and its destination routing.
     *
     * `stream`/`streamPort` name the registered stream the port writes to.
     * Every outgoing edge of the port must repeat the same mapping.
     */
    struct OutputPortDescriptor
    {
        ElementSchema schema{};
        StreamId stream{};
        std::uint32_t streamPort{0u};
    };

    /** Host-side node registration description. */
    struct NodeDescriptor
    {
        std::string name;
        OperationId operation{};
        std::vector<InputPortDescriptor> inputs;
        std::vector<OutputPortDescriptor> outputs;
    };

    /** Host-side edge description.
     *
     * `feedback == true` declares a back edge. A cycle that contains such an
     * edge is accepted; a cycle made only of ordinary edges is rejected.
     */
    struct EdgeDescriptor
    {
        NodeId from{};
        std::uint32_t outputPort{0u};
        NodeId to{};
        std::uint32_t inputPort{0u};
        StreamId stream{};
        std::uint32_t streamPort{0u};
        bool feedback{false};
    };

    /** A registered stream and its fixed-capacity chunk budget.
     *
     * `chunkCapacity` is the number of elements a single physical chunk holds
     * and `chunkCount` is the number of physical chunks the stream keeps
     * (two for a double-buffered assembly stream, one for a directly consumed
     * terminal stream).
     */
    struct StreamDescriptor
    {
        StreamId id{};
        std::uint32_t port{0u};
        std::uint32_t chunkCapacity{0u};
        std::uint32_t chunkCount{1u};

        constexpr std::uint32_t capacity() const noexcept
        {
            return chunkCapacity * chunkCount;
        }
    };

    /** Fixed capacities of the compiled tables.
     *
     * These bound the size of a `CompiledGraph` so the scheduler can hold it in
     * plain storage. They intentionally mirror the operation and port limits of
     * the packet vocabulary.
     */
    struct GraphLimits
    {
        static constexpr std::uint32_t maxNodes = 32u;
        static constexpr std::uint32_t maxPortsPerNode = maxPacketPorts;
        static constexpr std::uint32_t maxStreams = 32u;
        static constexpr std::uint32_t maxEdges = 128u;
        static constexpr std::uint32_t maxStreamBindings = maxNodes * maxPortsPerNode;
    };

    /** Compiled node entry consumed by the host scheduler. */
    struct CompiledNode
    {
        NodeId id{};
        OperationId operation{};
        std::uint32_t inputPortCount{0u};
        std::uint32_t outputPortCount{0u};
        std::uint32_t firstInputEdge{0u};
        std::uint32_t inputEdgeCount{0u};
        std::uint32_t firstOutputEdge{0u};
        std::uint32_t outputEdgeCount{0u};
        bool isSource{false};
        bool isSink{false};
    };

    /** Compiled edge entry consumed by the host scheduler. */
    struct CompiledEdge
    {
        NodeId from{};
        std::uint32_t outputPort{0u};
        NodeId to{};
        std::uint32_t inputPort{0u};
        StreamId stream{};
        std::uint32_t streamPort{0u};
        bool feedback{false};
    };

    /** Reference to a node port, used by per-stream producer/consumer tables. */
    struct CompiledPortBinding
    {
        NodeId node{};
        std::uint32_t port{0u};
    };

    /** One compiled stream with ranges into the flat binding tables. */
    struct CompiledStream
    {
        StreamId id{};
        std::uint32_t port{0u};
        std::uint32_t chunkCapacity{0u};
        std::uint32_t chunkCount{0u};
        std::uint32_t firstProducer{0u};
        std::uint32_t producerCount{0u};
        std::uint32_t firstConsumer{0u};
        std::uint32_t consumerCount{0u};

        constexpr std::uint32_t capacity() const noexcept
        {
            return chunkCapacity * chunkCount;
        }
    };

    /** Immutable, trivially copyable graph tables for the scheduler.
     *
     * - `nodes` carry operation ids and edge ranges (`inputEdges` /
     *   `outputEdges` hold indices into `edges`).
     * - `edges` carry the concrete stream mapping and feedback flag.
     * - `streams` carry capacities plus ranges into `streamProducers` and
     *   `streamConsumers`.
     * - `sources` and `sinks` list the terminal classes of nodes.
     *
     * All entries are value types with no owning storage. The whole struct may
     * be copied into device-visible memory if a later scheduler stage needs it
     * there; it must not be mutated after `buildGraph()` returned it.
     */
    struct CompiledGraph
    {
        std::array<CompiledNode, GraphLimits::maxNodes> nodes{};
        std::uint32_t nodeCount{0u};

        std::array<CompiledEdge, GraphLimits::maxEdges> edges{};
        std::uint32_t edgeCount{0u};

        std::array<std::uint32_t, GraphLimits::maxEdges> inputEdges{};
        std::array<std::uint32_t, GraphLimits::maxEdges> outputEdges{};

        std::array<CompiledStream, GraphLimits::maxStreams> streams{};
        std::uint32_t streamCount{0u};

        std::array<CompiledPortBinding, GraphLimits::maxStreamBindings> streamProducers{};
        std::uint32_t streamProducerCount{0u};

        std::array<CompiledPortBinding, GraphLimits::maxStreamBindings> streamConsumers{};
        std::uint32_t streamConsumerCount{0u};

        std::array<NodeId, GraphLimits::maxNodes> sources{};
        std::uint32_t sourceCount{0u};

        std::array<NodeId, GraphLimits::maxNodes> sinks{};
        std::uint32_t sinkCount{0u};
    };

    static_assert(std::is_trivially_copyable_v<CompiledNode>);
    static_assert(std::is_standard_layout_v<CompiledNode>);
    static_assert(std::is_trivially_copyable_v<CompiledEdge>);
    static_assert(std::is_standard_layout_v<CompiledEdge>);
    static_assert(std::is_trivially_copyable_v<CompiledStream>);
    static_assert(std::is_standard_layout_v<CompiledStream>);
    static_assert(std::is_trivially_copyable_v<CompiledPortBinding>);
    static_assert(std::is_standard_layout_v<CompiledPortBinding>);
    static_assert(std::is_trivially_copyable_v<CompiledGraph>);
    static_assert(std::is_standard_layout_v<CompiledGraph>);

    /** Either a validated graph or the structured error list. */
    using GraphBuildOutcome = std::variant<CompiledGraph, std::vector<GraphError>>;

    /** True when `buildGraph()` produced a validated graph. */
    inline bool succeeded(GraphBuildOutcome const& outcome) noexcept
    {
        return std::holds_alternative<CompiledGraph>(outcome);
    }

    /** Edge schema compatibility as documented at the top of this header.
     *
     * The writer's layout is consumable by the reader when the element type,
     * rank, and per-dimension extents match, every pitch is a whole number of
     * elements, and the innermost pitches are equal.
     */
    ALPAKA_FN_HOST constexpr bool schemasConsumableBy(
        ElementSchema const& writer,
        ElementSchema const& reader) noexcept
    {
        if(writer.elementType != reader.elementType)
            return false;
        if(writer.rank != reader.rank)
            return false;
        if(writer.elementSize == 0u || writer.elementSize != reader.elementSize)
            return false;

        if(writer.rank == 0u)
            return true;
        if(writer.rank > maxSchemaRank)
            return false;

        for(std::size_t d = 0u; d < writer.rank; ++d)
        {
            if(writer.extents[d] != reader.extents[d])
                return false;
            if(writer.pitchBytes[d] % writer.elementSize != 0u)
                return false;
            if(reader.pitchBytes[d] % reader.elementSize != 0u)
                return false;
        }

        return writer.pitchBytes[writer.rank - 1u] == reader.pitchBytes[reader.rank - 1u];
    }

    namespace detail
    {
        inline std::string inputLabel(std::string const& node, std::uint32_t port)
        {
            return node + ".input[" + std::to_string(port) + "]";
        }

        inline std::string outputLabel(std::string const& node, std::uint32_t port)
        {
            return node + ".output[" + std::to_string(port) + "]";
        }

        inline GraphError makeError(
            GraphErrorCode code,
            std::string message,
            std::string writerPort = {},
            std::string readerPort = {})
        {
            GraphError error;
            error.code = code;
            error.writerPort = std::move(writerPort);
            error.readerPort = std::move(readerPort);
            error.message = std::move(message);
            return error;
        }
    } // namespace detail

    /** Mutable host-side graph builder.
     *
     * `addNode()`, `addStream()`, and `addEdge()` only append declarations and
     * enforce the fixed table capacities. All semantic validation (ports,
     * schemas, stream bindings, duplicates, cycles, source/sink class) happens
     * in `buildGraph()`, which never throws and returns structured errors.
     */
    class GraphModel
    {
    public:
        /** Append a node. Returns the new id, or an invalid id when the fixed
         * node capacity is exhausted (the failure is retained as an error).
         */
        NodeId addNode(NodeDescriptor descriptor)
        {
            if(m_nodes.size() >= GraphLimits::maxNodes)
            {
                m_earlyErrors.push_back(detail::makeError(GraphErrorCode::CapacityExceeded, "too many graph nodes"));
                return NodeId::invalid();
            }
            m_nodes.push_back(std::move(descriptor));
            return NodeId{static_cast<std::uint32_t>(m_nodes.size() - 1u)};
        }

        /** Append a stream declaration. */
        bool addStream(StreamDescriptor descriptor)
        {
            if(m_streams.size() >= GraphLimits::maxStreams)
            {
                m_earlyErrors.push_back(detail::makeError(GraphErrorCode::CapacityExceeded, "too many graph streams"));
                return false;
            }
            m_streams.push_back(descriptor);
            return true;
        }

        /** Append an edge declaration. */
        bool addEdge(EdgeDescriptor descriptor)
        {
            if(m_edges.size() >= GraphLimits::maxEdges)
            {
                m_earlyErrors.push_back(detail::makeError(GraphErrorCode::CapacityExceeded, "too many graph edges"));
                return false;
            }
            m_edges.push_back(descriptor);
            return true;
        }

        std::vector<NodeDescriptor> const& nodes() const noexcept
        {
            return m_nodes;
        }

        std::vector<EdgeDescriptor> const& edges() const noexcept
        {
            return m_edges;
        }

        std::vector<StreamDescriptor> const& streams() const noexcept
        {
            return m_streams;
        }

        /** Validate the description and produce the compiled scheduler tables.
         *
         * Nothing is thrown. On failure the returned alternative contains every
         * error that could be collected in one pass.
         */
        GraphBuildOutcome buildGraph() const;

    private:
        std::vector<NodeDescriptor> m_nodes;
        std::vector<EdgeDescriptor> m_edges;
        std::vector<StreamDescriptor> m_streams;
        std::vector<GraphError> m_earlyErrors;

        std::string const& nodeName(NodeId id) const
        {
            static std::string const empty;
            if(id.value >= m_nodes.size())
                return empty;
            return m_nodes[id.value].name;
        }

        bool findStream(StreamId id, std::uint32_t port) const noexcept
        {
            for(auto const& stream : m_streams)
            {
                if(stream.id == id)
                    return stream.port == port;
            }
            return false;
        }

        bool hasStreamId(StreamId id) const noexcept
        {
            for(auto const& stream : m_streams)
            {
                if(stream.id == id)
                    return true;
            }
            return false;
        }
    };

    inline GraphBuildOutcome GraphModel::buildGraph() const
    {
        std::vector<GraphError> errors = m_earlyErrors;

        auto report = [&errors](
                          GraphErrorCode code,
                          std::string message,
                          std::string writerPort = {},
                          std::string readerPort = {})
        {
            errors.push_back(
                detail::makeError(code, std::move(message), std::move(writerPort), std::move(readerPort)));
        };

        // Fixed table capacities are errors, not a reason to index out of bounds.
        if(m_nodes.size() > GraphLimits::maxNodes)
            report(GraphErrorCode::CapacityExceeded, "too many graph nodes");
        if(m_edges.size() > GraphLimits::maxEdges)
            report(GraphErrorCode::CapacityExceeded, "too many graph edges");
        if(m_streams.size() > GraphLimits::maxStreams)
            report(GraphErrorCode::CapacityExceeded, "too many graph streams");

        auto const nodeCount = static_cast<std::uint32_t>(m_nodes.size());

        // Nodes: names, operation ids, port schemas, and duplicate output routing.
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            auto const& node = m_nodes[nodeIndex];
            auto const nodeLabel = node.name.empty() ? std::string{"<unnamed>"} : node.name;

            if(node.name.empty())
                report(GraphErrorCode::InvalidNodeName, "node " + std::to_string(nodeIndex) + " has an empty name");

            for(std::uint32_t other = 0u; other < nodeIndex; ++other)
            {
                if(!node.name.empty() && m_nodes[other].name == node.name)
                    report(GraphErrorCode::DuplicateNodeName, "duplicate node name '" + node.name + "'", node.name);
            }

            if(!node.operation.isValid())
                report(
                    GraphErrorCode::InvalidOperationId,
                    "node '" + nodeLabel + "' has an invalid operation id",
                    nodeLabel);

            if(node.inputs.size() > GraphLimits::maxPortsPerNode)
                report(GraphErrorCode::TooManyPorts, "node '" + nodeLabel + "' has too many input ports", nodeLabel);
            if(node.outputs.size() > GraphLimits::maxPortsPerNode)
                report(GraphErrorCode::TooManyPorts, "node '" + nodeLabel + "' has too many output ports", nodeLabel);

            for(std::uint32_t port = 0u; port < node.inputs.size(); ++port)
            {
                if(!isValid(node.inputs[port].schema) || node.inputs[port].schema.rank == 0u)
                    report(
                        GraphErrorCode::InvalidSchema,
                        "input port '" + detail::inputLabel(nodeLabel, port) + "' has an invalid schema",
                        detail::inputLabel(nodeLabel, port));
            }

            for(std::uint32_t port = 0u; port < node.outputs.size(); ++port)
            {
                if(!isValid(node.outputs[port].schema) || node.outputs[port].schema.rank == 0u)
                    report(
                        GraphErrorCode::InvalidSchema,
                        "output port '" + detail::outputLabel(nodeLabel, port) + "' has an invalid schema",
                        detail::outputLabel(nodeLabel, port));
            }

            for(std::uint32_t port = 0u; port < node.outputs.size(); ++port)
            {
                for(std::uint32_t other = 0u; other < port; ++other)
                {
                    if(node.outputs[port].stream == node.outputs[other].stream
                       && node.outputs[port].streamPort == node.outputs[other].streamPort)
                    {
                        report(
                            GraphErrorCode::DuplicatePort,
                            "node '" + nodeLabel + "' routes output ports " + std::to_string(other) + " and "
                                + std::to_string(port) + " to the same stream port",
                            detail::outputLabel(nodeLabel, port),
                            detail::outputLabel(nodeLabel, other));
                    }
                }
            }
        }

        // Streams: ids and capacities.
        for(std::uint32_t streamIndex = 0u; streamIndex < m_streams.size(); ++streamIndex)
        {
            auto const& stream = m_streams[streamIndex];
            if(!stream.id.isValid())
                report(GraphErrorCode::InvalidStream, "stream " + std::to_string(streamIndex) + " has an invalid id");
            if(stream.chunkCapacity == 0u || stream.chunkCount == 0u)
                report(
                    GraphErrorCode::InvalidStream,
                    "stream " + std::to_string(streamIndex) + " has a zero chunk capacity or count");
            for(std::uint32_t other = 0u; other < streamIndex; ++other)
            {
                if(m_streams[other].id == stream.id)
                    report(GraphErrorCode::DuplicateStream, "duplicate stream id " + std::to_string(stream.id.value));
            }
        }

        // Every declared output routing must name a registered stream port.
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            auto const& node = m_nodes[nodeIndex];
            auto const nodeLabel = node.name.empty() ? std::string{"<unnamed>"} : node.name;
            for(std::uint32_t port = 0u; port < node.outputs.size(); ++port)
            {
                auto const& output = node.outputs[port];
                if(!hasStreamId(output.stream))
                {
                    report(
                        GraphErrorCode::UnknownStream,
                        "output port '" + detail::outputLabel(nodeLabel, port) + "' routes to an unknown stream",
                        detail::outputLabel(nodeLabel, port));
                }
                else if(!findStream(output.stream, output.streamPort))
                {
                    report(
                        GraphErrorCode::StreamMismatch,
                        "output port '" + detail::outputLabel(nodeLabel, port) + "' routes to an unknown stream port",
                        detail::outputLabel(nodeLabel, port));
                }
            }
        }

        // Edges: endpoints, duplicates, self edges, stream mapping, and schemas.
        std::vector<bool> inputPortConnected(nodeCount * GraphLimits::maxPortsPerNode, false);
        std::vector<bool> outputPortConnected(nodeCount * GraphLimits::maxPortsPerNode, false);

        for(std::uint32_t edgeIndex = 0u; edgeIndex < m_edges.size(); ++edgeIndex)
        {
            auto const& edge = m_edges[edgeIndex];
            if(edge.from.value >= nodeCount || edge.to.value >= nodeCount)
            {
                report(
                    GraphErrorCode::UnknownNode,
                    "edge " + std::to_string(edgeIndex) + " references an unknown node");
                continue;
            }

            auto const& fromNode = m_nodes[edge.from.value];
            auto const& toNode = m_nodes[edge.to.value];
            auto const fromLabel = fromNode.name.empty() ? std::string{"<unnamed>"} : fromNode.name;
            auto const toLabel = toNode.name.empty() ? std::string{"<unnamed>"} : toNode.name;

            if(edge.outputPort >= fromNode.outputs.size() || edge.inputPort >= toNode.inputs.size())
            {
                report(
                    GraphErrorCode::UnknownPort,
                    "edge " + std::to_string(edgeIndex) + " references an unknown port",
                    detail::outputLabel(fromLabel, edge.outputPort),
                    detail::inputLabel(toLabel, edge.inputPort));
                continue;
            }

            auto const writerLabel = detail::outputLabel(fromLabel, edge.outputPort);
            auto const readerLabel = detail::inputLabel(toLabel, edge.inputPort);

            for(std::uint32_t other = 0u; other < edgeIndex; ++other)
            {
                auto const& previous = m_edges[other];
                if(previous.from == edge.from && previous.outputPort == edge.outputPort && previous.to == edge.to
                   && previous.inputPort == edge.inputPort)
                {
                    report(
                        GraphErrorCode::DuplicateEdge,
                        "duplicate edge from '" + writerLabel + "' to '" + readerLabel + "'",
                        writerLabel,
                        readerLabel);
                }
            }

            if(edge.from == edge.to && !edge.feedback)
            {
                report(
                    GraphErrorCode::SelfEdgeWithoutFeedback,
                    "self edge on '" + writerLabel + "' is not declared as a feedback edge",
                    writerLabel,
                    readerLabel);
            }

            auto const& writer = fromNode.outputs[edge.outputPort];
            if(writer.stream != edge.stream || writer.streamPort != edge.streamPort)
            {
                report(
                    GraphErrorCode::StreamMismatch,
                    "edge from '" + writerLabel + "' to '" + readerLabel
                        + "' does not match the writer's output routing",
                    writerLabel,
                    readerLabel);
            }
            if(!findStream(edge.stream, edge.streamPort))
            {
                report(
                    GraphErrorCode::UnknownStream,
                    "edge from '" + writerLabel + "' to '" + readerLabel + "' selects an unknown stream port",
                    writerLabel,
                    readerLabel);
            }
            else
            {
                // All edges into one input port must select the same stream port,
                // otherwise the input port would consume two different streams.
                for(std::uint32_t other = 0u; other < edgeIndex; ++other)
                {
                    auto const& previous = m_edges[other];
                    if(previous.to == edge.to && previous.inputPort == edge.inputPort
                       && (previous.stream != edge.stream || previous.streamPort != edge.streamPort))
                    {
                        report(
                            GraphErrorCode::StreamMismatch,
                            "input port '" + readerLabel + "' receives edges from different stream ports",
                            writerLabel,
                            readerLabel);
                    }
                }
            }

            auto const& writerSchema = fromNode.outputs[edge.outputPort].schema;
            auto const& readerSchema = toNode.inputs[edge.inputPort].schema;
            if(!schemasConsumableBy(writerSchema, readerSchema))
            {
                report(
                    GraphErrorCode::IncompatibleSchema,
                    "schema mismatch: '" + writerLabel + "' cannot be consumed by '" + readerLabel
                        + "' (element type, rank, extents, or pitch differ)",
                    writerLabel,
                    readerLabel);
            }

            if(edge.outputPort < GraphLimits::maxPortsPerNode)
                outputPortConnected[edge.from.value * GraphLimits::maxPortsPerNode + edge.outputPort] = true;
            if(edge.inputPort < GraphLimits::maxPortsPerNode)
                inputPortConnected[edge.to.value * GraphLimits::maxPortsPerNode + edge.inputPort] = true;
        }

        // Dangling ports would leave the scheduler with a port that never fires.
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            auto const& node = m_nodes[nodeIndex];
            auto const nodeLabel = node.name.empty() ? std::string{"<unnamed>"} : node.name;
            if(node.outputs.empty() && node.inputs.empty())
                report(
                    GraphErrorCode::UnreachableNode,
                    "node '" + nodeLabel + "' has no ports and cannot participate in the graph",
                    nodeLabel);

            for(std::uint32_t port = 0u; port < node.inputs.size(); ++port)
            {
                auto const connected = port < GraphLimits::maxPortsPerNode
                                       && inputPortConnected[nodeIndex * GraphLimits::maxPortsPerNode + port];
                if(!connected)
                    report(
                        GraphErrorCode::DanglingPort,
                        "input port '" + detail::inputLabel(nodeLabel, port) + "' has no incoming edge",
                        detail::inputLabel(nodeLabel, port));
            }
            for(std::uint32_t port = 0u; port < node.outputs.size(); ++port)
            {
                auto const connected = port < GraphLimits::maxPortsPerNode
                                       && outputPortConnected[nodeIndex * GraphLimits::maxPortsPerNode + port];
                if(!connected)
                    report(
                        GraphErrorCode::DanglingPort,
                        "output port '" + detail::outputLabel(nodeLabel, port) + "' has no outgoing edge",
                        detail::outputLabel(nodeLabel, port));
            }
        }

        // Source and sink classes.
        std::uint32_t sourceCount = 0u;
        std::uint32_t sinkCount = 0u;
        for(auto const& node : m_nodes)
        {
            if(node.inputs.empty())
                ++sourceCount;
            if(node.outputs.empty())
                ++sinkCount;
        }
        if(sourceCount == 0u)
        {
            auto const name = m_nodes.empty() ? std::string{"<empty graph>"} : m_nodes.front().name;
            report(
                GraphErrorCode::MissingSource,
                "graph has no source node" + (m_nodes.empty() ? std::string{} : ": first node '" + name + "'"));
        }
        if(sinkCount == 0u)
        {
            auto const name = m_nodes.empty() ? std::string{"<empty graph>"} : m_nodes.front().name;
            report(
                GraphErrorCode::MissingSink,
                "graph has no sink node" + (m_nodes.empty() ? std::string{} : ": first node '" + name + "'"));
        }

        // Adjacency over valid edges for reachability and cycle checks.
        std::vector<std::vector<std::uint32_t>> forward(nodeCount);
        std::vector<std::vector<std::uint32_t>> backward(nodeCount);
        for(auto const& edge : m_edges)
        {
            if(edge.from.value >= nodeCount || edge.to.value >= nodeCount)
                continue;
            forward[edge.from.value].push_back(edge.to.value);
            backward[edge.to.value].push_back(edge.from.value);
        }

        auto const reachableFrom = [&forward, nodeCount](std::vector<std::uint32_t> const& roots)
        {
            std::vector<bool> seen(nodeCount, false);
            std::vector<std::uint32_t> stack;
            for(auto root : roots)
            {
                if(root < nodeCount && !seen[root])
                {
                    seen[root] = true;
                    stack.push_back(root);
                }
            }
            while(!stack.empty())
            {
                auto const current = stack.back();
                stack.pop_back();
                for(auto const next : forward[current])
                {
                    if(!seen[next])
                    {
                        seen[next] = true;
                        stack.push_back(next);
                    }
                }
            }
            return seen;
        };

        auto const reachableTo = [&backward, nodeCount](std::vector<std::uint32_t> const& roots)
        {
            std::vector<bool> seen(nodeCount, false);
            std::vector<std::uint32_t> stack;
            for(auto root : roots)
            {
                if(root < nodeCount && !seen[root])
                {
                    seen[root] = true;
                    stack.push_back(root);
                }
            }
            while(!stack.empty())
            {
                auto const current = stack.back();
                stack.pop_back();
                for(auto const next : backward[current])
                {
                    if(!seen[next])
                    {
                        seen[next] = true;
                        stack.push_back(next);
                    }
                }
            }
            return seen;
        };

        std::vector<std::uint32_t> sourceIndices;
        std::vector<std::uint32_t> sinkIndices;
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            if(m_nodes[nodeIndex].inputs.empty())
                sourceIndices.push_back(nodeIndex);
            if(m_nodes[nodeIndex].outputs.empty())
                sinkIndices.push_back(nodeIndex);
        }

        auto const reachable = reachableFrom(sourceIndices);
        auto const reachingSink = reachableTo(sinkIndices);
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            auto const nodeLabel
                = m_nodes[nodeIndex].name.empty() ? std::string{"<unnamed>"} : m_nodes[nodeIndex].name;
            if(!reachable[nodeIndex])
                report(
                    GraphErrorCode::UnreachableNode,
                    "node '" + nodeLabel + "' is not reachable from any source",
                    nodeLabel);
            if(!reachingSink[nodeIndex])
                report(GraphErrorCode::UnreachableNode, "node '" + nodeLabel + "' cannot reach any sink", nodeLabel);
        }

        // Cycle detection over ordinary (non-feedback) edges only.
        std::vector<std::uint8_t> state(nodeCount, 0u);
        std::vector<GraphError> cycleErrors;
        auto const visit = [&](auto const& self, std::uint32_t nodeIndex) -> void
        {
            state[nodeIndex] = 1u;
            for(auto const& edge : m_edges)
            {
                if(edge.from.value != nodeIndex || edge.to.value >= nodeCount)
                    continue;
                // Feedback edges are the declared way to close a cycle.
                if(edge.feedback)
                    continue;
                if(state[edge.to.value] == 1u)
                {
                    auto const label
                        = m_nodes[edge.to.value].name.empty() ? std::string{"<unnamed>"} : m_nodes[edge.to.value].name;
                    cycleErrors.push_back(
                        detail::makeError(
                            GraphErrorCode::UndeclaredCycle,
                            "cycle through node '" + label + "' is not closed by a declared feedback edge",
                            detail::outputLabel(
                                m_nodes[nodeIndex].name.empty() ? "<unnamed>" : m_nodes[nodeIndex].name,
                                edge.outputPort),
                            detail::inputLabel(label, edge.inputPort)));
                    continue;
                }
                if(state[edge.to.value] == 0u)
                    self(self, edge.to.value);
            }
            state[nodeIndex] = 2u;
        };

        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            if(state[nodeIndex] == 0u)
                visit(visit, nodeIndex);
        }
        errors.insert(errors.end(), cycleErrors.begin(), cycleErrors.end());

        if(!errors.empty())
            return errors;

        // Everything validated: materialize the immutable tables.
        CompiledGraph compiled;
        compiled.nodeCount = nodeCount;
        compiled.edgeCount = static_cast<std::uint32_t>(m_edges.size());
        compiled.streamCount = static_cast<std::uint32_t>(m_streams.size());

        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            auto& entry = compiled.nodes[nodeIndex];
            entry.id = NodeId{nodeIndex};
            entry.operation = m_nodes[nodeIndex].operation;
            entry.inputPortCount = static_cast<std::uint32_t>(m_nodes[nodeIndex].inputs.size());
            entry.outputPortCount = static_cast<std::uint32_t>(m_nodes[nodeIndex].outputs.size());
            entry.isSource = m_nodes[nodeIndex].inputs.empty();
            entry.isSink = m_nodes[nodeIndex].outputs.empty();
            if(entry.isSource)
                compiled.sources[compiled.sourceCount++] = NodeId{nodeIndex};
            if(entry.isSink)
                compiled.sinks[compiled.sinkCount++] = NodeId{nodeIndex};
        }

        for(std::uint32_t edgeIndex = 0u; edgeIndex < compiled.edgeCount; ++edgeIndex)
        {
            auto const& edge = m_edges[edgeIndex];
            auto& entry = compiled.edges[edgeIndex];
            entry.from = edge.from;
            entry.outputPort = edge.outputPort;
            entry.to = edge.to;
            entry.inputPort = edge.inputPort;
            entry.stream = edge.stream;
            entry.streamPort = edge.streamPort;
            entry.feedback = edge.feedback;
        }

        std::vector<std::uint32_t> inputCount(nodeCount, 0u);
        std::vector<std::uint32_t> outputCount(nodeCount, 0u);
        for(auto const& edge : m_edges)
        {
            ++inputCount[edge.to.value];
            ++outputCount[edge.from.value];
        }
        std::uint32_t inputOffset = 0u;
        std::uint32_t outputOffset = 0u;
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            compiled.nodes[nodeIndex].firstInputEdge = inputOffset;
            compiled.nodes[nodeIndex].inputEdgeCount = inputCount[nodeIndex];
            compiled.nodes[nodeIndex].firstOutputEdge = outputOffset;
            compiled.nodes[nodeIndex].outputEdgeCount = outputCount[nodeIndex];
            inputOffset += inputCount[nodeIndex];
            outputOffset += outputCount[nodeIndex];
        }

        std::vector<std::uint32_t> inputCursor(nodeCount, 0u);
        std::vector<std::uint32_t> outputCursor(nodeCount, 0u);
        for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
        {
            inputCursor[nodeIndex] = compiled.nodes[nodeIndex].firstInputEdge;
            outputCursor[nodeIndex] = compiled.nodes[nodeIndex].firstOutputEdge;
        }
        for(std::uint32_t edgeIndex = 0u; edgeIndex < compiled.edgeCount; ++edgeIndex)
        {
            auto const& edge = m_edges[edgeIndex];
            compiled.inputEdges[inputCursor[edge.to.value]++] = edgeIndex;
            compiled.outputEdges[outputCursor[edge.from.value]++] = edgeIndex;
        }

        // Per-stream producer and consumer bindings.
        for(std::uint32_t streamIndex = 0u; streamIndex < compiled.streamCount; ++streamIndex)
        {
            auto const& descriptor = m_streams[streamIndex];
            auto& stream = compiled.streams[streamIndex];
            stream.id = descriptor.id;
            stream.port = descriptor.port;
            stream.chunkCapacity = descriptor.chunkCapacity;
            stream.chunkCount = descriptor.chunkCount;

            stream.firstProducer = compiled.streamProducerCount;
            for(std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex)
            {
                auto const& node = m_nodes[nodeIndex];
                for(std::uint32_t port = 0u; port < node.outputs.size(); ++port)
                {
                    auto const& output = node.outputs[port];
                    if(output.stream == descriptor.id && output.streamPort == descriptor.port)
                    {
                        compiled.streamProducers[compiled.streamProducerCount++]
                            = CompiledPortBinding{NodeId{nodeIndex}, port};
                    }
                }
            }
            stream.producerCount = compiled.streamProducerCount - stream.firstProducer;

            stream.firstConsumer = compiled.streamConsumerCount;
            for(auto const& edge : m_edges)
            {
                if(edge.stream != descriptor.id || edge.streamPort != descriptor.port)
                    continue;
                bool duplicate = false;
                for(std::uint32_t binding = stream.firstConsumer; binding < compiled.streamConsumerCount; ++binding)
                {
                    auto const& existing = compiled.streamConsumers[binding];
                    if(existing.node == edge.to && existing.port == edge.inputPort)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if(!duplicate)
                {
                    compiled.streamConsumers[compiled.streamConsumerCount++]
                        = CompiledPortBinding{edge.to, edge.inputPort};
                }
            }
            stream.consumerCount = compiled.streamConsumerCount - stream.firstConsumer;
        }

        return compiled;
    }
} // namespace graph
