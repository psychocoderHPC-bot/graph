// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/graph_model.hpp>
#include <graph/particle_feedback_graph.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace
{
    using graph::EdgeDescriptor;
    using graph::ElementSchema;
    using graph::GraphBuildOutcome;
    using graph::GraphError;
    using graph::GraphErrorCode;
    using graph::InputPortDescriptor;
    using graph::NodeDescriptor;
    using graph::NodeId;
    using graph::OutputPortDescriptor;
    using graph::StreamDescriptor;
    using graph::StreamId;

    ElementSchema floatSchema(std::size_t extent)
    {
        return graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 1u>{extent});
    }

    ElementSchema doubleSchema(std::size_t extent)
    {
        return graph::makeElementSchema<double>(alpaka::Vec<std::size_t, 1u>{extent});
    }

    InputPortDescriptor input(ElementSchema schema)
    {
        return InputPortDescriptor{schema};
    }

    OutputPortDescriptor output(ElementSchema schema, StreamId stream)
    {
        return OutputPortDescriptor{schema, stream, 0u};
    }

    NodeDescriptor makeNode(
        std::string name,
        std::uint32_t operation,
        std::vector<InputPortDescriptor> inputs,
        std::vector<OutputPortDescriptor> outputs)
    {
        NodeDescriptor node;
        node.name = std::move(name);
        node.operation = graph::OperationId{operation};
        node.inputs = std::move(inputs);
        node.outputs = std::move(outputs);
        return node;
    }

    EdgeDescriptor makeEdge(NodeId from, std::uint32_t outputPort, NodeId to, std::uint32_t inputPort, StreamId stream)
    {
        return EdgeDescriptor{from, outputPort, to, inputPort, stream, 0u, false};
    }

    EdgeDescriptor makeFeedbackEdge(
        NodeId from,
        std::uint32_t outputPort,
        NodeId to,
        std::uint32_t inputPort,
        StreamId stream)
    {
        return EdgeDescriptor{from, outputPort, to, inputPort, stream, 0u, true};
    }

    std::vector<GraphError> const* errorsOf(GraphBuildOutcome const& outcome)
    {
        return std::get_if<std::vector<GraphError>>(&outcome);
    }

    GraphError const* findError(std::vector<GraphError> const& errors, GraphErrorCode code)
    {
        auto const found = std::find_if(
            errors.begin(),
            errors.end(),
            [code](GraphError const& error) { return error.code == code; });
        return found == errors.end() ? nullptr : &*found;
    }
} // namespace

TEST_CASE("graph_model accepts and compiles a valid linear graph", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 8u, 2u}));

    auto const source = model.addNode(makeNode("Source", 1u, {}, {output(floatSchema(4u), stream)}));
    auto const sink = model.addNode(makeNode("Sink", 2u, {input(floatSchema(4u))}, {}));
    REQUIRE(model.addEdge(makeEdge(source, 0u, sink, 0u, stream)));

    auto outcome = model.buildGraph();
    REQUIRE(graph::succeeded(outcome));

    auto const& compiled = std::get<graph::CompiledGraph>(outcome);
    CHECK(compiled.nodeCount == 2u);
    CHECK(compiled.edgeCount == 1u);
    CHECK(compiled.streamCount == 1u);
    CHECK(compiled.sourceCount == 1u);
    CHECK(compiled.sinkCount == 1u);

    CHECK(compiled.nodes[0].isSource);
    CHECK_FALSE(compiled.nodes[0].isSink);
    CHECK(compiled.nodes[1].isSink);
    CHECK_FALSE(compiled.nodes[1].isSource);
    CHECK(compiled.nodes[0].firstOutputEdge == 0u);
    CHECK(compiled.nodes[0].outputEdgeCount == 1u);
    CHECK(compiled.nodes[1].firstInputEdge == 0u);
    CHECK(compiled.nodes[1].inputEdgeCount == 1u);
    CHECK(compiled.outputEdges[0] == 0u);
    CHECK(compiled.inputEdges[0] == 0u);

    CHECK(compiled.sources[0] == source);
    CHECK(compiled.sinks[0] == sink);

    CHECK(compiled.streams[0].id == stream);
    CHECK(compiled.streams[0].chunkCapacity == 8u);
    CHECK(compiled.streams[0].producerCount == 1u);
    CHECK(compiled.streams[0].consumerCount == 1u);
    CHECK(compiled.streamProducers[0].node == source);
    CHECK(compiled.streamConsumers[0].node == sink);
}

TEST_CASE("graph_model rejects a mismatched element type naming both ports", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 4u, 1u}));
    auto const writer = model.addNode(makeNode("Writer", 1u, {}, {output(floatSchema(4u), stream)}));
    auto const reader = model.addNode(makeNode("Reader", 2u, {input(doubleSchema(4u))}, {}));
    REQUIRE(model.addEdge(makeEdge(writer, 0u, reader, 0u, stream)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);

    auto const* error = findError(*errors, GraphErrorCode::IncompatibleSchema);
    REQUIRE(error != nullptr);
    CHECK(error->writerPort == "Writer.output[0]");
    CHECK(error->readerPort == "Reader.input[0]");
    CHECK(error->message.find("Writer.output[0]") != std::string::npos);
    CHECK(error->message.find("Reader.input[0]") != std::string::npos);
}

TEST_CASE("graph_model rejects a rank mismatch", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    auto const rankTwo = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 2u>{2u, 2u});

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 4u, 1u}));
    auto const writer = model.addNode(makeNode("Writer", 1u, {}, {output(floatSchema(4u), stream)}));
    auto const reader = model.addNode(makeNode("Reader", 2u, {input(rankTwo)}, {}));
    REQUIRE(model.addEdge(makeEdge(writer, 0u, reader, 0u, stream)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);
    CHECK(findError(*errors, GraphErrorCode::IncompatibleSchema) != nullptr);
}

TEST_CASE("graph_model rejects an extent mismatch", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 8u, 1u}));
    auto const writer = model.addNode(makeNode("Writer", 1u, {}, {output(floatSchema(4u), stream)}));
    auto const reader = model.addNode(makeNode("Reader", 2u, {input(floatSchema(8u))}, {}));
    REQUIRE(model.addEdge(makeEdge(writer, 0u, reader, 0u, stream)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);
    CHECK(findError(*errors, GraphErrorCode::IncompatibleSchema) != nullptr);
}

TEST_CASE("graph_model rejects a pitch that is not an element-size multiple", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    auto writerSchema = floatSchema(4u);
    writerSchema.pitchBytes[0] = 5u; // not a multiple of sizeof(float)

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 4u, 1u}));
    auto const writer = model.addNode(makeNode("Writer", 1u, {}, {output(writerSchema, stream)}));
    auto const reader = model.addNode(makeNode("Reader", 2u, {input(floatSchema(4u))}, {}));
    REQUIRE(model.addEdge(makeEdge(writer, 0u, reader, 0u, stream)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);
    CHECK(findError(*errors, GraphErrorCode::IncompatibleSchema) != nullptr);
}

TEST_CASE("graph_model rejects an undeclared cycle", "[graph][graph_model]")
{
    constexpr StreamId sourceToA{1u};
    constexpr StreamId aToB{2u};
    constexpr StreamId bToSink{3u};
    constexpr StreamId bToA{4u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{sourceToA, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{aToB, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{bToSink, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{bToA, 0u, 4u, 1u}));

    auto const source = model.addNode(makeNode("Source", 1u, {}, {output(floatSchema(4u), sourceToA)}));
    auto const a = model.addNode(
        makeNode("A", 2u, {input(floatSchema(4u)), input(floatSchema(4u))}, {output(floatSchema(4u), aToB)}));
    auto const b = model.addNode(makeNode(
        "B",
        3u,
        {input(floatSchema(4u))},
        {output(floatSchema(4u), bToSink), output(floatSchema(4u), bToA)}));
    auto const sink = model.addNode(makeNode("Sink", 4u, {input(floatSchema(4u))}, {}));

    REQUIRE(model.addEdge(makeEdge(source, 0u, a, 0u, sourceToA)));
    REQUIRE(model.addEdge(makeEdge(a, 0u, b, 0u, aToB)));
    REQUIRE(model.addEdge(makeEdge(b, 0u, sink, 0u, bToSink)));
    REQUIRE(model.addEdge(makeEdge(b, 1u, a, 1u, bToA)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);

    auto const* cycle = findError(*errors, GraphErrorCode::UndeclaredCycle);
    REQUIRE(cycle != nullptr);
    CHECK(cycle->message.find("cycle") != std::string::npos);
}

TEST_CASE("graph_model accepts a declared feedback cycle", "[graph][graph_model]")
{
    constexpr StreamId sourceToA{1u};
    constexpr StreamId aToB{2u};
    constexpr StreamId bToSink{3u};
    constexpr StreamId bToA{4u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{sourceToA, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{aToB, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{bToSink, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{bToA, 0u, 4u, 1u}));

    auto const source = model.addNode(makeNode("Source", 1u, {}, {output(floatSchema(4u), sourceToA)}));
    auto const a = model.addNode(
        makeNode("A", 2u, {input(floatSchema(4u)), input(floatSchema(4u))}, {output(floatSchema(4u), aToB)}));
    auto const b = model.addNode(makeNode(
        "B",
        3u,
        {input(floatSchema(4u))},
        {output(floatSchema(4u), bToSink), output(floatSchema(4u), bToA)}));
    auto const sink = model.addNode(makeNode("Sink", 4u, {input(floatSchema(4u))}, {}));

    REQUIRE(model.addEdge(makeEdge(source, 0u, a, 0u, sourceToA)));
    REQUIRE(model.addEdge(makeEdge(a, 0u, b, 0u, aToB)));
    REQUIRE(model.addEdge(makeEdge(b, 0u, sink, 0u, bToSink)));
    REQUIRE(model.addEdge(makeFeedbackEdge(b, 1u, a, 1u, bToA)));

    auto outcome = model.buildGraph();
    REQUIRE(graph::succeeded(outcome));

    auto const& compiled = std::get<graph::CompiledGraph>(outcome);
    CHECK(compiled.edgeCount == 4u);
    CHECK(compiled.streams[3].id == bToA);

    std::uint32_t feedbackEdges = 0u;
    for(std::uint32_t edge = 0u; edge < compiled.edgeCount; ++edge)
    {
        if(compiled.edges[edge].feedback)
            ++feedbackEdges;
    }
    CHECK(feedbackEdges == 1u);
}

TEST_CASE("graph_model rejects a self edge without the feedback flag", "[graph][graph_model]")
{
    constexpr StreamId sourceToA{1u};
    constexpr StreamId aToSink{2u};
    constexpr StreamId self{3u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{sourceToA, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{aToSink, 0u, 4u, 1u}));
    REQUIRE(model.addStream(StreamDescriptor{self, 0u, 4u, 1u}));

    auto const source = model.addNode(makeNode("Source", 1u, {}, {output(floatSchema(4u), sourceToA)}));
    auto const a = model.addNode(makeNode(
        "A",
        2u,
        {input(floatSchema(4u)), input(floatSchema(4u))},
        {output(floatSchema(4u), aToSink), output(floatSchema(4u), self)}));
    auto const sink = model.addNode(makeNode("Sink", 3u, {input(floatSchema(4u))}, {}));

    REQUIRE(model.addEdge(makeEdge(source, 0u, a, 0u, sourceToA)));
    REQUIRE(model.addEdge(makeEdge(a, 0u, sink, 0u, aToSink)));
    REQUIRE(model.addEdge(makeEdge(a, 1u, a, 1u, self)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);
    CHECK(findError(*errors, GraphErrorCode::SelfEdgeWithoutFeedback) != nullptr);
}

TEST_CASE("graph_model rejects a duplicate edge", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    graph::GraphModel model;
    REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 4u, 1u}));
    auto const source = model.addNode(makeNode("Source", 1u, {}, {output(floatSchema(4u), stream)}));
    auto const sink = model.addNode(makeNode("Sink", 2u, {input(floatSchema(4u))}, {}));
    REQUIRE(model.addEdge(makeEdge(source, 0u, sink, 0u, stream)));
    REQUIRE(model.addEdge(makeEdge(source, 0u, sink, 0u, stream)));

    auto outcome = model.buildGraph();
    REQUIRE_FALSE(graph::succeeded(outcome));
    auto const* errors = errorsOf(outcome);
    REQUIRE(errors != nullptr);
    CHECK(findError(*errors, GraphErrorCode::DuplicateEdge) != nullptr);
}

TEST_CASE("graph_model rejects a missing source or sink and names the node", "[graph][graph_model]")
{
    constexpr StreamId stream{1u};

    SECTION("sink without source")
    {
        graph::GraphModel model;
        REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 4u, 1u}));
        model.addNode(makeNode("SinkOnly", 1u, {input(floatSchema(4u))}, {}));

        auto outcome = model.buildGraph();
        REQUIRE_FALSE(graph::succeeded(outcome));
        auto const* errors = errorsOf(outcome);
        REQUIRE(errors != nullptr);
        auto const* error = findError(*errors, GraphErrorCode::MissingSource);
        REQUIRE(error != nullptr);
        CHECK(error->message.find("SinkOnly") != std::string::npos);
    }

    SECTION("source without sink")
    {
        graph::GraphModel model;
        REQUIRE(model.addStream(StreamDescriptor{stream, 0u, 4u, 1u}));
        model.addNode(makeNode("SourceOnly", 1u, {}, {output(floatSchema(4u), stream)}));

        auto outcome = model.buildGraph();
        REQUIRE_FALSE(graph::succeeded(outcome));
        auto const* errors = errorsOf(outcome);
        REQUIRE(errors != nullptr);
        auto const* error = findError(*errors, GraphErrorCode::MissingSink);
        REQUIRE(error != nullptr);
        CHECK(error->message.find("SourceOnly") != std::string::npos);
    }
}

TEST_CASE("graph_model particle feedback graph validates for supported chunk sizes", "[graph][graph_model]")
{
    for(auto const chunkSize : {1u, 2u, 128u, 100u})
    {
        INFO("chunkSize=" << chunkSize);

        auto model = graph::makeParticleFeedbackGraph(chunkSize);
        auto outcome = model.buildGraph();
        REQUIRE(graph::succeeded(outcome));

        auto const& compiled = std::get<graph::CompiledGraph>(outcome);
        CHECK(compiled.nodeCount == 5u);
        CHECK(compiled.edgeCount == 7u);
        CHECK(compiled.streamCount == 4u);
        CHECK(compiled.sourceCount == 1u);
        CHECK(compiled.sinkCount == 1u);
        CHECK(compiled.nodes[0].isSource);
        CHECK(compiled.nodes[4].isSink);

        std::uint32_t feedbackEdges = 0u;
        for(std::uint32_t edge = 0u; edge < compiled.edgeCount; ++edge)
            if(compiled.edges[edge].feedback)
                ++feedbackEdges;
        CHECK(feedbackEdges == 2u);

        // The shared filter input has three producers and one consumer.
        CHECK(compiled.streams[0].producerCount == 3u);
        CHECK(compiled.streams[0].consumerCount == 1u);

        for(std::uint32_t node = 0u; node < compiled.nodeCount; ++node)
        {
            // One input port may be fed by several producers (the shared filter
            // input), so only its port count is fixed. Every output port emits
            // through exactly one edge in this graph.
            CHECK(compiled.nodes[node].inputPortCount == model.nodes()[node].inputs.size());
            CHECK(compiled.nodes[node].outputPortCount == model.nodes()[node].outputs.size());
            CHECK(compiled.nodes[node].inputEdgeCount >= compiled.nodes[node].inputPortCount);
            CHECK(compiled.nodes[node].outputEdgeCount == model.nodes()[node].outputs.size());
        }
    }
}

TEST_CASE("graph_model particle feedback graph carries the chunk size into stream capacities", "[graph][graph_model]")
{
    for(auto const chunkSize : {1u, 2u, 128u, 100u})
    {
        INFO("chunkSize=" << chunkSize);

        auto model = graph::makeParticleFeedbackGraph(chunkSize);
        auto outcome = model.buildGraph();
        REQUIRE(graph::succeeded(outcome));

        auto const& compiled = std::get<graph::CompiledGraph>(outcome);
        for(std::uint32_t stream = 0u; stream < compiled.streamCount; ++stream)
        {
            CHECK(compiled.streams[stream].chunkCapacity == chunkSize);
            CHECK(compiled.streams[stream].capacity() >= chunkSize);
        }

        // The two bin streams and the shared filter input are double buffered;
        // the terminal sink stream is consumed directly.
        CHECK(compiled.streams[0].chunkCount == 2u);
        CHECK(compiled.streams[1].chunkCount == 2u);
        CHECK(compiled.streams[2].chunkCount == 2u);
        CHECK(compiled.streams[3].chunkCount == 1u);

        for(auto const& node : model.nodes())
        {
            for(auto const& port : node.inputs)
                CHECK(port.schema.extents[0] == chunkSize);
            for(auto const& port : node.outputs)
                CHECK(port.schema.extents[0] == chunkSize);
        }
    }
}
