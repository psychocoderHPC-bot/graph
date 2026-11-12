// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// The scripted fixture lives in the sibling integration directory; the unit
// test target adds that directory to its include path in tests/CMakeLists.txt.
#include "toy_ops.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>
#include <vector>

namespace
{
    using graph::toy::ToyItem;
    using graph::toy::ToyResult;

    std::vector<ToyItem> makeItems(std::uint32_t count)
    {
        std::vector<ToyItem> items(count);
        for(std::uint32_t i = 0u; i < count; ++i)
            items[i] = graph::toy::toyMakeItem(i, 17u);
        return items;
    }

    /** Total number of items a source request needs when each invocation may
     *  emit at most `limit` items. */
    std::uint64_t expectedInvocations(std::uint64_t count, std::uint32_t limit)
    {
        if(limit == 0u)
            return 0u;
        return (count + limit - 1u) / limit;
    }

    /** Compile-only instantiation of every operation for the active backend.
     *
     * The kernel is never enqueued at runtime. Referencing it instantiates the
     * operation entry points for the concrete accelerator type, including the
     * CUDA device function pointers. A real device execution belongs to the
     * worker-pool milestone.
     */
    struct ToyInstantiationKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            static_assert(std::is_same_v<graph::OperationFn<Acc>, decltype(&graph::toy::ToySource<Acc>::operation)>);
            static_assert(
                std::is_same_v<graph::OperationFn<Acc>, decltype(&graph::toy::ToyPassThrough<Acc>::operation)>);
            static_assert(std::is_same_v<graph::OperationFn<Acc>, decltype(&graph::toy::ToySplit<Acc>::operation)>);
            static_assert(std::is_same_v<graph::OperationFn<Acc>, decltype(&graph::toy::ToyRevisit<Acc>::operation)>);
            static_assert(std::is_same_v<graph::OperationFn<Acc>, decltype(&graph::toy::ToySink<Acc>::operation)>);
            static_assert(std::is_same_v<graph::OperationFn<Acc>, decltype(&graph::toy::ToyAbort<Acc>::operation)>);

            graph::OperationFn<Acc> functions[]{
                &graph::toy::ToySource<Acc>::operation,
                &graph::toy::ToyPassThrough<Acc>::operation,
                &graph::toy::ToySplit<Acc>::operation,
                &graph::toy::ToyRevisit<Acc>::operation,
                &graph::toy::ToySink<Acc>::operation,
                &graph::toy::ToyAbort<Acc>::operation};
            for(auto function : functions)
            {
                if(function == nullptr)
                    continue;
            }
        }
    };
} // namespace

TEST_CASE("toy_ops item is a stable 16-byte POD", "[graph][toy]")
{
    static_assert(sizeof(ToyItem) == 16u);
    static_assert(alignof(ToyItem) == 8u);
    static_assert(std::is_trivially_copyable_v<ToyItem>);
    static_assert(std::is_standard_layout_v<ToyItem>);

    ToyItem const item{5u, 7u, 9u};
    REQUIRE(item.id == 5u);
    REQUIRE(item.value == 7u);
    REQUIRE(item.visit == 9u);
}

TEST_CASE("toy_ops source emits exactly count and splits the remainder", "[graph][toy]")
{
    for(std::uint32_t const chunkSize : {1u, 2u, 128u, 100u})
    {
        std::uint64_t const count = 1000u;
        std::uint64_t const firstId = 11u;
        std::uint64_t const seed = 42u;

        graph::RequestSpec request{firstId, count, seed};
        std::vector<ToyItem> buffer(chunkSize);
        std::uint64_t emittedTotal = 0u;
        std::uint64_t calls = 0u;

        while(request.count != 0u)
        {
            auto const plan = graph::toy::toySourceEmit(request, buffer.data(), chunkSize, chunkSize, 0u);
            REQUIRE(plan.emitted <= chunkSize);
            REQUIRE(plan.emitted > 0u);
            REQUIRE(plan.errorFlags == graph::toy::toyErrorNone);

            for(std::uint32_t i = 0u; i < plan.emitted; ++i)
            {
                std::uint64_t const id = request.firstId + i;
                REQUIRE(buffer[i].id == id);
                REQUIRE(buffer[i].value == graph::toy::toyValue(id, seed));
                REQUIRE(buffer[i].visit == 0u);
            }

            emittedTotal += plan.emitted;
            request = plan.continuation;
            ++calls;
            REQUIRE(calls <= expectedInvocations(count, chunkSize));
        }

        REQUIRE(emittedTotal == count);
        REQUIRE(calls == expectedInvocations(count, chunkSize));
        REQUIRE(request.count == 0u);
    }
}

TEST_CASE("toy_ops source continuation is a resumable id range", "[graph][toy]")
{
    graph::RequestSpec const request{100u, 10u, 3u};
    std::vector<ToyItem> buffer(4u);
    auto const plan = graph::toy::toySourceEmit(request, buffer.data(), 4u, 4u, 0u);

    REQUIRE(plan.emitted == 4u);
    REQUIRE(plan.continuation.firstId == 104u);
    REQUIRE(plan.continuation.count == 6u);
    REQUIRE(plan.continuation.seed == 3u);

    graph::RequestSpec const empty{7u, 0u, 9u};
    auto const none = graph::toy::toySourceEmit(empty, buffer.data(), 4u, 4u, 0u);
    REQUIRE(none.emitted == 0u);
    REQUIRE(none.continuation.count == 0u);
    REQUIRE(none.continuation.firstId == 7u);
}

TEST_CASE("toy_ops source reports request overflow and out-of-range ids", "[graph][toy]")
{
    graph::RequestSpec const overflow{std::numeric_limits<std::uint64_t>::max() - 1u, 8u, 0u};
    std::vector<ToyItem> buffer(4u);
    auto const plan = graph::toy::toySourceEmit(overflow, buffer.data(), 4u, 4u, 0u);
    REQUIRE(plan.emitted == 0u);
    REQUIRE((plan.errorFlags & graph::toy::toyErrorRequestOverflow) != 0u);

    graph::RequestSpec const request{0u, 8u, 0u};
    // Size the buffer from the capacity the emit call declares. The emit path
    // promises never to write beyond that declared capacity.
    constexpr std::uint32_t declaredCapacity = 8u;
    std::vector<ToyItem> sizedFromCapacity(declaredCapacity);
    auto const limited
        = graph::toy::toySourceEmit(request, sizedFromCapacity.data(), declaredCapacity, declaredCapacity, 4u);
    REQUIRE(limited.emitted == declaredCapacity);
    REQUIRE((limited.errorFlags & graph::toy::toyErrorIdOutOfRange) != 0u);
    REQUIRE((limited.errorFlags & graph::toy::toyErrorOutputOverflow) == 0u);

    graph::toy::ToyInstrumentation control{};
    graph::toy::toyMergeErrorWord(control, limited.errorFlags);
    REQUIRE((control.errorWord & graph::toy::toyErrorIdOutOfRange) != 0u);
}

TEST_CASE("toy_ops source never writes past a declared capacity", "[graph][toy]")
{
    // The request needs a full chunk of 8 items, but the caller declares room
    // for only 4. The emit path must clamp to the declared capacity, report the
    // overflow error bit, and leave the canary region untouched.
    constexpr std::uint32_t declaredCapacity = 4u;
    constexpr std::uint32_t bufferSize = 8u;
    constexpr std::uint64_t canary = 0xC0'FFEEu;

    std::vector<ToyItem> buffer(bufferSize);
    for(std::uint32_t i = 0u; i < bufferSize; ++i)
        buffer[i] = ToyItem{canary, static_cast<std::uint32_t>(canary), static_cast<std::uint32_t>(canary)};

    graph::RequestSpec const request{0u, 8u, 0u};
    auto const plan = graph::toy::toySourceEmit(request, buffer.data(), declaredCapacity, 8u, 0u);

    REQUIRE(plan.emitted == declaredCapacity);
    REQUIRE((plan.errorFlags & graph::toy::toyErrorOutputOverflow) != 0u);
    REQUIRE(plan.continuation.firstId == declaredCapacity);
    REQUIRE(plan.continuation.count == 4u);

    for(std::uint32_t i = 0u; i < declaredCapacity; ++i)
    {
        REQUIRE(buffer[i].id == i);
        REQUIRE(buffer[i].value == graph::toy::toyValue(i, 0u));
    }
    for(std::uint32_t i = declaredCapacity; i < bufferSize; ++i)
    {
        REQUIRE(buffer[i].id == canary);
        REQUIRE(buffer[i].value == static_cast<std::uint32_t>(canary));
        REQUIRE(buffer[i].visit == static_cast<std::uint32_t>(canary));
    }
}

TEST_CASE("toy_ops pass-through copies unchanged to the routed bin", "[graph][toy]")
{
    auto const input = makeItems(8u);
    std::vector<ToyItem> output(8u);
    auto const outcome = graph::toy::toyPassThroughCopy(input.data(), 8u, output.data(), 8u, 0u);

    REQUIRE(outcome.copied == 8u);
    REQUIRE(outcome.errorFlags == graph::toy::toyErrorNone);
    for(std::uint32_t i = 0u; i < 8u; ++i)
    {
        REQUIRE(output[i].id == input[i].id);
        REQUIRE(output[i].value == input[i].value);
        REQUIRE(output[i].visit == input[i].visit);
    }

    graph::RoutingInfo low{};
    low.outputPort = 0u;
    graph::RoutingInfo high{};
    high.outputPort = 1u;
    REQUIRE(graph::toy::toyPassThroughPort(low) == 0u);
    REQUIRE(graph::toy::toyPassThroughPort(high) == 1u);
    REQUIRE(graph::toy::toyPassThroughPort(low) != graph::toy::toyPassThroughPort(high));
}

TEST_CASE("toy_ops split routes by id parity with per-destination ranks", "[graph][toy]")
{
    auto const input = makeItems(10u);
    std::vector<ToyItem> bin0(10u);
    std::vector<ToyItem> bin1(10u);
    auto const outcome = graph::toy::toySplitCopy(input.data(), 10u, bin0.data(), 10u, bin1.data(), 10u, 0u);

    REQUIRE(outcome.bin0Count == 5u);
    REQUIRE(outcome.bin1Count == 5u);
    REQUIRE(outcome.errorFlags == graph::toy::toyErrorNone);

    for(std::uint32_t i = 0u; i < 5u; ++i)
    {
        REQUIRE(bin0[i].id == 2u * i);
        REQUIRE(bin1[i].id == 2u * i + 1u);
    }
    for(std::uint32_t index = 0u; index < 10u; ++index)
    {
        REQUIRE(graph::toy::toySplitRank(input.data(), 10u, index) == index / 2u);
        REQUIRE(graph::toy::toySplitBin(input[index]) == (index % 2u));
    }

    std::vector<ToyItem> small0(3u);
    std::vector<ToyItem> small1(3u);
    auto const pressure = graph::toy::toySplitCopy(input.data(), 10u, small0.data(), 3u, small1.data(), 3u, 0u);
    REQUIRE((pressure.errorFlags & graph::toy::toyErrorOutputOverflow) != 0u);
}

TEST_CASE("toy_ops revisit increments visit exactly revisitCount times then sinks", "[graph][toy]")
{
    for(std::uint32_t const revisitCount : {0u, 1u, 3u, 5u})
    {
        ToyItem item = graph::toy::toyMakeItem(12u, 99u);
        std::uint32_t feedbackRounds = 0u;
        std::uint32_t sinkVisits = 0u;

        for(std::uint32_t step = 0u; step < 32u; ++step)
        {
            auto const outcome = graph::toy::toyRevisitPlan(item, revisitCount, 0u);
            REQUIRE(outcome.errorFlags == graph::toy::toyErrorNone);
            if(!outcome.toFeedback)
            {
                ++sinkVisits;
                item = outcome.item;
                break;
            }
            ++feedbackRounds;
            item = outcome.item;
        }

        REQUIRE(feedbackRounds == revisitCount);
        REQUIRE(sinkVisits == 1u);
        REQUIRE(item.visit == revisitCount);
        REQUIRE(item.id == 12u);
        REQUIRE(item.value == graph::toy::toyValue(12u, 99u));
    }

    auto const outOfRange = graph::toy::toyRevisitPlan(graph::toy::toyMakeItem(12u, 0u), 0u, 5u);
    REQUIRE((outOfRange.errorFlags & graph::toy::toyErrorIdOutOfRange) != 0u);
}

TEST_CASE("toy_ops sink writes each id once and detects duplicates", "[graph][toy]")
{
    std::vector<ToyResult> results(8u);
    std::vector<std::uint32_t> writes(8u, 0u);

    for(std::uint32_t id = 0u; id < 8u; ++id)
    {
        // Leave id 5 out to exercise the missing-visit detector first.
        if(id == 5u)
            continue;
        auto const outcome
            = graph::toy::toySinkStore(ToyItem{id, id * 3u, id % 3u}, results.data(), 8u, writes.data(), 8u);
        REQUIRE(outcome.stored);
        REQUIRE(outcome.errorFlags == graph::toy::toyErrorNone);
        REQUIRE(outcome.previousWrites == 0u);
        REQUIRE(writes[id] == 1u);
        REQUIRE(results[id].sinkVisits == 1u);
        REQUIRE(results[id].value == id * 3u);
        REQUIRE(results[id].visits == id % 3u);
    }

    // A missing visit is visible as a zero slot and a zero write counter.
    REQUIRE(results[5].sinkVisits == 0u);
    REQUIRE(writes[5] == 0u);
    auto const late = graph::toy::toySinkStore(ToyItem{5u, 15u, 2u}, results.data(), 8u, writes.data(), 8u);
    REQUIRE(late.stored);
    REQUIRE(late.previousWrites == 0u);
    REQUIRE(results[5].sinkVisits == 1u);

    auto const duplicate = graph::toy::toySinkStore(ToyItem{3u, 9u, 1u}, results.data(), 8u, writes.data(), 8u);
    REQUIRE(duplicate.stored);
    REQUIRE(duplicate.previousWrites == 1u);
    REQUIRE((duplicate.errorFlags & graph::toy::toyErrorDuplicateSink) != 0u);
    REQUIRE(results[3].sinkVisits == 2u);

    auto const outOfRange = graph::toy::toySinkStore(ToyItem{100u, 0u, 0u}, results.data(), 8u, writes.data(), 8u);
    REQUIRE_FALSE(outOfRange.stored);
    REQUIRE((outOfRange.errorFlags & graph::toy::toyErrorIdOutOfRange) != 0u);
}

TEST_CASE("toy_ops abort sets the diagnostic abort state", "[graph][toy]")
{
    graph::toy::ToyInstrumentation control{};
    REQUIRE(control.abortRequested == 0u);
    graph::toy::toyRequestAbort(control);
    REQUIRE(control.abortRequested == 1u);
    REQUIRE((control.errorWord & graph::toy::toyErrorAborted) != 0u);

    // The abort state is separate from drain: it is never set by normal ops.
    graph::toy::ToyInstrumentation clean{};
    REQUIRE(clean.abortRequested == 0u);
    REQUIRE((clean.errorWord & graph::toy::toyErrorAborted) == 0u);
}

TEST_CASE("toy_ops graph validates with and without the feedback edge", "[graph][toy]")
{
    for(std::uint32_t const chunkSize : {1u, 2u, 128u, 100u})
    {
        graph::toy::ToyGraphConfig config{};
        config.itemCount = 1000u;
        config.chunkSize = chunkSize;
        config.revisitCount = 3u;
        config.feedback = true;

        auto const model = graph::toy::makeToyGraph(config);
        auto const outcome = model.buildGraph();
        REQUIRE(graph::succeeded(outcome));
        auto const& compiled = std::get<graph::CompiledGraph>(outcome);

        REQUIRE(compiled.nodeCount == 4u);
        REQUIRE(compiled.streamCount == 5u);
        REQUIRE(compiled.edgeCount == 5u);
        REQUIRE(compiled.sourceCount == 1u);
        REQUIRE(compiled.sinkCount == 1u);

        std::uint32_t feedbackEdges = 0u;
        for(std::uint32_t edge = 0u; edge < compiled.edgeCount; ++edge)
        {
            if(compiled.edges[edge].feedback)
                ++feedbackEdges;
        }
        REQUIRE(feedbackEdges == 1u);

        // Splits have two input ports (source input plus feedback) and the
        // revisit node has two output ports (sink plus feedback).
        REQUIRE(compiled.nodes[1u].inputPortCount == 2u);
        REQUIRE(compiled.nodes[2u].outputPortCount == 2u);
    }

    graph::toy::ToyGraphConfig acyclic{};
    acyclic.itemCount = 250u;
    acyclic.chunkSize = 32u;
    acyclic.revisitCount = 0u;
    acyclic.feedback = false;

    auto const model = graph::toy::makeToyGraph(acyclic);
    auto const outcome = model.buildGraph();
    REQUIRE(graph::succeeded(outcome));
    auto const& compiled = std::get<graph::CompiledGraph>(outcome);

    REQUIRE(compiled.nodeCount == 4u);
    REQUIRE(compiled.streamCount == 4u);
    REQUIRE(compiled.edgeCount == 4u);
    REQUIRE(compiled.sourceCount == 1u);
    REQUIRE(compiled.sinkCount == 1u);
    REQUIRE(compiled.nodes[1u].inputPortCount == 1u);
    REQUIRE(compiled.nodes[2u].outputPortCount == 1u);

    for(std::uint32_t edge = 0u; edge < compiled.edgeCount; ++edge)
        REQUIRE_FALSE(compiled.edges[edge].feedback);

    graph::toy::ToyGraphConfig invalid{};
    invalid.chunkSize = 0u;
    REQUIRE_FALSE(graph::succeeded(graph::toy::makeToyGraph(invalid).buildGraph()));
}

TEST_CASE("toy_ops operations instantiate for the active accelerator", "[graph][toy]")
{
#if GRAPH_HAS_CUDA
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto const launchCfg = alpaka::onHost::ThreadSpec{1u, 32u, alpaka::exec::gpuCuda};
#else
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto const launchCfg = alpaka::onHost::FrameSpec{1u, 1u, alpaka::exec::cpuSerial};
#endif
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    // Compile-only: this branch never runs, but the call instantiates the
    // kernel and the operation entry points for the real accelerator type.
    bool const neverLaunch = device.getName() == "graph-toy-compile-only-never";
    if(neverLaunch)
        queue.enqueue(launchCfg, ToyInstantiationKernel{});
    REQUIRE_FALSE(neverLaunch);
}
