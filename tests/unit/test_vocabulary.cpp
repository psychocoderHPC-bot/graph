// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/concepts.hpp>
#include <graph/detail/job.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstdint>
#include <type_traits>

namespace
{
    /** Stand-in accelerator for host-only registry tests. */
    struct HostTestAcc
    {
        static constexpr std::uint32_t getWarpSize()
        {
            return 1u;
        }
    };

    void hostOperation(HostTestAcc const&, graph::JobDescriptor const&, graph::PacketAccessor const&)
    {
    }

    /** Instantiates the complete vocabulary for the concrete accelerator type.
     *
     * The kernel also runs the width-one warp paths on a host executor. It is
     * launched on the CUDA backend as well when that backend is enabled, which
     * instantiates every type for the GPU accelerator type.
     */
    struct VocabularyKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::Vec<std::uint32_t, 1u> extent) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Fn = graph::OperationFn<Acc>;

            static_assert(std::is_pointer_v<Fn>);
            static_assert(std::is_function_v<std::remove_pointer_t<Fn>>);
            static_assert(std::is_trivially_copyable_v<graph::JobDescriptor>);
            static_assert(std::is_trivially_copyable_v<graph::PacketAccessor>);
            static_assert(std::is_trivially_copyable_v<graph::OperationTable<Acc>>);
            static_assert(std::is_trivially_copyable_v<graph::ElementSchema>);

            graph::WarpGroup<Acc> warp{acc};

            // Instantiate the descriptors and the accessor on the accelerator.
            graph::PacketAccessor accessor{};
            accessor.clear();
            graph::OperationTable<Acc> table{};
            auto const unresolved = table.resolve(graph::JobDescriptor{}.operation);
            graph::PacketRecord record{};
            auto const logical = record.logicalCount();
            graph::RequestSpec request{};
            graph::JobDescriptor job{};
            graph::JobKind const kind = job.kind;
            (void) unresolved;
            (void) logical;
            (void) request;
            (void) kind;

            auto const width = warp.workerWidth();
            auto const lane = warp.laneIndex();
            auto const broadcastValue = warp.template broadcast<std::uint64_t>(0x1234'5678'90ab'cdefull);
            auto const mask = warp.ballot(true);
            auto const bits = graph::WarpGroup<Acc>::popcount(mask);
            auto const position = warp.rank(mask);

            auto const stride = extent.x();
            for(auto [index] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{extent}))
            {
                out[index] = width;
                out[stride + index] = broadcastValue == 0x1234'5678'90ab'cdefull ? 1u : 0u;
                out[2u * stride + index] = bits;
                out[3u * stride + index] = position;
                out[4u * stride + index] = lane;
            }
        }
    };

    constexpr std::uint32_t maxThreads = 32u;
    constexpr std::uint32_t valuesPerThread = 5u;

    template<typename T_Device>
    void runVocabularyKernel(T_Device& device, std::uint32_t numThreads, auto executor)
    {
        auto queue = device.makeQueue();

        auto hostBuffer = alpaka::onHost::allocHost<std::uint32_t>(maxThreads * valuesPerThread);
        for(std::uint32_t i = 0u; i < maxThreads * valuesPerThread; ++i)
            hostBuffer[i] = 0u;

        auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);
        auto const threadSpec = alpaka::onHost::ThreadSpec{1u, numThreads, executor};
        queue.enqueue(threadSpec, VocabularyKernel{}, deviceBuffer, alpaka::Vec<std::uint32_t, 1u>{numThreads});
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);

        auto const stride = numThreads;
        for(std::uint32_t lane = 0u; lane < numThreads; ++lane)
        {
            auto const width = hostBuffer[lane];
            CHECK(width >= 1u);
            CHECK(hostBuffer[stride + lane] == 1u);
            CHECK(hostBuffer[2u * stride + lane] == width);
            CHECK(hostBuffer[3u * stride + lane] == hostBuffer[4u * stride + lane]);
            CHECK(hostBuffer[4u * stride + lane] < width);
        }
    }
} // namespace

TEST_CASE("graph vocabulary is trivially copyable and ABI stable", "[graph][vocabulary]")
{
    static_assert(std::is_trivially_copyable_v<graph::ElementSchema>);
    static_assert(std::is_trivially_copyable_v<graph::PacketHandle>);
    static_assert(std::is_trivially_copyable_v<graph::PacketRecord>);
    static_assert(std::is_trivially_copyable_v<graph::RequestSpec>);
    static_assert(std::is_trivially_copyable_v<graph::RoutingInfo>);
    static_assert(std::is_trivially_copyable_v<graph::JobDescriptor>);
    static_assert(std::is_trivially_copyable_v<graph::OperationId>);

    using Fn = graph::OperationFn<HostTestAcc>;
    static_assert(std::is_pointer_v<Fn>);
    static_assert(std::is_function_v<std::remove_pointer_t<Fn>>);

    graph::JobDescriptor job{};
    job.kind = graph::JobKind::END_OF_INPUT;
    CHECK(graph::isControlJob(job.kind));
    job.kind = graph::JobKind::OPERATION;
    CHECK_FALSE(graph::isControlJob(job.kind));
}

TEST_CASE("graph element schemas describe and check edges", "[graph][vocabulary]")
{
    auto const schema = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 1u>{4u});
    CHECK(schema.rank == 1u);
    CHECK(schema.elementSize == sizeof(float));
    CHECK(schema.alignment == alignof(float));
    CHECK(schema.extents[0] == 4u);
    CHECK(schema.pitchBytes[0] == sizeof(float));
    CHECK(graph::isValid(schema));
    CHECK(graph::schemasCompatible(schema, schema));

    auto const smaller = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 1u>{2u});
    CHECK(graph::schemasEdgeCompatible(schema, smaller));
    CHECK_FALSE(graph::schemasEdgeCompatible(smaller, schema));
    CHECK_FALSE(graph::schemasCompatible(schema, smaller));

    auto const otherType = graph::makeElementSchema<double>(alpaka::Vec<std::size_t, 1u>{4u});
    CHECK_FALSE(graph::schemasEdgeCompatible(schema, otherType));

    auto const matrix = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 2u>{2u, 3u});
    CHECK(matrix.rank == 2u);
    CHECK(matrix.pitchBytes[1] == sizeof(float));
    CHECK(matrix.pitchBytes[0] == 3u * sizeof(float));
    CHECK(graph::isValid(matrix));
}

TEST_CASE("graph packet views and masked spans stay in bounds", "[graph][vocabulary]")
{
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[2] = {-1.0f, -1.0f};

    auto const schema = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 1u>{4u});

    graph::PacketRecord inputRecord{};
    inputRecord.handle = graph::PacketHandle{3u, 7u};
    inputRecord.elementCount = 3u;
    inputRecord.capacity = 4u;
    inputRecord.schema = schema;

    graph::PacketRecord outputRecord{};
    outputRecord.elementCount = 1u;
    outputRecord.capacity = 2u;
    outputRecord.schema = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 1u>{2u});

    graph::PacketAccessor accessor{};
    CHECK(accessor.setInput(0u, &inputRecord, input));
    CHECK(accessor.setOutput(0u, &outputRecord, output));
    CHECK_FALSE(accessor.setInput(graph::maxPacketPorts, &inputRecord, input));

    auto const inputView = accessor.inputView<float>();
    CHECK(inputView.getExtents()[0] == 4u);

    auto const inputSpan = accessor.inputSpan<float>();
    REQUIRE(inputSpan.size() == 3u);
    CHECK(inputSpan.read(0u) == 1.0f);
    CHECK(inputSpan.read(2u) == 3.0f);
    CHECK(inputSpan.read(3u) == 0.0f);

    auto outputSpan = accessor.outputSpan<float>();
    REQUIRE(outputSpan.size() == 1u);
    outputSpan.write(0u, 42.0f);
    outputSpan.write(1u, 99.0f);
    CHECK(output[0] == 42.0f);
    CHECK(output[1] == -1.0f);

    auto const nested = graph::makeCheckedPacketSpan<float>(inputRecord, input, 1u, 99u);
    CHECK(nested.size() == 2u);
    CHECK(nested.read(0u) == 2.0f);
    CHECK(nested.subSpan(1u, 99u).size() == 1u);

    auto const absent = accessor.inputSpan<float>(1u);
    CHECK(absent.size() == 0u);

    auto const direct = graph::makePacketView<float>(inputRecord, input);
    CHECK(direct.getExtents()[0] == 4u);
    CHECK(direct[0] == 1.0f);
}

TEST_CASE("graph operation table validates and resolves hosts registrations", "[graph][vocabulary]")
{
    auto const schema = graph::makeElementSchema<float>(alpaka::Vec<std::size_t, 1u>{4u});
    graph::OperationTable<HostTestAcc> table{};

    auto const id = graph::OperationId{7u};
    CHECK(table.registerOperation(id, schema, schema, &hostOperation) == graph::RegistrationResult::Ok);
    CHECK(table.resolve(id) == &hostOperation);
    CHECK(table.find(id) != nullptr);
    CHECK(table.find(graph::OperationId{8u}) == nullptr);
    CHECK(table.registerOperation(id, schema, schema, &hostOperation) == graph::RegistrationResult::DuplicateId);
    CHECK(
        table.registerOperation(graph::OperationId{}, schema, schema, &hostOperation)
        == graph::RegistrationResult::InvalidId);
    CHECK(
        table.registerOperation(graph::OperationId{8u}, schema, schema, nullptr)
        == graph::RegistrationResult::NullFunction);

    auto const empty = graph::ElementSchema{};
    CHECK(graph::isValid(empty));
    CHECK(
        table.registerOperation(graph::OperationId{9u}, empty, schema, &hostOperation)
        == graph::RegistrationResult::Ok);
}

TEST_CASE("graph warp group implements the width-one fallback on the host", "[graph][vocabulary]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);

    auto device = selector.makeDevice(0);
    runVocabularyKernel(device, 1u, alpaka::exec::cpuSerial);
}

#if GRAPH_HAS_CUDA
TEST_CASE("graph vocabulary instantiates for the CUDA accelerator", "[graph][vocabulary][cuda]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);

    auto device = selector.makeDevice(0);
    runVocabularyKernel(device, maxThreads, alpaka::exec::gpuCuda);
}
#endif
