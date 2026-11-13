// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Both module headers in one translation unit. They used to define
// `graph::StreamId` independently, which made this include pair a hard compile
// error. Keeping the pair here is the regression guard for that defect.
#include <catch2/catch_test_macros.hpp>
#include <graph/chunk_assembly.hpp>
#include <graph/graph_model.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
    /** Stream id type as seen through each module's own data structure.
     *
     * `remove_cvref_t` is deliberate: nvcc/EDG does not canonicalize the
     * `decltype` result of a member access for `std::is_same`, so a direct
     * comparison with `graph::StreamId` fails in a CUDA pass even though the
     * types are identical. Stripping cv/ref first canonicalizes the type and
     * keeps the identity check portable across host and CUDA compilation.
     */
    using ModelStreamId = std::remove_cvref_t<decltype(std::declval<graph::StreamDescriptor const&>().id)>;
    using AssemblyStreamId
        = std::remove_cvref_t<decltype(std::declval<graph::AssemblyStream<std::uint32_t> const&>().id)>;

    // A stream id taken from each module's own data structure must be the same
    // type; a reintroduced second definition or a renamed spelling breaks here.
    static_assert(std::is_same_v<ModelStreamId, AssemblyStreamId>);
    static_assert(std::is_same_v<ModelStreamId, graph::StreamId>);
    static_assert(std::is_same_v<AssemblyStreamId, graph::StreamId>);
} // namespace

TEST_CASE("StreamId is one type across graph_model and chunk_assembly", "[stream_id][unity]")
{
    constexpr graph::StreamId modelId{7u};
    graph::StreamDescriptor const descriptor{modelId, 0u, 128u, 1u};

    graph::AssemblyStream<std::uint32_t> states[1]{};
    // A value built on the graph_model side is stored in assembly state and
    // read back through the assembly API.
    states[0].id = descriptor.id;
    states[0].capacity = 4u;
    states[0].count = 0u;

    graph::ChunkArena<std::uint32_t> const arena{};
    graph::StreamAssembler<std::uint32_t> const assembler{states, 1u, arena};

    CHECK(assembler.find(modelId) == 0u);
    CHECK(assembler.stream(0).id == descriptor.id);
    CHECK(assembler.stream(0).id == modelId);
    CHECK(assembler.stream(0).id.value == 7u);
    CHECK(assembler.stream(0).id.isValid());

    constexpr graph::StreamId invalid{};
    CHECK_FALSE(invalid.isValid());
    CHECK(invalid == graph::StreamId::invalid());
    CHECK(invalid.value == graph::StreamId::invalidValue);
}
