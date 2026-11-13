// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Unit tests for the functor-reported output counts in the operation ABI.
//
// The scheduler publishes exactly the count a functor reports per port. This
// test pins the two cheap invariants the rest of the pipeline relies on:
// a count above the port capacity is rejected, and a port that never reports a
// count (or reports zero) publishes nothing.

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/concepts.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>

#include <cstdint>
#include <type_traits>

namespace
{
    graph::PacketRecord makeRecord(std::uint32_t capacity)
    {
        graph::PacketRecord record{};
        record.capacity = capacity;
        record.schema = graph::makeElementSchema<std::uint32_t>(alpaka::Vec<std::size_t, 1u>{capacity});
        return record;
    }
} // namespace

TEST_CASE("packet accessor rejects an output count above capacity", "[graph][operation][counts]")
{
    graph::PacketAccessor accessor{};
    graph::PacketRecord record = makeRecord(4u);
    std::uint32_t storage[8]{};

    REQUIRE(accessor.setOutput(0u, &record, storage));
    // A fresh port reports zero, which is the documented publish-nothing
    // default; setOutput also resets it.
    CHECK(accessor.outputCount(0u) == 0u);
    CHECK_FALSE(accessor.setOutputCount(0u, 5u));
    CHECK(accessor.outputCount(0u) == 0u);

    CHECK(accessor.setOutputCount(0u, 4u));
    CHECK(accessor.outputCount(0u) == 4u);
    CHECK_FALSE(accessor.setOutputCount(0u, 5u));
    CHECK(accessor.outputCount(0u) == 4u);

    // Out-of-range and unbound ports are rejected, never silently accepted.
    CHECK_FALSE(accessor.setOutputCount(graph::maxPacketPorts, 0u));
    CHECK_FALSE(accessor.setOutputCount(1u, 0u));
    CHECK(accessor.outputCount(1u) == 0u);
    CHECK(accessor.outputCount(graph::maxPacketPorts) == 0u);
}

TEST_CASE("packet accessor zero count publishes nothing by default", "[graph][operation][counts]")
{
    graph::PacketAccessor accessor{};
    graph::PacketRecord record = makeRecord(8u);
    std::uint32_t storage[8]{};

    // The default is zero for every port, so a functor that writes no output
    // (or never reports) appends nothing downstream.
    for(std::uint32_t port = 0u; port < graph::maxPacketPorts; ++port)
        CHECK(accessor.outputCount(port) == 0u);

    REQUIRE(accessor.setOutput(0u, &record, storage));
    CHECK(accessor.outputCount(0u) == 0u);
    CHECK(accessor.setOutputCount(0u, 0u));
    CHECK(accessor.outputCount(0u) == 0u);

    // Binding a port again starts from a clean zero even after a nonzero
    // report, so a reused accessor cannot inherit an old count.
    CHECK(accessor.setOutputCount(0u, 3u));
    REQUIRE(accessor.setOutput(0u, &record, storage));
    CHECK(accessor.outputCount(0u) == 0u);

    accessor.clear();
    CHECK(accessor.outputCount(0u) == 0u);
}

TEST_CASE("packet accessor remains a trivially copyable standard-layout value", "[graph][operation][counts]")
{
    static_assert(std::is_trivially_copyable_v<graph::PacketAccessor>);
    static_assert(std::is_standard_layout_v<graph::PacketAccessor>);
    SUCCEED();
}
