// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>
#include <graph/histogram.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
    auto sentinelHistogram(std::vector<std::int64_t> const& weights, std::uint64_t expectedCount)
    {
        return graph::buildHistogram(std::span<std::int64_t const>{weights}, expectedCount);
    }

    auto markerHistogram(
        std::vector<std::int64_t> const& weights,
        std::vector<std::uint32_t> const& visits,
        std::uint64_t expectedCount)
    {
        return graph::buildHistogram(
            std::span<std::int64_t const>{weights},
            std::span<std::uint32_t const>{visits},
            expectedCount);
    }

    auto contains(std::string const& text, std::string const& needle) -> bool
    {
        return text.find(needle) != std::string::npos;
    }
} // namespace

TEST_CASE("histogram empty input produces an empty result", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{};
    auto const result = sentinelHistogram(weights, 0u);

    REQUIRE(result.ok);
    CHECK(result.error.empty());
    CHECK(result.histogram.empty());
    CHECK(result.histogram.totalCount() == 0u);
    CHECK(result.histogram.render().empty());

    std::vector<std::uint32_t> const visits{};
    auto const marked = markerHistogram(weights, visits, 0u);
    REQUIRE(marked.ok);
    CHECK(marked.histogram.empty());
}

TEST_CASE("histogram single weight yields one row", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{7};
    auto const result = sentinelHistogram(weights, 1u);

    REQUIRE(result.ok);
    REQUIRE(result.histogram.rows.size() == 1u);
    CHECK((result.histogram.rows[0] == graph::HistogramRow{7, 1u}));
    CHECK(result.histogram.totalCount() == 1u);
    CHECK(result.histogram.render() == "7 1\n");
}

TEST_CASE("histogram zero-weight row is present when a zero weight occurs", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{0, 0, 3};
    auto const result = sentinelHistogram(weights, 3u);

    REQUIRE(result.ok);
    REQUIRE(result.histogram.rows.size() == 2u);
    CHECK((result.histogram.rows[0] == graph::HistogramRow{0, 2u}));
    CHECK((result.histogram.rows[1] == graph::HistogramRow{3, 1u}));
    CHECK(result.histogram.render() == "0 2\n3 1\n");
}

TEST_CASE("histogram sparse and large weights need no dense weight-dependent allocation", "[graph][histogram]")
{
    auto const largest = std::numeric_limits<std::int64_t>::max();
    std::vector<std::int64_t> const weights{1'000'000, 7, 1'000'000, largest};
    auto const result = sentinelHistogram(weights, 4u);

    REQUIRE(result.ok);
    REQUIRE(result.histogram.rows.size() == 3u);
    CHECK((result.histogram.rows[0] == graph::HistogramRow{7, 1u}));
    CHECK((result.histogram.rows[1] == graph::HistogramRow{1'000'000, 2u}));
    CHECK((result.histogram.rows[2] == graph::HistogramRow{largest, 1u}));
    CHECK(result.histogram.totalCount() == 4u);
}

TEST_CASE("histogram duplicate weights are merged and unsorted input renders deterministically", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{9, 1, 5, 1, 1001, 5, 9};
    auto const result = sentinelHistogram(weights, 7u);

    REQUIRE(result.ok);
    REQUIRE(result.histogram.rows.size() == 4u);
    CHECK((result.histogram.rows[0] == graph::HistogramRow{1, 2u}));
    CHECK((result.histogram.rows[1] == graph::HistogramRow{5, 2u}));
    CHECK((result.histogram.rows[2] == graph::HistogramRow{9, 2u}));
    CHECK((result.histogram.rows[3] == graph::HistogramRow{1001, 1u}));
    CHECK(result.histogram.render() == "1 2\n5 2\n9 2\n1001 1\n");

    // Same multiset in a different order must render byte-for-byte identically.
    std::vector<std::int64_t> const reordered{1001, 9, 9, 5, 1, 5, 1};
    auto const again = sentinelHistogram(reordered, 7u);
    REQUIRE(again.ok);
    CHECK(again.histogram.render() == result.histogram.render());
}

TEST_CASE("histogram counts must sum to N", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{1, 2, 3};
    auto const result = sentinelHistogram(weights, 2u);

    CHECK_FALSE(result.ok);
    CHECK(result.histogram.empty());
    CHECK(contains(result.error, "expected 2 final-weight slots"));
    CHECK(contains(result.error, "got 3"));

    std::vector<std::uint32_t> const visits{1, 1, 1};
    auto const marked = markerHistogram(weights, visits, 4u);
    CHECK_FALSE(marked.ok);
    CHECK(contains(marked.error, "expected 4 final-weight slots"));
}

TEST_CASE("histogram a duplicate sink write is detected", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{10, 11, 12};
    std::vector<std::uint32_t> const visits{1, 2, 1};
    auto const result = markerHistogram(weights, visits, 3u);

    CHECK_FALSE(result.ok);
    CHECK(result.histogram.empty());
    CHECK(contains(result.error, "source id 1"));
    CHECK(contains(result.error, "terminated 2 times"));
}

TEST_CASE("histogram a missing sink termination is detected", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{10, 11, 12};
    std::vector<std::uint32_t> const visits{1, 0, 1};
    auto const result = markerHistogram(weights, visits, 3u);

    CHECK_FALSE(result.ok);
    CHECK(contains(result.error, "source id 1"));
    CHECK(contains(result.error, "terminated 0 times"));
}

TEST_CASE("histogram an unwritten sentinel slot is reported", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{4, graph::unwrittenWeight};
    auto const result = sentinelHistogram(weights, 2u);

    CHECK_FALSE(result.ok);
    CHECK(contains(result.error, "source id 1"));
    CHECK(contains(result.error, "no final weight"));
}

TEST_CASE("histogram negative final weights are rejected", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{4, -1};
    auto const result = sentinelHistogram(weights, 2u);

    CHECK_FALSE(result.ok);
    CHECK(contains(result.error, "source id 1"));
    CHECK(contains(result.error, "negative final weight -1"));

    std::vector<std::uint32_t> const visits{1, 1};
    auto const marked = markerHistogram(weights, visits, 2u);
    CHECK_FALSE(marked.ok);
    CHECK(contains(marked.error, "negative final weight"));
}

TEST_CASE("histogram sink-visit counters must match the source count", "[graph][histogram]")
{
    std::vector<std::int64_t> const weights{1, 2};
    std::vector<std::uint32_t> const visits{1};
    auto const result = markerHistogram(weights, visits, 2u);

    CHECK_FALSE(result.ok);
    CHECK(contains(result.error, "expected 2 sink-visit counters"));
    CHECK(contains(result.error, "got 1"));
}
