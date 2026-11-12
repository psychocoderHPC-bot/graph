// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

/** Host-side exact final-weight histogram.
 *
 * This is the final-result processing step of the particle-feedback example
 * (design section 8, "Histogram"). It is host-only: after the worker kernel
 * exits, the per-source final weights are copied into a pinned host buffer
 * (see `graph/detail/copy_queue.hpp`) and passed here as a flat span.
 *
 * # Data layout
 *
 * There is exactly one final-weight slot per source id. Only the sink
 * (`DestroyParticles`) writes a slot; the slot is written exactly once when the
 * particle reaches the sink. Two validation variants are accepted:
 *
 * - `buildHistogram(weights, expectedNumberOfParticles)` treats
 *   `unwrittenWeight` as "this source id never terminated". It can detect a
 *   missing result but cannot distinguish a duplicate sink write from a normal
 *   one.
 * - `buildHistogram(weights, sinkVisits, expectedNumberOfParticles)` additionally
 *   takes one visit counter per source id. A duplicate sink write increments the
 *   counter twice and is reported instead of silently hiding a missing result.
 *   Use this overload in validation builds and tests.
 *
 * # Output
 *
 * The result is a sorted sparse histogram: one `{weight, count}` row per
 * distinct weight in ascending order, including a weight-zero row when present.
 * It never allocates a dense array indexed by weight, so large or sparse weights
 * cost nothing beyond the number of particles.
 *
 * `Histogram::render()` produces one `weight count` line per row in ascending
 * order. Two runs (CPU and GPU, different chunk sizes, different schedules)
 * that observe the same multiset of weights therefore render byte-for-byte the
 * same text.
 *
 * # Validation
 *
 * A successful build guarantees:
 * - the number of weight slots and visit counters equals `N`,
 * - every source id terminated exactly once,
 * - every weight is non-negative (accumulated weights never become negative),
 * - no count or total-count arithmetic overflowed,
 * - `sum(row.count) == N`.
 */
namespace graph
{
    /** Sentinel stored in a final-weight slot that the sink never wrote.
     *
     * `std::numeric_limits<std::int64_t>::lowest()` is outside the valid weight
     * range (weights are non-negative), so it cannot collide with a real result.
     */
    inline constexpr std::int64_t unwrittenWeight = std::numeric_limits<std::int64_t>::lowest();

    /** One sparse histogram row. */
    struct HistogramRow
    {
        std::int64_t weight{};
        std::uint64_t count{};

        friend constexpr bool operator==(HistogramRow const&, HistogramRow const&) = default;
    };

    /** Sorted sparse histogram of exact final weights. */
    struct Histogram
    {
        /** Rows sorted by ascending `weight`, one row per distinct weight. */
        std::vector<HistogramRow> rows{};

        bool empty() const noexcept
        {
            return rows.empty();
        }

        /** Sum of all row counts. */
        std::uint64_t totalCount() const noexcept
        {
            std::uint64_t total = 0u;
            for(auto const& row : rows)
                total += row.count;
            return total;
        }

        /** Deterministic text rendering: one `weight count` line per row. */
        std::string render() const
        {
            std::string text;
            for(auto const& row : rows)
            {
                text += std::to_string(row.weight);
                text += ' ';
                text += std::to_string(row.count);
                text += '\n';
            }
            return text;
        }

        friend constexpr bool operator==(Histogram const&, Histogram const&) = default;
    };

    /** Outcome of a histogram build.
     *
     * On failure `error` explains which source id or invariant failed and
     * `histogram` is empty. On success `ok` is true and `error` is empty.
     */
    struct HistogramBuildResult
    {
        bool ok{false};
        std::string error{};
        Histogram histogram{};

        explicit operator bool() const noexcept
        {
            return ok;
        }
    };

    namespace detail
    {
        /** Count, sort, and run-length encode validated weights. */
        inline HistogramBuildResult buildFromValidatedWeights(
            std::span<std::int64_t const> weights,
            std::uint64_t expectedNumberOfParticles)
        {
            HistogramBuildResult result{};
            result.ok = true;

            std::vector<std::int64_t> sorted{weights.begin(), weights.end()};
            std::sort(sorted.begin(), sorted.end());

            for(auto const weight : sorted)
            {
                if(!result.histogram.rows.empty() && result.histogram.rows.back().weight == weight)
                {
                    auto& count = result.histogram.rows.back().count;
                    if(count == std::numeric_limits<std::uint64_t>::max())
                    {
                        result.ok = false;
                        result.error = "histogram: count overflow at weight " + std::to_string(weight);
                        result.histogram = Histogram{};
                        return result;
                    }
                    ++count;
                }
                else
                {
                    result.histogram.rows.push_back(HistogramRow{weight, 1u});
                }
            }

            if(result.histogram.totalCount() != expectedNumberOfParticles)
            {
                result.ok = false;
                result.error = "histogram: counts sum to " + std::to_string(result.histogram.totalCount())
                               + " but N is " + std::to_string(expectedNumberOfParticles);
                result.histogram = Histogram{};
                return result;
            }

            return result;
        }
    } // namespace detail

    /** Build a histogram from sentinel-marked final-weight slots.
     *
     * @param weights one slot per source id; an unwritten slot equals
     *                `unwrittenWeight`.
     * @param expectedNumberOfParticles the source count `N` (`0` is valid).
     *
     * A slot that still holds `unwrittenWeight` names a source id that never
     * reached the sink and fails the build.
     */
    inline HistogramBuildResult buildHistogram(
        std::span<std::int64_t const> weights,
        std::uint64_t expectedNumberOfParticles)
    {
        if(weights.size() != expectedNumberOfParticles)
        {
            return HistogramBuildResult{
                false,
                "histogram: expected " + std::to_string(expectedNumberOfParticles) + " final-weight slots, got "
                    + std::to_string(weights.size()),
                {}};
        }

        for(std::size_t id = 0u; id < weights.size(); ++id)
        {
            auto const weight = weights[id];
            if(weight == unwrittenWeight)
            {
                return HistogramBuildResult{
                    false,
                    "histogram: source id " + std::to_string(id) + " has no final weight (sink did not terminate)",
                    {}};
            }
            if(weight < 0)
            {
                return HistogramBuildResult{
                    false,
                    "histogram: source id " + std::to_string(id) + " has negative final weight "
                        + std::to_string(weight),
                    {}};
            }
        }

        return detail::buildFromValidatedWeights(weights, expectedNumberOfParticles);
    }

    /** Build a histogram from final weights plus per-source sink-visit counters.
     *
     * @param weights     one final-weight slot per source id.
     * @param sinkVisits  one counter per source id; the sink increments it on
     *                    every termination. Exactly one is required.
     * @param expectedNumberOfParticles the source count `N` (`0` is valid).
     *
     * This overload detects a duplicate sink write (counter greater than one)
     * and a missing termination (counter zero), so a duplicate cannot hide a
     * missing result.
     */
    inline HistogramBuildResult buildHistogram(
        std::span<std::int64_t const> weights,
        std::span<std::uint32_t const> sinkVisits,
        std::uint64_t expectedNumberOfParticles)
    {
        if(weights.size() != expectedNumberOfParticles)
        {
            return HistogramBuildResult{
                false,
                "histogram: expected " + std::to_string(expectedNumberOfParticles) + " final-weight slots, got "
                    + std::to_string(weights.size()),
                {}};
        }
        if(sinkVisits.size() != expectedNumberOfParticles)
        {
            return HistogramBuildResult{
                false,
                "histogram: expected " + std::to_string(expectedNumberOfParticles) + " sink-visit counters, got "
                    + std::to_string(sinkVisits.size()),
                {}};
        }

        for(std::size_t id = 0u; id < weights.size(); ++id)
        {
            auto const visits = sinkVisits[id];
            if(visits != 1u)
            {
                return HistogramBuildResult{
                    false,
                    "histogram: source id " + std::to_string(id) + " was terminated " + std::to_string(visits)
                        + " times; every source id must terminate exactly once",
                    {}};
            }

            auto const weight = weights[id];
            if(weight == unwrittenWeight)
            {
                return HistogramBuildResult{
                    false,
                    "histogram: source id " + std::to_string(id) + " has no final weight (sink did not terminate)",
                    {}};
            }
            if(weight < 0)
            {
                return HistogramBuildResult{
                    false,
                    "histogram: source id " + std::to_string(id) + " has negative final weight "
                        + std::to_string(weight),
                    {}};
            }
        }

        return detail::buildFromValidatedWeights(weights, expectedNumberOfParticles);
    }
} // namespace graph
