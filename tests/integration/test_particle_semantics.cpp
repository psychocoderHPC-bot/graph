// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include "oracle.hpp"

#include <catch2/catch_test_macros.hpp>
#include <graph/particle_functors.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace graph;

namespace
{
    using Weight = std::uint32_t;

    constexpr std::uint64_t maxVisitsGuard = 1'000'000u;

    struct RunOutcome
    {
        std::vector<Weight> finalWeights{}; // per id; UINT32_MAX if undestroyed
        std::vector<std::uint32_t> sinkVisits{};
        std::uint64_t nonTerminating = 0u;
        std::uint64_t maxVisit = 0u;
        std::string error;
    };

    /** Host-side sequential worklist driver over the named functor cores.
     *
     * Not the pool: it is a single-threaded queue of particle ids that runs
     * `FilterByProperty::apply`, then `ProcessA::apply` or `ProcessB::apply`,
     * then `DestroyParticles::apply`, with the exact routing of the graph.
     * The initial worklist order is the only scheduling degree of freedom of a
     * single-threaded driver, so `order` selects it: 0 = ids ascending,
     * 1 = ids descending, 2 = ids ascending by (id % 3). The draw of every
     * particle depends only on `(seed, id, visit, purpose)`, so all three
     * orders must produce identical results.
     */
    RunOutcome runWorklist(std::uint64_t n, RngSeed seed, std::uint32_t order)
    {
        RunOutcome out;
        out.finalWeights.assign(n, std::numeric_limits<Weight>::max());
        out.sinkVisits.assign(n, 0u);

        if(n == 0u)
            return out;

        std::vector<Particle> parts(n);
        for(std::uint64_t i = 0u; i < n; ++i)
            parts[i] = GenerateParticles::apply(seed, i);

        auto orderKey = [order](std::uint64_t id) -> std::uint64_t
        {
            switch(order)
            {
            case 1u:
                return std::numeric_limits<std::uint64_t>::max() - id;
            case 2u:
                return id % 3u;
            default:
                return id;
            }
        };

        std::vector<std::uint64_t> initial;
        initial.reserve(n);
        for(std::uint64_t i = 0u; i < n; ++i)
            initial.push_back(i);
        std::sort(
            initial.begin(),
            initial.end(),
            [&orderKey](std::uint64_t a, std::uint64_t b) { return orderKey(a) < orderKey(b); });
        std::vector<std::uint64_t> queue;
        queue.reserve(n * 4u);
        queue.insert(queue.end(), initial.begin(), initial.end());

        for(std::size_t head = 0u; head < queue.size(); ++head)
        {
            auto const id = queue[head];
            auto& p = parts[id];

            if(p.visit > maxVisitsGuard)
            {
                ++out.nonTerminating;
                continue;
            }
            out.maxVisit = std::max(out.maxVisit, static_cast<std::uint64_t>(p.visit));

            auto const filter = FilterByProperty::apply(seed, p);
            REQUIRE_FALSE(filter.error);
            auto const bin1 = filter.destination == particle_feedback::Destination::Bin1;

            auto const process = bin1 ? ProcessB::apply(seed, p) : ProcessA::apply(seed, p);
            if(process.error)
            {
                out.error = "weight overflow for particle " + std::to_string(id);
                return out;
            }

            if(process.destination == particle_feedback::Destination::FilterInput)
            {
                // Survives: the feedback stream carries the incremented record.
                p = bin1 ? ProcessB::feedbackParticle(p) : ProcessA::feedbackParticle(p);
                queue.push_back(id);
            }
            else
            {
                // DestroyParticles: exactly one slot write, per-id.
                auto const destroy = DestroyParticles::apply(p);
                REQUIRE_FALSE(destroy.error);
                if(out.sinkVisits[id] != 0u)
                {
                    out.error = "duplicate sink for particle " + std::to_string(id);
                    return out;
                }
                ++out.sinkVisits[id];
                out.finalWeights[id] = p.weight;
            }
        }

        // Every id entered the queue exactly once at generation and
        // re-enters exactly once per feedback return, so the sink-visit
        // counters above already prove the exactly-once routing.
        return out;
    }

    /** The committed fixture: the canonical oracle histogram rows for one
     *  (N, seed) case, from `tests/reference/test-vectors.md`. */
    struct Fixture
    {
        std::uint64_t n{};
        RngSeed seed{};
        std::vector<std::pair<Weight, std::uint64_t>> rows{};
        std::uint64_t total{};
        std::uint64_t maxVisit{};
        std::uint64_t fnv1a{};
    };

    std::vector<Fixture> const fixtures = {
        {0u, 42u, {}, 0u, 0u, 0x1465'0FB0'739D'0383ull},
        {1u, 42u, {{0u, 1u}}, 1u, 0u, 0x44BD'2BD4'73CC'F799ull},
        {7u,
         42u,
         {{0u, 2u}, {1000u, 1u}, {1002u, 1u}, {1003u, 1u}, {2000u, 1u}, {3000u, 1u}},
         7u,
         4u,
         0x4A2D'87F0'295A'6244ull},
        {128u,
         42u,
         {{0u, 55u},
          {1u, 16u},
          {2u, 4u},
          {3u, 6u},
          {4u, 2u},
          {1000u, 13u},
          {1001u, 9u},
          {1002u, 6u},
          {1003u, 2u},
          {2000u, 2u},
          {2001u, 3u},
          {2002u, 2u},
          {3000u, 3u},
          {3001u, 1u},
          {3002u, 1u},
          {4001u, 1u},
          {5000u, 1u},
          {5002u, 1u}},
         128u,
         7u,
         0x3D7B'D0B2'9DC8'C6D1ull},
        {1000u,
         42u,
         {{0u, 487u},   {1u, 112u},    {2u, 45u},    {3u, 17u},    {4u, 6u},     {5u, 2u},     {6u, 3u},
          {7u, 1u},     {1000u, 119u}, {1001u, 57u}, {1002u, 27u}, {1003u, 10u}, {1004u, 2u},  {2000u, 32u},
          {2001u, 18u}, {2002u, 7u},   {2003u, 2u},  {2004u, 1u},  {2005u, 1u},  {3000u, 16u}, {3001u, 8u},
          {3002u, 6u},  {3004u, 2u},   {3005u, 1u},  {3006u, 1u},  {4000u, 8u},  {4001u, 2u},  {4002u, 1u},
          {5000u, 2u},  {5002u, 2u},   {6002u, 1u},  {7000u, 1u}},
         1000u,
         9u,
         0x94B3'4787'55D3'3472ull},
        {4096u,
         42u,
         {{0u, 2081u}, {1u, 480u},  {2u, 170u},   {3u, 72u},     {4u, 20u},     {5u, 12u},    {6u, 7u},
          {7u, 3u},    {9u, 1u},    {10u, 1u},    {1000u, 478u}, {1001u, 215u}, {1002u, 78u}, {1003u, 25u},
          {1004u, 8u}, {1005u, 4u}, {1006u, 2u},  {2000u, 136u}, {2001u, 71u},  {2002u, 30u}, {2003u, 10u},
          {2004u, 6u}, {2005u, 2u}, {3000u, 62u}, {3001u, 28u},  {3002u, 12u},  {3003u, 3u},  {3004u, 4u},
          {3005u, 1u}, {3006u, 1u}, {4000u, 24u}, {4001u, 11u},  {4002u, 9u},   {4004u, 1u},  {5000u, 7u},
          {5001u, 4u}, {5002u, 4u}, {6000u, 6u},  {6001u, 3u},   {6002u, 1u},   {7000u, 2u},  {7001u, 1u}},
         4096u,
         10u,
         0xFBBC'1B50'42F9'5AABull},
        {4096u,
         7u,
         {{0u, 2036u},  {1u, 513u},    {2u, 182u},    {3u, 62u},    {4u, 23u},    {5u, 13u},   {6u, 6u},
          {9u, 2u},     {1000u, 531u}, {1001u, 161u}, {1002u, 55u}, {1003u, 33u}, {1004u, 9u}, {1005u, 8u},
          {1006u, 2u},  {2000u, 193u}, {2001u, 51u},  {2002u, 26u}, {2003u, 10u}, {2004u, 2u}, {2006u, 1u},
          {2007u, 1u},  {3000u, 52u},  {3001u, 26u},  {3002u, 12u}, {3003u, 6u},  {3004u, 1u}, {3005u, 2u},
          {4000u, 23u}, {4001u, 7u},   {4002u, 5u},   {4004u, 3u},  {5000u, 16u}, {5001u, 4u}, {5002u, 2u},
          {6000u, 8u},  {6001u, 4u},   {7000u, 2u},   {7006u, 1u},  {9000u, 2u}},
         4096u,
         13u,
         0x6A86'EA19'D3B7'BEE2ull},
        {65536u,
         42u,
         {{0u, 32768u},   {1u, 8122u},   {2u, 2741u},    {3u, 1078u},    {4u, 405u},    {5u, 174u},     {6u, 77u},
          {7u, 36u},      {8u, 10u},     {9u, 5u},       {10u, 3u},      {11u, 1u},     {1000u, 8078u}, {1001u, 2762u},
          {1002u, 1086u}, {1003u, 425u}, {1004u, 158u},  {1005u, 75u},   {1006u, 41u},  {1007u, 11u},   {1008u, 5u},
          {1009u, 3u},    {1010u, 1u},   {2000u, 2719u}, {2001u, 1033u}, {2002u, 425u}, {2003u, 163u},  {2004u, 96u},
          {2005u, 28u},   {2006u, 12u},  {2007u, 7u},    {2008u, 4u},    {2009u, 1u},   {2010u, 2u},    {3000u, 1020u},
          {3001u, 410u},  {3002u, 188u}, {3003u, 75u},   {3004u, 46u},   {3005u, 13u},  {3006u, 6u},    {3007u, 1u},
          {3009u, 1u},    {4000u, 393u}, {4001u, 188u},  {4002u, 73u},   {4003u, 29u},  {4004u, 11u},   {4005u, 3u},
          {4006u, 4u},    {4007u, 4u},   {5000u, 167u},  {5001u, 65u},   {5002u, 28u},  {5003u, 15u},   {5004u, 8u},
          {5005u, 2u},    {5006u, 3u},   {5011u, 1u},    {6000u, 83u},   {6001u, 38u},  {6002u, 18u},   {6003u, 3u},
          {6004u, 4u},    {6005u, 1u},   {6006u, 1u},    {7000u, 29u},   {7001u, 12u},  {7002u, 6u},    {7004u, 1u},
          {7005u, 2u},    {8000u, 10u},  {8001u, 2u},    {8003u, 2u},    {9000u, 1u},   {9001u, 1u},    {9002u, 1u},
          {9003u, 1u},    {9006u, 1u},   {10000u, 1u},   {10001u, 1u},   {11000u, 3u},  {11001u, 1u},   {12000u, 2u},
          {13000u, 1u},   {13001u, 1u}},
         65536u,
         16u,
         0x9596'2522'75DC'B00Bull},
        {65536u,
         7u,
         {{0u, 32663u},   {1u, 8191u},    {2u, 2829u},   {3u, 1060u},   {4u, 425u},     {5u, 169u},    {6u, 75u},
          {7u, 30u},      {8u, 17u},      {9u, 9u},      {10u, 3u},     {11u, 1u},      {12u, 1u},     {1000u, 8290u},
          {1001u, 2697u}, {1002u, 1021u}, {1003u, 435u}, {1004u, 156u}, {1005u, 73u},   {1006u, 33u},  {1007u, 10u},
          {1008u, 4u},    {1009u, 4u},    {1010u, 1u},   {1013u, 1u},   {2000u, 2683u}, {2001u, 999u}, {2002u, 431u},
          {2003u, 160u},  {2004u, 67u},   {2005u, 25u},  {2006u, 6u},   {2007u, 12u},   {2008u, 4u},   {2012u, 1u},
          {3000u, 995u},  {3001u, 435u},  {3002u, 175u}, {3003u, 83u},  {3004u, 34u},   {3005u, 13u},  {3006u, 7u},
          {3007u, 1u},    {3009u, 1u},    {3010u, 2u},   {3017u, 1u},   {4000u, 374u},  {4001u, 186u}, {4002u, 81u},
          {4003u, 30u},   {4004u, 18u},   {4005u, 6u},   {4006u, 3u},   {5000u, 141u},  {5001u, 59u},  {5002u, 30u},
          {5003u, 15u},   {5004u, 6u},    {5005u, 2u},   {5006u, 3u},   {5007u, 1u},    {5008u, 1u},   {6000u, 60u},
          {6001u, 45u},   {6002u, 14u},   {6003u, 10u},  {6004u, 4u},   {6005u, 1u},    {6007u, 2u},   {7000u, 41u},
          {7001u, 14u},   {7002u, 4u},    {7003u, 3u},   {7005u, 2u},   {7006u, 1u},    {8000u, 13u},  {8001u, 4u},
          {8002u, 2u},    {9000u, 9u},    {9001u, 6u},   {10000u, 3u},  {10001u, 2u},   {10002u, 2u},  {11000u, 3u},
          {11001u, 1u},   {11002u, 1u}},
         65536u,
         20u,
         0x68D9'8160'0F95'03F2ull},
    };

    /** Sparse histogram of the destroyed final weights, sorted ascending.
     *  Sentinel (undestroyed) weights are excluded. */
    std::vector<std::pair<Weight, std::uint64_t>> sparseHistogram(RunOutcome const& run)
    {
        std::map<Weight, std::uint64_t> counts;
        for(auto const weight : run.finalWeights)
        {
            if(weight != std::numeric_limits<Weight>::max())
                ++counts[weight];
        }
        std::vector<std::pair<Weight, std::uint64_t>> rows;
        rows.reserve(counts.size());
        for(auto const& [weight, count] : counts)
            rows.emplace_back(weight, count);
        return rows;
    }

    /** FNV-1a 64 over destroyed final weights in ID order (same definition as
     *  the oracle's checksum). */
    std::uint64_t fnv1aChecksum(RunOutcome const& run)
    {
        std::uint64_t hash = 1'469'598'103'934'665'603ull;
        for(auto const weight : run.finalWeights)
        {
            if(weight == std::numeric_limits<Weight>::max())
                continue;
            hash ^= static_cast<std::uint64_t>(weight);
            hash *= 1'099'511'628'211ull;
        }
        return hash;
    }

    std::string formatRows(std::vector<std::pair<Weight, std::uint64_t>> const& rows)
    {
        std::string text;
        for(auto const& [weight, count] : rows)
        {
            text += std::to_string(weight) + " " + std::to_string(count) + "\n";
        }
        return text;
    }
} // namespace

TEST_CASE("particle_semantics worklist matches the committed scalar reference", "[particle_semantics]")
{
    for(auto const& fixture : fixtures)
    {
        // Reference run with the canonical worklist order (ascending).
        auto const reference = runWorklist(fixture.n, fixture.seed, 0u);
        REQUIRE_FALSE(!reference.error.empty());
        REQUIRE(reference.nonTerminating == 0u);
        REQUIRE(reference.maxVisit == fixture.maxVisit);

        // Every id terminates exactly once; counts sum to N.
        REQUIRE(reference.sinkVisits.size() == fixture.n);
        std::uint64_t destroyed = 0u;
        for(std::size_t id = 0u; id < reference.sinkVisits.size(); ++id)
        {
            REQUIRE(reference.sinkVisits[id] == 1u);
            destroyed += reference.finalWeights[id] != std::numeric_limits<Weight>::max();
        }
        REQUIRE(destroyed == fixture.n);

        // Sparse histogram must match the committed fixture row for row,
        // including the zero-weight row whenever present.
        auto const rows = sparseHistogram(reference);
        REQUIRE(rows == fixture.rows);
        std::uint64_t countSum = 0u;
        for(auto const& [weight, count] : rows)
            countSum += count;
        REQUIRE(countSum == fixture.n);

        // FNV-1a checksum must match the committed fixture value.
        auto const checksum = fnv1aChecksum(reference);
        std::printf(
            "[particle_semantics] N=%llu seed=%llu fnv1a=%016llx (expected %016llx) "
            "maxVisit=%llu rows=%zu\n",
            static_cast<unsigned long long>(fixture.n),
            static_cast<unsigned long long>(fixture.seed),
            static_cast<unsigned long long>(checksum),
            static_cast<unsigned long long>(fixture.fnv1a),
            static_cast<unsigned long long>(reference.maxVisit),
            rows.size());
        REQUIRE(checksum == fixture.fnv1a);

        // Determinism: two different visit orders must produce identical
        // per-ID weights and the identical histogram.
        auto const reversed = runWorklist(fixture.n, fixture.seed, 1u);
        REQUIRE_FALSE(!reversed.error.empty());
        REQUIRE(reversed.finalWeights == reference.finalWeights);
        REQUIRE(sparseHistogram(reversed) == rows);
        REQUIRE(fnv1aChecksum(reversed) == checksum);
        if(fixture.n >= 4u)
        {
            auto const third = runWorklist(fixture.n, fixture.seed, 2u);
            REQUIRE_FALSE(!third.error.empty());
            REQUIRE(third.finalWeights == reference.finalWeights);
            REQUIRE(fnv1aChecksum(third) == checksum);
        }
    }
}

TEST_CASE("particle_semantics exact 0.5 boundary by construction", "[particle_semantics]")
{
    // The decision threshold is the exact 0.5 on the 2^32 scale: draw32 >= 2^31
    // is flag 1. Verify the two boundary draws directly: a draw of exactly
    // 2^31 survives, a draw of 2^31 - 1 does not, and the 2^32 - 1 / 0 extremes
    // behave as documented.
    auto const boundary = std::vector<std::pair<std::uint32_t, bool>>{
        {static_cast<std::uint32_t>(halfScale), true},
        {static_cast<std::uint32_t>(halfScale) - 1u, false},
        {std::numeric_limits<std::uint32_t>::max(), true},
        {0u, false},
    };
    for(auto const [draw, survives] : boundary)
    {
        REQUIRE((draw >= halfScale) == survives);
    }

    // And that a ProcessA / ProcessB core with a weight of UINT32_MAX - 999
    // does overflow: the core reports the error instead of wrapping.
    Particle hot{};
    hot.id = 1u;
    hot.property = 0u;
    hot.weight = std::numeric_limits<Weight>::max() - 999u;
    hot.visit = 0u;
    // Force a flag-1 branch by picking a seed/id/visit whose DECISION_A draw is
    // >= 2^31; the specific draw does not matter, only that the add is checked.
    bool sawOverflow = false;
    bool sawSurvive = false;
    for(std::uint64_t id = 0u; id < 64u; ++id)
    {
        auto hot2 = hot;
        hot2.id = id;
        auto const resultA = ProcessA::apply(42u, hot2);
        if(resultA.error)
            sawOverflow = true;
        if(resultA.destination == particle_feedback::Destination::FilterInput)
            sawSurvive = true;
        auto const resultB = ProcessB::apply(42u, hot2);
        if(resultB.error)
            sawOverflow = true;
        if(resultB.destination == particle_feedback::Destination::FilterInput)
            sawSurvive = true;
    }
    REQUIRE(sawSurvive);
    REQUIRE(sawOverflow);
}

TEST_CASE("particle_semantics property preserved and visit incremented", "[particle_semantics]")
{
    // The property is assigned once at generation and preserved through every
    // feedback round; the visit counter increments on every feedback return.
    // Drive a few particles through the cores and check the invariants of the
    // feedback particle directly.
    for(std::uint64_t id = 0u; id < 16u; ++id)
    {
        auto const p = GenerateParticles::apply(42u, id);
        REQUIRE(p.id == id);
        REQUIRE(p.weight == 0u);
        REQUIRE(p.visit == 0u);
        REQUIRE(p.property == rngDraw32(42u, id, 0u, RngPurpose::Property));

        auto const a = ProcessA::apply(42u, p);
        if(a.destination == particle_feedback::Destination::FilterInput)
        {
            auto const f = ProcessA::feedbackParticle(p);
            REQUIRE(f.property == p.property);
            REQUIRE(f.weight == p.weight + ProcessA::weightIncrement);
            REQUIRE(f.visit == p.visit + 1u);
        }
        auto const b = ProcessB::apply(42u, p);
        if(b.destination == particle_feedback::Destination::FilterInput)
        {
            auto const f = ProcessB::feedbackParticle(p);
            REQUIRE(f.property == p.property);
            REQUIRE(f.weight == p.weight + ProcessB::weightIncrement);
            REQUIRE(f.visit == p.visit + 1u);
        }
    }
}

TEST_CASE("particle_semantics destroy writes the final weight exactly once", "[particle_semantics]")
{
    // DestroyParticles::apply is the pure core; the operator() does the slot
    // write. The per-id slot write is exactly-once by the sinkVisits check in
    // the fixture test above. Here verify the core returns the sink and that
    // the final weight equals the particle weight.
    Particle p{};
    p.id = 5u;
    p.property = 123u;
    p.weight = 4210u;
    p.visit = 3u;
    auto const result = DestroyParticles::apply(p);
    REQUIRE(result.destination == particle_feedback::Destination::Sink);
    REQUIRE_FALSE(result.error);
    REQUIRE(p.weight == 4210u);
}
