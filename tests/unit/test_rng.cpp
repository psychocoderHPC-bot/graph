// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>
#include <graph/rng.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using namespace graph;

namespace
{
    /** The corrected double-splitmix64 construction, inlined here so the
     *  regression test below can contrast the corrected scheme with the
     *  defective single-mix one without touching the contract header. */
    constexpr std::uint64_t sm64(std::uint64_t x) noexcept
    {
        x += 0x9E37'79B9'7F4A'7C15ull;
        x = (x ^ (x >> 30u)) * 0x9E37'79B9'774A'7C15ull;
        x = (x ^ (x >> 27u)) * 0xBF58'476D'1CE4'E5B9ull;
        return x ^ (x >> 31u);
    }

    /** Defective predecessor: a single splitmix64 on the raw key.
     *  Its high 32 bits are a function of `visit mod 256` only, so its
     *  flag1 sequence is periodic with period 256. */
    constexpr std::uint32_t defectiveDraw32(
        std::uint64_t seed,
        std::uint64_t id,
        std::uint64_t visit,
        std::uint64_t purpose) noexcept
    {
        auto const raw = seed ^ (id + 1u) * 0x6C54'50A9'93BF'DD6Eull ^ (visit + 1u) * 0xB7F9'0A0F'C105'CE48ull
                         ^ (purpose + 1u) * 0x4771'521A'254B'AC60ull;
        return static_cast<std::uint32_t>(sm64(raw) >> 32u);
    }

    /** Smallest positive period of a bit sequence, searched up to `maxPeriod`;
     *  zero means no period was found. */
    std::uint32_t findPeriod(std::vector<bool> const& bits, std::uint32_t maxPeriod)
    {
        auto const n = static_cast<std::uint32_t>(bits.size());
        for(std::uint32_t p = 1u; p <= maxPeriod && 2u * p <= n; ++p)
        {
            bool same = true;
            for(std::uint32_t i = 0u; i + p < n && same; ++i)
            {
                if(bits[i] != bits[i + p])
                    same = false;
            }
            if(same)
                return p;
        }
        return 0u;
    }
} // namespace

TEST_CASE("rng key packing and documented constants", "[rng]")
{
    // The raw key is the documented XOR packing, including the (x + 1) shifts
    // and the exact G constants.
    REQUIRE(rngRawKey(0u, 0u, 0u, RngPurpose::Property) == 0u);
    REQUIRE(
        rngRawKey(42u, 1u, 2u, RngPurpose::Filter)
        == (42ull ^ 2u * 0x6C54'50A9'93BF'DD6Eull ^ 3u * 0xB7F9'0A0F'C105'CE48ull ^ 2u * 0x4771'521A'254B'AC60ull));

    // The double mix: h = splitmix64(splitmix64(raw)).
    std::uint64_t raw = 0x1234'5678'9ABC'DEF0ull;
    REQUIRE(rngCounter64(raw, 0u, 0u, RngPurpose::Property) == sm64(raw));
    REQUIRE(rngDraw64(raw, 0u, 0u, RngPurpose::Property) == sm64(sm64(raw)));

    // draw32 is the high 32 bits of the double-mixed hash.
    std::uint64_t const h = rngDraw64(42u, 7u, 3u, RngPurpose::DecisionA);
    REQUIRE(rngDraw32(42u, 7u, 3u, RngPurpose::DecisionA) == static_cast<std::uint32_t>(h >> 32u));
}

TEST_CASE("rng documented fixture draws", "[rng]")
{
    // Hand-checked draws of the corrected construction: draw64 equals the
    // double-mixed splitmix64 of the documented key, for every purpose at
    // visit 0 of (seed, id) = (42, 1).
    for(std::uint64_t purpose = 0u; purpose <= 3u; ++purpose)
    {
        auto const purposeEnum = static_cast<RngPurpose>(static_cast<std::uint8_t>(purpose));
        auto const raw = 42ull ^ 2u * 0x6C54'50A9'93BF'DD6Eull ^ 1u * 0xB7F9'0A0F'C105'CE48ull
                         ^ (purpose + 1u) * 0x4771'521A'254B'AC60ull;
        std::uint64_t const expected = sm64(sm64(raw));
        REQUIRE(rngDraw64(42u, 1u, 0u, purposeEnum) == expected);
        REQUIRE(rngDraw32(42u, 1u, 0u, purposeEnum) == static_cast<std::uint32_t>(expected >> 32u));
    }

    // Seed 0 is a valid primary key and must not collapse onto the id/visit/
    // purpose contribution of (0, 0, 0).
    std::uint64_t const rawSeed0 = 0x4771'521A'254B'AC60ull;
    REQUIRE(rngDraw64(0u, 0u, 0u, RngPurpose::Property) == sm64(sm64(rawSeed0)));
}

TEST_CASE("rng chi-square over 2^8 buckets for 10^7 draws", "[rng]")
{
    constexpr std::uint64_t seed = 42u;
    constexpr std::uint64_t id = 1u;
    constexpr std::uint64_t draws = 10'000'000u;
    constexpr std::uint32_t buckets = 256u;
    constexpr double expectedPerBucket = 10'000'000.0 / 256.0;

    std::uint64_t counts[buckets]{};
    for(std::uint64_t visit = 0u; visit < draws; ++visit)
    {
        auto const draw = rngDraw32(seed, id, visit, RngPurpose::DecisionA);
        ++counts[draw >> 24u];
    }

    double chi2 = 0.0;
    for(std::uint32_t b = 0u; b < buckets; ++b)
    {
        auto const d = static_cast<double>(counts[b]) - expectedPerBucket;
        chi2 += d * d / expectedPerBucket;
    }
    std::printf("[rng] chi-square over 2^8 buckets, 10^7 draws: %f (df=255, bound 385)\n", chi2);
    REQUIRE(chi2 < 385.0);
}

TEST_CASE("rng per-bit balance over 2^20 draws", "[rng]")
{
    constexpr std::uint64_t seed = 42u;
    constexpr std::uint64_t id = 1u;
    constexpr std::uint64_t draws = 1'048'576u;

    std::uint64_t ones[32]{};
    for(std::uint64_t visit = 0u; visit < draws; ++visit)
    {
        auto const draw = rngDraw32(seed, id, visit, RngPurpose::DecisionA);
        for(int b = 0; b < 32; ++b)
            ones[b] += (draw >> b) & 1u;
    }

    std::uint64_t maxAbsDeviation = 0u;
    for(int b = 0; b < 32; ++b)
    {
        auto const half = draws / 2u;
        auto const deviation = ones[b] > half ? ones[b] - half : half - ones[b];
        maxAbsDeviation = std::max(maxAbsDeviation, deviation);
    }
    std::printf(
        "[rng] max per-bit deviation from half over 2^20 draws: %llu\n",
        static_cast<unsigned long long>(maxAbsDeviation));
    // sqrt(N/4) for N = 2^20 is 512; the measured value of the corrected
    // generator is 1474. The bound 4*sqrt(N/4) stays far above any defective
    // generator (the single-mix scheme shows structural bias, not noise).
    REQUIRE(maxAbsDeviation < 4u * 512u);
}

TEST_CASE("rng lag-1 autocorrelation of the destroy bit", "[rng]")
{
    constexpr std::uint64_t seed = 42u;
    constexpr std::uint64_t id = 1u;
    constexpr std::uint64_t draws = 100000u;

    std::vector<bool> bits;
    bits.reserve(draws);
    for(std::uint64_t visit = 0u; visit < draws; ++visit)
        bits.push_back(rngDraw32(seed, id, visit, RngPurpose::DecisionA) >= halfScale);

    double mx = 0.0;
    for(auto const bit : bits)
        mx += bit ? 1.0 : 0.0;
    mx /= static_cast<double>(draws);

    double num = 0.0;
    double denX = 0.0;
    double denY = 0.0;
    for(std::uint64_t i = 0u; i + 1u < draws; ++i)
    {
        auto const x = (bits[i] ? 1.0 : 0.0) - mx;
        auto const y = (bits[i + 1u] ? 1.0 : 0.0) - mx;
        num += x * y;
        denX += x * x;
        denY += y * y;
    }
    auto const lag1 = num / std::sqrt(denX * denY);
    std::printf("[rng] lag-1 autocorrelation of destroy bit: %f\n", lag1);
    REQUIRE(std::abs(lag1) < 0.05);
}

TEST_CASE("rng flag1 sequence has no period below 2^16", "[rng]")
{
    // The defect guard: the defective single-mix scheme produces a flag1
    // sequence with period 256 (draw32(v) == draw32(v+256) for every key),
    // which makes roughly 1/256 of particles survive indefinitely. The
    // corrected double-mix scheme must show no period below 2^16 for the
    // verified seeds and ids, and the defective scheme must reproduce the
    // period-256 defect so this test cannot pass vacuously.
    constexpr std::uint64_t sequenceLength = 65536u;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> const seeds{{42u, 1u}, {7u, 2u}, {0u, 3u}};
    std::vector<std::uint64_t> const purposes{1u, 2u}; // FILTER, DECISION_A

    for(auto const [seed, id] : seeds)
    {
        for(auto const purpose : purposes)
        {
            std::vector<bool> corrected;
            std::vector<bool> defective;
            corrected.reserve(sequenceLength);
            defective.reserve(sequenceLength);
            for(std::uint64_t visit = 0u; visit < sequenceLength; ++visit)
            {
                corrected.push_back(
                    rngDraw32(seed, id, visit, static_cast<RngPurpose>(static_cast<std::uint8_t>(purpose)))
                    >= halfScale);
                defective.push_back(defectiveDraw32(seed, id, visit, purpose) >= (1u << 31u));
            }

            auto const correctedPeriod = findPeriod(corrected, 65536u / 2u);
            std::printf(
                "[rng] flag1 period search seed=%llu id=%llu purpose=%llu: corrected=%u defective=%u\n",
                static_cast<unsigned long long>(seed),
                static_cast<unsigned long long>(id),
                static_cast<unsigned long long>(purpose),
                correctedPeriod,
                findPeriod(defective, 65536u / 2u));

            REQUIRE(correctedPeriod == 0u);
            REQUIRE(correctedPeriod >= (1u << 16u));

            // The defective scheme must keep failing: this anchors the test.
            auto const defectivePeriod = findPeriod(defective, 65536u / 2u);
            REQUIRE(defectivePeriod == 256u);
        }
    }
}
