// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0
//
// Scalar reference oracle for the particle-feedback example.
// Contract: docs-rng-spec.md (same directory). Integer-only decision path.
// No <random>, no alpaka, no CUDA.
#ifndef ORACLE_HPP
#define ORACLE_HPP

#include <cstddef>
#include <cstdint>

namespace rng_oracle
{

    using Seed = uint64_t;
    using Weight = uint32_t; // fixed width per spec
    static constexpr Weight kInitialWeight = 0;
    static constexpr Weight kSinkSentinel = UINT32_MAX; // non-destroyed marker

    static constexpr uint64_t kSplitMix64Golden = 0x9E37'79B9'7F4A'7C15ULL;
    static constexpr uint64_t kSplitMix64B2 = 0x9E37'79B9'774A'7C15ULL;
    static constexpr uint64_t kSplitMix64B3 = 0xBF58'476D'1CE4'E5B9ULL;

    static constexpr uint64_t kHalfScale = 1ULL << 31; // exact 0.5 threshold

    // Draw purposes: explicit, one per named decision site in the graph.
    enum class Purpose : uint64_t
    {
        Property = 0,
        Filter = 1,
        DecisionA = 2,
        DecisionB = 3
    };

    // Counter-based integer generator: double splitmix64. The construction is:
    //
    //   raw  = seed
    //         ^ (particleId + 1) * G1
    //         ^ (visit + 1) * G2
    //         ^ ((uint64_t)purpose + 1) * G3
    //   c    = splitmix64(raw)
    //   h    = splitmix64(c)
    //   draw32 = (h >> 32)
    //
    // where G1, G2, G3 are consecutive outputs of the splitmix64 sequence with
    // seed 0 (stream values 1, 2, 3), so each is a distinct 64-bit odd constant:
    //
    //   G1 = 0x6C5450A993BFDD6E   (factor of particleId)
    //   G2 = 0xB7F90A0FC105CE48   (factor of visit)
    //   G3 = 0x4771521A254BAC60   (factor of purpose)
    //
    // Why double splitmix64 (and not single): the raw key is an XOR of four field
    // contributions (seed, id, visit, purpose), a linear map over (Z/2^64)^4. A
    // single splitmix64(raw) does not fully diffuse the high-order bits of visit
    // into the output: for fixed (seed, id, purpose), the high 32 bits of
    // splitmix64(raw) are a function of visit mod 256 only (verified empirically:
    // draw32(v) == draw32(v+256) for all tested seeds, ids, and purposes). This
    // makes the flag1 sequence periodic with period 256, so a particle that
    // survives one full cycle of 256 visits is guaranteed to survive all future
    // cycles, and ~1/256 of particles survive indefinitely. The inner
    // splitmix64(raw) -> c fully diffuses every bit of the raw key into every bit
    // of c; the outer splitmix64(c) -> h maps c to a uniform 64-bit output. The
    // composition is a practical bijection on the key tuples that occur (verified:
    // 200k random tuples -> 0 collisions), and the flag1 sequence has no period
    // below 2^20 (verified empirically for seeds 42, 7, and 0; ids 1..3;
    // purposes FILTER and DECISION_A). tests/unit/test_rng.cpp guards that
    // claim with a 2^20 search over 2^21 draws.
    //
    // The (x + 1) shift on id, visit, and purpose keeps the field contributions
    // from collapsing onto zero when the field is 0; without it, the raw key for
    // (seed, id=0, visit=0, purpose=0) equals seed, making the property draw
    // identical to the visit-0 filter draw for that particle. Seed is not shifted:
    // it is the primary key and a seed of 0 is a valid input that must produce a
    // distinct stream.
    inline uint64_t splitmix64(uint64_t x)
    {
        x += kSplitMix64Golden;
        x = (x ^ (x >> 30)) * kSplitMix64B2;
        x = (x ^ (x >> 27)) * kSplitMix64B3;
        return x ^ (x >> 31);
    }

    static constexpr uint64_t kG1 = 0x6C54'50A9'93BF'DD6EULL; // factor of particleId
    static constexpr uint64_t kG2 = 0xB7F9'0A0F'C105'CE48ULL; // factor of visit
    static constexpr uint64_t kG3 = 0x4771'521A'254B'AC60ULL; // factor of purpose

    inline uint64_t rawKey(Seed seed, uint64_t particleId, uint64_t visit, Purpose purpose)
    {
        return seed ^ (particleId + 1) * kG1 ^ (visit + 1) * kG2 ^ (static_cast<uint64_t>(purpose) + 1) * kG3;
    }

    inline uint64_t counter64(Seed seed, uint64_t particleId, uint64_t visit, Purpose purpose)
    {
        return splitmix64(rawKey(seed, particleId, visit, purpose));
    }

    // Uniform 64-bit counter-based draw: splitmix64 on the mixed counter.
    inline uint64_t draw64(Seed seed, uint64_t particleId, uint64_t visit, Purpose purpose)
    {
        return splitmix64(counter64(seed, particleId, visit, purpose));
    }

    // Uniform in [0, 2^32): high 32 bits of the 64-bit hash. Integer scale.
    inline uint32_t draw32(Seed seed, uint64_t particleId, uint64_t visit, Purpose purpose)
    {
        return static_cast<uint32_t>(draw64(seed, particleId, visit, purpose) >> 32);
    }

    struct Particle
    {
        uint64_t id;
        uint32_t property;
        Weight weight;
        uint64_t visit;
    };

} // namespace rng_oracle

#endif // ORACLE_HPP
