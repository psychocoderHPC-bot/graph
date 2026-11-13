// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

/** Backend-independent integer random contract of the particle-feedback example
 *
 * Every draw is a pure function of the key tuple `(seed, particleId, visit,
 * purpose)`, so results are independent of chunk size, backend, worker, and
 * scheduling order. The construction is the corrected double-splitmix64
 * scheme from the RNG contract (see `tests/reference/` for the scalar
 * reference implementation and the canonical test vectors):
 *
 *     raw  = seed
 *          ^ (particleId + 1) * G1
 *          ^ (visit + 1) * G2
 *          ^ ((uint64_t)purpose + 1) * G3
 *     c    = splitmix64(raw)
 *     h    = splitmix64(c)
 *     draw32 = (h >> 32)
 *
 * # Key packing
 *
 * `G1`, `G2`, and `G3` are consecutive outputs of the splitmix64 sequence with
 * seed 0 (stream values 1, 2, 3), so each is a distinct 64-bit odd constant.
 * The `(x + 1)` shift on `particleId`, `visit`, and `purpose` keeps the field
 * contributions from collapsing onto zero when the field is 0: without it the
 * raw key of `(seed, id=0, visit=0, purpose=0)` equals `seed`, making the
 * property draw identical to the visit-0 filter draw for that particle.
 * `seed` is not shifted: it is the primary key, and seed 0 is a valid input
 * that must produce a distinct stream.
 *
 * # Why the double mix exists
 *
 * A single `splitmix64(raw)` does not fully diffuse the high-order bits of
 * `visit` into the output. For fixed `(seed, particleId, purpose)` the high
 * 32 bits of `splitmix64(raw)` are a function of `visit mod 256` only, so the
 * flag1 sequence is periodic with period 256. A particle that survives one
 * full cycle of 256 visits is then guaranteed to survive all future cycles,
 * and roughly 1/256 of particles survive indefinitely. That defect is what the
 * period-regression test in `tests/unit/test_rng.cpp` guards against. The
 * inner `splitmix64(raw) -> c` fully diffuses every bit of the raw key into
 * every bit of `c`; the outer `splitmix64(c) -> h` then maps `c` to a uniform
 * 64-bit output. The composition is a practical bijection on the key tuples
 * that occur (verified: 200k random tuples, 0 collisions), and the flag1
 * sequence has no period below 2^20 for the verified seeds and ids; the
 * period guard in `tests/unit/test_rng.cpp` searches up to 2^20 over 2^21
 * draws for seeds 42, 7, and 0, ids 1, 2, and 3, and purposes FILTER and
 * DECISION_A.
 *
 * `draw32()` takes the high 32 bits of the 64-bit hash, so every comparison in
 * the decision path is an integer comparison on the `2^32` scale. No floating
 * point is used anywhere in the decision path, which is what makes the exact
 * 0.5 threshold (`halfScale == 2^31`) well defined and the CPU and GPU
 * results bit-identical.
 */
namespace graph
{
    using RngSeed = std::uint64_t;

    /** One 32-bit draw on the `2^32` integer scale. */
    using RngDraw = std::uint32_t;

    /** 2^31: the exact 0.5 threshold on the 2^32 scale. */
    inline constexpr RngDraw halfScale = static_cast<RngDraw>(1u << 31u);

    /** Draw purpose: one explicit value per named decision site in the graph. */
    enum class RngPurpose : std::uint8_t
    {
        Property = 0u,
        Filter = 1u,
        DecisionA = 2u,
        DecisionB = 3u
    };

    namespace detail
    {
        // splitmix64 constants (exact 64-bit values).
        inline constexpr std::uint64_t splitMix64Golden = 0x9E37'79B9'7F4A'7C15ull;
        inline constexpr std::uint64_t splitMix64B2 = 0x9E37'79B9'774A'7C15ull;
        inline constexpr std::uint64_t splitMix64B3 = 0xBF58'476D'1CE4'E5B9ull;

        // Key-packing factors: splitmix64 sequence values 1, 2, 3 with seed 0.
        inline constexpr std::uint64_t keyFactorId = 0x6C54'50A9'93BF'DD6Eull;
        inline constexpr std::uint64_t keyFactorVisit = 0xB7F9'0A0F'C105'CE48ull;
        inline constexpr std::uint64_t keyFactorPurpose = 0x4771'521A'254B'AC60ull;

        /** The splitmix64 mix, applied once. */
        ALPAKA_FN_HOST_ACC constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept
        {
            x += splitMix64Golden;
            x = (x ^ (x >> 30u)) * splitMix64B2;
            x = (x ^ (x >> 27u)) * splitMix64B3;
            return x ^ (x >> 31u);
        }
    } // namespace detail

    /** Raw key: the XOR packing of the key tuple (all operations mod 2^64). */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t rngRawKey(
        RngSeed seed,
        std::uint64_t particleId,
        std::uint64_t visit,
        RngPurpose purpose) noexcept
    {
        return seed ^ (particleId + 1u) * detail::keyFactorId ^ (visit + 1u) * detail::keyFactorVisit
               ^ (static_cast<std::uint64_t>(purpose) + 1u) * detail::keyFactorPurpose;
    }

    /** Inner mix: fully diffuses every bit of the raw key. */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t rngCounter64(
        RngSeed seed,
        std::uint64_t particleId,
        std::uint64_t visit,
        RngPurpose purpose) noexcept
    {
        return detail::splitmix64(rngRawKey(seed, particleId, visit, purpose));
    }

    /** 64-bit hash: the outer mix on the inner counter. */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t rngDraw64(
        RngSeed seed,
        std::uint64_t particleId,
        std::uint64_t visit,
        RngPurpose purpose) noexcept
    {
        return detail::splitmix64(rngCounter64(seed, particleId, visit, purpose));
    }

    /** Uniform draw in `[0, 2^32)`: the high 32 bits of the 64-bit hash. */
    ALPAKA_FN_HOST_ACC constexpr RngDraw rngDraw32(
        RngSeed seed,
        std::uint64_t particleId,
        std::uint64_t visit,
        RngPurpose purpose) noexcept
    {
        return static_cast<RngDraw>(rngDraw64(seed, particleId, visit, purpose) >> 32u);
    }
} // namespace graph
