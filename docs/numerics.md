<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->
# Numerics
The particle-feedback example is integer-only. Every random draw is a
pure function of the key tuple `(seed, particleId, visit, purpose)`, so
results are independent of chunk size, backend, worker, and scheduling order.
No floating point appears in the decision path, which makes the exact 0.5
threshold well defined and the CPU and GPU results bit-identical.
`include/graph/rng.hpp` (contract mirrored by the scalar reference in
`tests/reference/oracle.hpp`):

```
raw = seed
    ^ (particleId + 1) * G1
    ^ (visit + 1) * G2
    ^ ((uint64_t)purpose + 1) * G3
c   = splitmix64(raw)
h   = splitmix64(c)
draw32 = h >> 32
```

All operations are mod 2^64. The constants `G1 = 0x6C5450A993BFDD6E`
(factor of `particleId`), `G2 = 0xB7F90A0FC105CE48` (factor of `visit`), and
`G3 = 0x4771521A254BAC60` (factor of `purpose`) are consecutive outputs of the
splitmix64 sequence with seed 0 (stream values 1, 2, 3), so each is a distinct
64-bit odd constant. splitmix64 uses the constants `0x9E3779B97F4A7C15`
(addition), `0x9E3779B9774A7C15` (first multiply), and `0xBF58476D1CE4E5B9`
(second multiply), with the shifts 30, 27, and 31.
The `(x + 1)` shift on `particleId`, `visit`, and `purpose` keeps the field
contributions from collapsing onto zero when the field is 0: without it the
raw key of `(seed, id=0, visit=0, purpose=0)` equals `seed`, making the
property draw identical to the visit-0 filter draw for that particle. `seed`
is not shifted: it is the primary key, and seed 0 must produce a distinct
stream.
`RngPurpose` has one value per named decision site: 0 = `Property`
(draw at generation), 1 = `Filter` (bin routing), 2 = `DecisionA` (Bin 0
survival), 3 = `DecisionB` (Bin 1 survival). The key is packed as the XOR of
the four field contributions above. The purpose is widened to 64 bits and
shifted by 1 before multiplication, so a purpose of 0 does not zero its
contribution.
A single `splitmix64(raw)` does not fully diffuse the high-order bits of
`visit` into the output. For fixed `(seed, particleId, purpose)` the high 32
bits of `splitmix64(raw)` are a function of `visit mod 256` only, so the
flag-1 sequence is periodic with period 256. A particle that survives one
cycle of 256 visits is guaranteed to survive all future cycles, and roughly
1/256 of particles survive indefinitely. The inner `splitmix64(raw) -> c`
fully diffuses every bit of the raw key into every bit of `c`; the outer
`splitmix64(c) -> h` maps `c` to a uniform 64-bit output. The composition is
a practical bijection on the key tuples that occur (verified: 200k random
tuples, 0 collisions), and the flag-1 sequence has no period below 2^20 for
the verified seeds and ids. `tests/unit/test_rng.cpp` guards that exact claim:
it samples 2^21 draws and searches for a period up to 2^20 over seeds 42, 7,
and 0, ids 1, 2, and 3, and purposes FILTER and DECISION_A.

`draw32()` takes the high 32 bits of the 64-bit hash, so every
comparison in the decision path is an integer comparison on the 2^32 scale.
The exact 0.5 threshold is `halfScale = 2^31`; a decision draw survives when
`draw >= 2^31`. `tests/integration/test_particle_gpu_kernel.cpp` constructs
the exact boundary by finding keys with `draw32(DECISION_A) == 2^31` and
`== 2^31 - 1` and checks that only the first survives.

- `GenerateParticles`: input request `(firstId, count, seed)`, output
  filter input. `property = draw32(seed, id, 0, Property)`, weight 0, visit 0.
- `FilterByProperty`: input filter input, output Bin 0 or Bin 1.
  `draw32(seed, id, visit, Filter) >= property` to Bin 1, else Bin 0.
- `ProcessA`: input Bin 0, output feedback or sink.
  `draw32(seed, id, visit, DecisionA) >= 2^31`: weight += 1000, visit += 1, to
  filter input; else to sink.
- `ProcessB`: input Bin 1, output feedback or sink.
  `draw32(seed, id, visit, DecisionB) >= 2^31`: weight += 1, visit += 1, to
  filter input; else to sink.
- `DestroyParticles`: input sink, output none (terminal). Writes the final
  weight to the particle's unique result slot.

Field widths (`include/graph/particle_functors.hpp`,
`sizeof(Particle) == 24`): `id` is `uint64_t` (unique source id), `property`
is `uint32_t` (draw at generation, preserved forever), `weight` is `uint32_t`
(accumulated weight, starts at 0), `visit` is `uint32_t` (feedback rounds,
starts at 0).

Overflow policy: `weight` is a fixed-width unsigned 32-bit integer.
Every increment goes through `checkedWeightAdd()`, which reports overflow in
64-bit arithmetic instead of wrapping. An overflow surfaces through the
functor's per-call error out-parameter; it is a hard error, never a wrap.

A run is keyed by `(seed, id, visit, purpose)`. The same `(N, seed)`
produces the same multiset of final weights at every chunk size, on every
backend, with every worker count and scheduling order, because the draw of
every particle depends only on its key.
`tests/integration/test_particle_semantics.cpp` runs the worklist in three
orderings and requires identical results.

The acceptance fixtures are `tests/reference/test-vectors.md`, generated
by `tests/reference/oracle.cpp` with `maxVisits=1000000`. The file holds 9
cases: `N` in {0, 1, 7, 128, 1000, 4096, 65536} at seed 42, and `N` in
{4096, 65536} at seed 7. For every case `nonTerminating` is 0. The per-case
histogram rows, `maxVisit`, and an FNV-1a checksum over the destroyed weights
(id order) are the acceptance oracle.
