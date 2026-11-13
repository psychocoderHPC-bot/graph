<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# Acceptance checklist

Release gate for the initial implementation, per design sections 9 and 10.
State of the tree at commit `a12bfbe7c87497c91aaa8503a7fc5545c8c3d169`
(`main`). Sections 4, 6, and 8 also cite the W17 harness run at merge commit
`807b4375c1a49faa59ebae56fc2c251e6ce8a2d4`, which adds only the harness script
and documentation to that tree. Every command below carries the hal environment
scrub prefix:

```sh
env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH
```

`hal` reports `sharing=unknown`, so timings are presence evidence, not
benchmarks (`docs/known-limitations.md`).

## 1. CPU configure, build, and CTest

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRAPH_ENABLE_CUDA=OFF
cmake --build build-cpu -j 16
ctest --test-dir build-cpu --output-on-failure --output-junit junit-cpu.xml --timeout 600 --no-tests=error
```

Expected: configure and build succeed, 143/143 tests pass, 0 skipped.
Status: **PASS** on `a12bfbe7` (W16 run, 2026-11-13, hal container
`c692209a108c`, 17.85 s).

## 2. CUDA configure, build, and CTest

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGRAPH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j 16
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-cuda --output-on-failure \
    --output-junit junit-cuda.xml --timeout 900 --no-tests=error
```

Expected: configure and build succeed, 161/161 tests pass, 0 skipped. A
skipped CUDA test fails the gate (the harness checks the JUnit report
independently of the CTest exit status).
Status: **PASS** on `a12bfbe7` (W16 run, 37.14 s, A30, driver 615.71.09,
CUDA 13.4.92).

## 3. Example runs

Target `graph_particle_feedback` in `build-*/examples/particle_feedback/`.

| Case | Command | Expected | Status |
| --- | --- | --- | --- |
| default | `--backend cpu --particles 65536 --chunk-size 128 --seed 42` | exit 0; histogram rows equal the `### N=65536 seed=42` section of `tests/reference/test-vectors.md`; counts sum to 65536 | **PENDING** (no target) |
| zero | `--backend cpu --particles 0 --chunk-size 128 --seed 42` | exit 0; empty histogram | **PENDING** (no target) |
| chunk100 | `--backend cpu --particles 65536 --chunk-size 100 --seed 42` | exit 0; rows identical to the default case, so one fixture per `(N, seed)` | **PENDING** (no target) |
| CUDA default | `--backend cuda --particles 65536 --chunk-size 128 --seed 42` | exit 0; rows identical to the CPU default case | **PENDING** (no target) |

`tests/reference/test-vectors.md` holds the committed fixtures for
N=0, 1, 7, 128, 1000, 4096 (seeds 42 and 7) and 65536 (seeds 42 and 7).

## 4. 20x pool repeat

`scripts/hal-acceptance.sh --pool-repeat 20` runs the CUDA CTest cases matching
`-R pool` 20 times each through `ctest --repeat until-fail:20`.
Status: **PASS** on `807b4375` (W17 run: 480/480 case runs, 24 cases x 20,
0 failed, 87.00 s; `pool-repeat.log` holds 480 start lines and 480 result
lines).

## 5. Ladder and worker-count sweep

Both are CTest cases in the CUDA suite, so they run inside section 2:

- `pool gpu correctness ladder: rising worker count, exactly-once sink`
  (rungs 1w, 2w, 8w, 64w, plus the full-geometry rung with 17 descriptors and
  712 items).
- `pool gpu worker-count sweep: the former abort threshold stays green`
  (128, 192, 256, 384, 512, 768, 896 workers; requires the 256 rung to run).

Status: **PASS** on `a12bfbe7` (W16 run). The W16 ladder line:

```text
[ladder] rung=0 full geometry workers=896 accepted=17 imported=17 claimed=69
executed=69 published=69 retired=23 distinctWorkers=68 items=712
sink(correct/dup/missing/wrong)=712/0/0/0 stopReason=1 OK
```

## 6. Memcheck run

`scripts/hal-acceptance.sh --sanitize` runs compute-sanitizer memcheck over two
CUDA groups: `[graph][pool][ladder],[graph][pool][probe]` and
`[chunk_assembly][gpu]`.
Status: **PASS** on `807b4375` (W17 run). Both
`sanitizer-memcheck-pool-ladder-probe.log` and
`sanitizer-memcheck-chunk-publication.log` report `ERROR SUMMARY: 0 errors`.

## 7. Format and REUSE

```sh
scripts/check-format.sh   # clang-format 20.1.8, dry-run, --Werror
scripts/reuse-lint.sh     # reuse 6.3.0
```

Status: **PASS** on `a12bfbe7` (W16 run: 65 tracked files, no differences;
REUSE 3.3 compliant).

## 8. Recorded measured facts the gate depends on

Each fact below is quoted from an artifact that exists.

| Fact | Value | Source |
| --- | --- | --- |
| Worker count of the binder-instantiated kernel | 90 registers/thread with the test toy binder; 20 blocks/SM; 1120 blocks | `docs/known-limitations.md`, `docs/architecture.md`; P1 probe `results/p2_fat.txt` (fat 84 registers: 20 blocks/SM, 1120 coresident) |
| Wave actually used by the pool tests | 896 workers, 16 residents/SM (128-register floor over the captured 48) | W16 `diagnostics-cuda.txt`: `workers=896 ... residentsPerSm=16` |
| Persistent kernel launched once per pool | `launchCount=1` in every pool record | W16 `diagnostics-cuda.txt` (`drain`, `idle-resume`, `pressure`) |
| Selected atomic transport on the A30 | `OrderedSystemAtomic`, with `hostNativeAtomicSupported=0`, `deviceSystemScopeRmwNative=0`, `concurrentManagedAccess=1`, `mappedPinnedAllocation=1` | W16 `diagnostics-cuda.txt` capability record |
| Update loss for concurrent host/device RMW on mapped memory | measured on the A30 by the P1 probe; recorded in `docs/known-limitations.md`; the A30's eligible concurrent-RMW allocation is managed memory | P1 probe `results/p1_c.txt` (mapped RMW alone: 256000/256000 unique), `docs/known-limitations.md`, `tests/unit/test_system_atomic.cpp` |
| Mapped pinned ingress throughput | about 1.6 to 1.67 M msg/s with a single-warp consumer (1671894.5 and 1603948.7 msg/s) | P1 probe `results/p1_a.txt`, `results/p1_a_egress_per_message.txt`; `artifacts/P1-protocol-probes.md` section 6 |
| Copy-queue overlap | requires pinned buffers on CUDA | `copy_queue overlaps a busy worker queue on CUDA`; `copy_queue rejects pageable host buffers on CUDA` |
| Sink accounting at all worker counts 128..896 | 712/0/0/0 (correct/duplicate/missing/wrong) at every rung of the ladder and sweep | W16 `diagnostics-cuda.txt`; the sweep asserts `sink.correct == totalItems` and zero duplicate/missing/wrong per worker count |
| Memcheck | clean: both CUDA memcheck groups report `0 errors` | W17 `sanitizer-memcheck-pool-ladder-probe.log`, `sanitizer-memcheck-chunk-publication.log` (`ERROR SUMMARY: 0 errors`) |

## 9. Open items

1. **Example and end-to-end gate.** `examples/particle_feedback/CMakeLists.txt`
   adds no target at `a12bfbe7`. Sections 3 and 4 of the design (example runs,
   CTest example registration) are not exercisable until the wiring node lands.
   The CLI header, fixtures, and oracle are ready.
2. **Chunk-assembly dataflow node.** The in-flight node that connects
   `generate -> filter -> A/B -> feedback` through the pool is not in this
   tree. `tests/integration/test_particle_semantics.cpp` covers the worklist
   semantics off-pool against the scalar oracle, but the pipeline itself runs
   nowhere yet.
3. **Chunk-publication barrier guard.** No automated regression guard fails on
   the A30 if the `WarpBarrier` in `StreamReservation::commit()` is reverted
   to a fence plus `warp.all`. `tests/integration/test_chunk_publication_gpu.cpp`
   observed 0 stale elements in 200 launches of 100 rounds, and the patched
   variant also stayed clean, so the window is not reliably observable on this
   GPU (`docs/known-limitations.md`).
4. **Register capture.** The build-time capture probes the rendezvous kernel,
   not the pool kernel with a user binder. The wave therefore rests on the
   128-register floor until `PoolConfig::registerHint` names a measured value
   (`docs/known-limitations.md`).
5. **CI pipeline.** The GitLab workflow is committed; the project connection,
   runner tags, and the CUDA image are not provisioned. No pipeline result
   exists for any commit (`docs/ci.md`).

## 10. PASS/PENDING table per design section 9/10

| Item (design section) | Status |
| --- | --- |
| 9: CPU build and CTest | PASS |
| 9: CUDA build and CTest, no skipped tests | PASS |
| 9: example registered and run on CPU | PENDING (no target) |
| 9: example registered and run on NVIDIA | PENDING (no target) |
| 9: example run at a size that exercises multiple workers and rounds | PENDING (no target) |
| 9: exact expected histogram for at least one fixed-seed fixture | PENDING (no test consumes `tests/reference/test-vectors.md` end-to-end) |
| 9: HAL record with commit, dependencies, toolchain, counts, occupancy, transport | PASS (`docs/hal-acceptance.md`, W16 run) |
| 9: automated CPU checks | PENDING (job committed, pipeline never ran) |
| 9: automated HAL CPU/NVIDIA checks | PENDING (jobs committed, runners not provisioned) |
| 10.1: pinned revision, CPU/CUDA builds, device pointers, warp sync, occupancy, host/device atomics probed | PASS |
| 10.2: ingress, packet storage, ready ring validated with scripted jobs, late ingress, contention | PASS |
| 10.3: chunk assembly, rotation, pressure, quiescence with exact-once feedback tests | PASS |
| 10.4: particle functors and example compared on HAL CPU and NVIDIA | PARTIAL (functors verified; example absent) |
| 10.5: documentation, CI, push, validated commit recorded | PARTIAL (docs and CI config committed; pipeline pending) |
| 10: same functor structs on both backends | PASS |
| 10: one persistent GPU kernel serves the entire run | PASS (toy pipeline; `launchCount=1`) |
| 10: host submissions reach the kernel while it is running | PASS (`pool core accepts a late submission to an already running idle pool`, `ingress round trip`) |
| 10: pool exits with every generated particle exactly once in the histogram | PARTIAL (toy sink 712/0/0/0 and histogram unit tests; particle end-to-end absent) |

## What remains before delivery

1. Wire the example target, register it with CTest on both backends, and run
   the three fixture cases plus the CUDA comparison.
2. Add the end-to-end test that consumes `tests/reference/test-vectors.md`.
3. Run the example at a multi-worker, multi-round size on the A30.
4. Provision the GitLab runners and run the committed workflow end to end.
5. Close the chunk-publication barrier guard gap, or document it as a known
   limitation with the P1 probe evidence.
