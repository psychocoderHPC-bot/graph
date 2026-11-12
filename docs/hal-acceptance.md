<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# HAL acceptance record

This record covers the repository state at `main` commit
`52be5c72efb8d313ee8cfd30f31a15a47aa7f124`. The CPU and CUDA test suites pass
there. The persistent-pool and particle-feedback gates are not available at
that commit, so they are not claimed as verified. The example gates are
recorded as NOT-BUILT.

The harness run below used branch commit
`10184a8afb12a536b719ff16939fc70460f2b271`. Relative to `52be5c72` that commit
adds `scripts/hal-acceptance.sh` and `docs/hal-acceptance.template.md` only.
`CMakeLists.txt`, `cmake/`, `include/`, `tests/`, `examples/`, `probes/`, and
`.gitlab-ci.yml` are unchanged, so the build inputs under test are the ones at
`52be5c72`.

## Environment

| Field | Value |
| --- | --- |
| Validated code commit | `52be5c72efb8d313ee8cfd30f31a15a47aa7f124` (`main`) |
| Harness commit | `10184a8afb12a536b719ff16939fc70460f2b271` |
| Run date | 2026-11-12 (UTC) |
| Working tree | clean (`git status --porcelain` empty) |
| Host | hal container `c692209a108c`, Linux `6.8.0-138-generic`, x86_64 |
| CPU | AMD EPYC 7452 32-Core Processor, 64 logical CPUs visible |
| GPU | NVIDIA A30, compute capability 8.0 as reported by `nvidia-smi`, 24576 MiB |
| NVIDIA driver | 615.71.09 |
| CUDA toolkit | 13.4.92 (`nvcc` release 13.4) |
| C++ compiler | g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| alpaka pin | `4780dc786a2eb1a5452a1c66d94a8b442d411ab8` |
| Catch2 | `v3.15.0` |
| Build type | `Release`, Ninja |
| Build parallelism | `-j 16` |
| CUDA architecture | `CMAKE_CUDA_ARCHITECTURES=80` |
| CUDA device selection | `CUDA_VISIBLE_DEVICES=0` |

The run used the hal local dependency checkouts for alpaka and Catch2. The
alpaka checkout is at the pinned commit above and Catch2 is version 3.15.0.
Both dependency locations are optional: without them CMake fetches the same
pins through `FetchContent`.

## Gate results

| Gate | Result | Evidence |
| --- | --- | --- |
| CPU configure and build | PASS | `-DGRAPH_ENABLE_CUDA=OFF`, `Release`, Ninja, `-j 16` |
| CPU CTest | PASS | 119/119 passed, 0 failed, 0 skipped |
| CUDA configure and build | PASS | `-DGRAPH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`, `Release`, Ninja, `-j 16` |
| CUDA CTest | PASS | 127/127 passed, 0 failed, 0 skipped |
| No skipped CUDA tests | PASS | `skipped=0`, `disabled=0` in `junit-cuda.xml` |
| Example on CPU | NOT-BUILT | no `graph_particle_feedback` target exists at `52be5c72` |
| Example on CUDA | NOT-BUILT | no `graph_particle_feedback` target exists at `52be5c72` |
| Format | PASS | clang-format 20.1.8, 50 tracked files, no differences |
| REUSE | PASS | reuse 6.3.0, 72/72 files with license information at the harness commit, REUSE 3.3 compliant |
| Overall | PASS | every executed gate passed; the example gate is not claimed |

The CUDA suite has eight more CTest entries than the CPU suite: the CUDA
instantiations of chunk assembly, copy-queue overlap and rejection, graph
vocabulary, two ready-ring cross-warp visibility cases, and CUDA residency.
None is skipped.

## Commands

Every configure, build, test, and example command is wrapped in the hal
environment scrub prefix:

```sh
env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
    -u CPATH -u CPLUS_INCLUDE_PATH
```

The harness invocation used for this record:

```sh
ALPAKA=/path/to/alpaka-checkout     # at 4780dc786a2eb1a5452a1c66d94a8b442d411ab8
CATCH2=/path/to/catch2-checkout     # v3.15.0
ARTIFACT=$(mktemp -d)

scripts/hal-acceptance.sh \
    --alpaka-dir "$ALPAKA" \
    --catch2-dir "$CATCH2" \
    --jobs 16 \
    --artifact-dir "$ARTIFACT"
```

Without local dependency checkouts, let CMake fetch the pins:

```sh
ARTIFACT=$(mktemp -d)
scripts/hal-acceptance.sh --jobs 16 --artifact-dir "$ARTIFACT"
```

The per-configuration commands the harness ran are equivalent to:

```sh
cmake -S . -B "$ARTIFACT/build-cpu" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGRAPH_ENABLE_CUDA=OFF \
    -DGRAPH_ALPAKA_SOURCE_DIR="$ALPAKA" \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$CATCH2"
cmake --build "$ARTIFACT/build-cpu" -j 16
ctest --test-dir "$ARTIFACT/build-cpu" --output-on-failure \
    --output-junit "$ARTIFACT/junit-cpu.xml" --timeout 600 --no-tests=error

cmake -S . -B "$ARTIFACT/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGRAPH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DGRAPH_ALPAKA_SOURCE_DIR="$ALPAKA" \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$CATCH2"
cmake --build "$ARTIFACT/build-cuda" -j 16
CUDA_VISIBLE_DEVICES=0 ctest --test-dir "$ARTIFACT/build-cuda" \
    --output-on-failure --output-junit "$ARTIFACT/junit-cuda.xml" \
    --timeout 900 --no-tests=error

scripts/check-format.sh
scripts/reuse-lint.sh
```

`--cuda-arch` defaults to `80`. The value must match the target device. The A30
reports compute capability 8.0, and the CI job also builds `80`.

The harness bounds each configure, build, and lint command with `timeout`
(1800 s, 3600 s, and 600 s by default). CTest has its per-test timeout and each
example run has a wall-clock timeout of 300 s by default. No gate waits without
a bound.

## Test counts

| Configuration | Total | Passed | Failed | Skipped | Disabled | Suite time |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | 119 | 119 | 0 | 0 | 0 | 17.20 s |
| CUDA | 127 | 127 | 0 | 0 | 0 | 28.98 s |

Counts come from the CTest JUnit reports (`junit-cpu.xml`, `junit-cuda.xml`).
A skipped CUDA test fails the harness; the gate is independent of the CTest
exit status so a hardware test cannot pass by being skipped.

## Occupancy, geometry, and atomic transport

The harness extracts these lines from the JUnit `system-out` blocks and the
configure logs. Values are reported by the code under test; no value is
synthesized when the test output does not contain it.

CUDA build:

```text
[residency] backend=cuda geometry blocks=1792 threadsPerBlock=32 warpsPerBlock=1 workerCount=1792 smCount=56 residentsPerSm=32 registersPerThread=20 captured=1
[residency] backend=cuda arrived=1792/1792 allArrived=1 completed=1792 minObserved=1792 launchCount=1 singleWave=1
-- graph: captured persistent-kernel register usage: 20 registers/thread
system_atomic capability record: device="NVIDIA A30" attribute{deviceSystemScopeRmwNative=0,hostNativeAtomicSupported=0,concurrentManagedAccess=1,mappedPinnedAllocation=1,mappedSystemAtomicVerified=0} verification{mappedPinnedAllocation=1,expected=64001,orderedObserved=64001,nativeObserved=64001,orderedVerified=1,nativeVerified=1} effective{deviceSystemScopeRmwNative=1,mappedSystemAtomicVerified=1} selected=OrderedSystemAtomic selectedNativeOptIn=NativeSystemRmw
```

CPU build:

```text
[residency] backend=cpu geometry blocks=64 threadsPerBlock=1 warpsPerBlock=1 workerCount=64 smCount=64 residentsPerSm=1 registersPerThread=128 captured=0
[residency] backend=cpu arrived=64/64 allArrived=1 completed=64 minObserved=64 launchCount=1 singleWave=1
system_atomic capability record: device="AMD EPYC 7452 32-Core Processor" attribute{deviceSystemScopeRmwNative=1,hostNativeAtomicSupported=1,concurrentManagedAccess=0,mappedPinnedAllocation=1,mappedSystemAtomicVerified=0} verification{mappedPinnedAllocation=1,expected=1001,orderedObserved=1001,nativeObserved=1001,orderedVerified=1,nativeVerified=1} effective{deviceSystemScopeRmwNative=1,mappedSystemAtomicVerified=1} selected=OrderedSystemAtomic selectedNativeOptIn=NativeSystemRmw
-- graph: CUDA is disabled; persistent-kernel register count uses the 128-register fallback
```

The committed record adds one non-code file, so a re-run of the REUSE gate over
this tree reports 73/73 files with license information. The gate passed at the
harness commit with 72/72.

The selected ingress transport is `OrderedSystemAtomic` on both backends. The
raw system-scope RMW transport is reachable only through the explicit opt-in
and is reported as `NativeSystemRmw` after the live verification passes. On the
A30 the driver reports `deviceSystemScopeRmwNative=0`, and the verification
observes the ordered path correctly.

The CUDA build also reports the cross-backend check:

```text
[particle_gpu] CUDA: 32 lanes matched the host cores
[particle_gpu] CUDA-vs-CPU per-lane result checksum: equal (all 32 lanes)
```

## Example results

The example is not built at `52be5c72`. `examples/particle_feedback/CMakeLists.txt`
adds no target; the directory holds the CLI header and a placeholder only. The
harness records `example: not-built` for both configurations and does not claim
the example gate. Once the target exists the gate becomes an executed pass or
fail automatically.

The harness expects the example node to expose:

- target `graph_particle_feedback`, built to
  `examples/particle_feedback/graph_particle_feedback` (the harness also probes
  `examples/` and `bin/`, and falls back to a `find` in the build tree);
- options `--backend cpu|cuda`, `--particles N`, `--chunk-size Y`, `--seed S`;
- exit status 0 for a successful run;
- histogram lines as bare `weight count` integer pairs, sorted ascending by
  weight, with counts summing to `N`;
- identical rows for the same `(N, seed)` at every chunk size, so one committed
  fixture per `(N, seed)` is sufficient.

The harness runs three cases per backend:

| Case | Arguments |
| --- | --- |
| default | `--particles 65536 --chunk-size 128 --seed 42` |
| zero | `--particles 0 --chunk-size 128 --seed 42` |
| chunk100 | `--particles 65536 --chunk-size 100 --seed 42` |

It checks the exit status, the histogram total, and an exact row match against
the committed fixture section `### N=<N> seed=<S>` in
`tests/reference/test-vectors.md`. For `N=0` the expected section is empty.
Without a fixture the case records `exact=no-fixture` and still checks the
total.

## How to re-run

On a host with a C++20 compiler, CMake 3.25 or newer, Ninja, `clang-format`
20.1.8, `reuse` 6.3.0, and, for the CUDA configuration, `nvcc` and an NVIDIA
GPU:

```sh
pip install "clang-format==20.1.8" "reuse==6.3.0"
scripts/hal-acceptance.sh --artifact-dir "$(mktemp -d)"
```

The run writes everything to the artifact directory and never modifies the
checkout unless `--write-doc` is passed. The artifact directory contains:

| File | Content |
| --- | --- |
| `environment.txt` | machine-readable `key=value` environment block |
| `nvidia-smi-L.txt`, `nvidia-smi-query.csv` | raw GPU inventory |
| `configure-<cfg>.log`, `build-<cfg>.log` | configure and build logs |
| `junit-<cfg>.xml`, `ctest-<cfg>.log` | JUnit report and CTest output |
| `counts-<cfg>.txt` | parsed totals, failures, skips, disabled, passes |
| `example-<cfg>-<case>.txt` | raw example output per case |
| `diagnostics.md` | occupancy, geometry, register, and transport lines |
| `summary.md`, `hal-acceptance.md` | gate table and filled record |

Exit status: `0` when every executed gate passed and nothing is pending, `1`
when a gate failed, `2` when no gate failed but a configuration or required
gate did not run. `--allow-pending` turns the `2` case into `0` for partial
runs. `--allow-dirty` records a dirty tree instead of refusing to run.

## What is not yet verified

- The particle-feedback example and its histogram. The target does not exist at
  `52be5c72`. Both example gates are `NOT-BUILT`.
- The end-to-end particle pipeline (generate, filter, feedback, destroy) and
  the exact-once final-weight accounting on the persistent pool. These arrive
  with the pool and example milestones and are not exercised by this record.
- The GitLab runner path from `docs/ci.md`. No pipeline has run; the HAL jobs
  remain pending until the runners and the CUDA image are provisioned. This
  record is a manual HAL run, not a CI result.
- The lint gates ran on hal, not in the `python:3.13-slim` CI image. The tool
  versions match the pins, but the image is not the CI image.
- Any GPU other than the A30 allocated to this run, and any architecture other
  than `sm_80`.
