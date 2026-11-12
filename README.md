<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# graph

Persistent warp-worker task graph on alpaka3.

`graph` is a C++20 library and example that runs a task graph on a persistent
accelerator worker pool. One hardware warp is one worker; all lanes execute the
same indirect device-function call on one packet at a time. The pool is launched
once per lifetime, accepts host submissions while it is running, drains on an
ordered close, and reports per-particle results. A particle-feedback example
exercises a graph with a feedback cycle and prints an exact final-weight
histogram. The project validates on a CPU backend and on an NVIDIA GPU.

## Status

Implemented and green at this commit (HAL acceptance record:
`docs/hal-acceptance.md`): the build, the library sources, the unit and
integration tests, and the hardware gates. The CPU and CUDA CTest suites pass
(119/119 and 127/127), and the CUDA suite covers the residency rendezvous, the
copy-queue overlap, the chunk-assembly publication guard, the ready-ring
cross-warp visibility, and the particle-gpu kernel comparison. Format and REUSE
pass.

Pending at this commit: the host `Pool` object and the persistent pool kernel
that combine ingestion, claim, dispatch, and drain into one launch; the
particle-feedback example target (`examples/particle_feedback/CMakeLists.txt`
adds no target, so both example gates are `NOT-BUILT`); the end-to-end
particle pipeline on the pool; and the GitLab CI runners. Details:
`docs/known-limitations.md`.

## Pinned dependency

| Component | Revision | Branch |
| --- | --- | --- |
| [alpaka](https://github.com/alpaka-group/alpaka) | [`4780dc786a2eb1a5452a1c66d94a8b442d411ab8`](https://github.com/alpaka-group/alpaka/commit/4780dc786a2eb1a5452a1c66d94a8b442d411ab8) | `dev_v3.x` |

The pinned revision is the reference for architecture, code style, and
formatting. Builds may override the checkout location explicitly for local
development.

## Layout

| Path | Purpose |
| --- | --- |
| `include/graph/` | Library headers: rings, pool, chunk assembly, RNG, functors, model, ABI |
| `include/graph/detail/` | Backend adapters, launch geometry, rendezvous, warp primitives, copy queue |
| `examples/particle_feedback/` | Example graph description, functors, and CLI (example target pending) |
| `tests/unit/` | Focused Catch2 cases |
| `tests/integration/` | Pool, pipeline, and hardware integration tests |
| `tests/reference/` | Scalar oracle and the acceptance test vectors |
| `probes/` | CUDA device-ABI and warp-barrier probes |
| `cmake/` | Build modules, options, and register capture |
| `docs/` | Design, architecture, numerics, limitations, build, CI, and acceptance records |
| `scripts/` | Local helper scripts |
| `LICENSES/` | Full license texts |

## Build and test

Quickstart for the CPU and CUDA configurations is below. The full local
validation that matches CI, the pinned tool versions, the environment scrubbing
on `hal`, and the JUnit report commands are in `docs/build.md`.

### Documentation

- `docs/design.md`: design summary and index
- `docs/architecture.md`: module map, data path, ordering rules, launch geometry
- `docs/numerics.md`: exact RNG construction and particle semantics
- `docs/known-limitations.md`: current engineering gaps
- `docs/build.md`: build and test commands
- `docs/ci.md`: CI pipeline status
- `docs/hal-acceptance.md`: HAL hardware acceptance record

## License

MPL-2.0. See `LICENSE` and `LICENSES/MPL-2.0.txt`. Files copied from alpaka
retain their upstream copyright notices.

## Build and run the bootstrap tests

The M0 bootstrap configures, builds, and runs a Catch2 smoke test for both a CPU
and an NVIDIA CUDA configuration. The smoke test selects the backend from the
`GRAPH_HAS_CUDA` feature macro and runs one fill kernel, so it exercises the
device selection, queue, kernel launch, and copy path.

alpaka is fetched at the pinned revision
`4780dc786a2eb1a5452a1c66d94a8b442d411ab8`. Pass
`-DGRAPH_ALPAKA_SOURCE_DIR=/path/to/alpaka` to use an existing checkout instead
(useful for offline builds). Catch2 is pinned to `v3.15.0`, the revision the
pinned alpaka test tree uses.

### CPU

```bash
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRAPH_ENABLE_CUDA=OFF
cmake --build build-cpu -j 16
ctest --test-dir build-cpu --output-on-failure --timeout 120
```

The CPU configuration does not require a CUDA toolkit. `GRAPH_ENABLE_OPENMP`
defaults to `ON` when OpenMP is found and to `OFF` otherwise.

### NVIDIA CUDA

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j 16
ctest --test-dir build-cuda --output-on-failure --timeout 120
```

`GRAPH_ENABLE_CUDA=ON` enables the CUDA language and relocatable device code
(`alpaka_RELOCATABLE_DEVICE_CODE=ON`). When `CMAKE_CUDA_ARCHITECTURES` is not
set, the configure step reads the compute capability of the first visible GPU
through `nvidia-smi` and falls back to `80` (NVIDIA A30/A100); the default used
by the bootstrap validation on `hal` is `80` for the A30.
