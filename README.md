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

Status: **scaffolding**. The build, library sources, tests, and the example are
not implemented yet; see `docs/design.md`.

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
| `include/graph/` | Pool, packet schemas, queues, and backend adapters |
| `examples/particle_feedback/` | Portable example, functors, and example graph |
| `tests/unit/` | Focused Catch2 cases |
| `tests/integration/` | Worker pool and end-to-end pipeline tests |
| `docs/` | Design summary and later build/validation records |
| `scripts/` | Local helper scripts |
| `LICENSES/` | Full license texts |

## Build and test

Not available yet. The scaffolding commit contains no CMake project.

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
