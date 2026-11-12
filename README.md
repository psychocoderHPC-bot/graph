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
