<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# Design summary

`graph` runs a task graph on a persistent accelerator worker pool built on
alpaka3, pinned at
[`4780dc786a2eb1a5452a1c66d94a8b442d411ab8`](https://github.com/alpaka-group/alpaka/commit/4780dc786a2eb1a5452a1c66d94a8b442d411ab8)
(`dev_v3.x`).

- One persistent worker kernel runs per pool lifetime. It stays alive while
  idle, accepts host submissions from a mapped SPSC ingress ring, and stops only
  after an ordered close and a complete drain.
- One hardware warp is one worker. All lanes execute the same indirect,
  registered device-function call on one packet. The same functor structs run on
  CPU and GPU.
- A project-owned system-scope atomic adapter supplies the cross-system
  publication that the pinned alpaka `onAcc::atomicOp` does not cover.
- Work is distributed through a bounded device-side ready ring with explicit
  ownership and device-scope acquire/release publication.
- Packet pools hold fixed-capacity chunks. Each assembling stream keeps two
  physical chunks (`A`, `B`) and a logical counter, so one rotation per append
  suffices.
- Completion uses a quiescence protocol that tracks accepted, queued, active,
  reserved, and partial work. Queue emptiness alone does not prove completion.
- The example graph generates particles, filters by a random property, processes
  two bins with a feedback cycle, destroys particles, and prints the exact
  final-weight histogram.

References: [alpaka README](https://github.com/alpaka-group/alpaka/blob/4780dc786a2eb1a5452a1c66d94a8b442d411ab8/README.md),
[warp tutorial](https://github.com/alpaka-group/alpaka/blob/4780dc786a2eb1a5452a1c66d94a8b442d411ab8/docs/source/tutorial/warp.rst),
[atomic.hpp](https://github.com/alpaka-group/alpaka/blob/4780dc786a2eb1a5452a1c66d94a8b442d411ab8/include/alpaka/onAcc/atomic.hpp),
and the [contribution guide](https://github.com/alpaka-group/alpaka/blob/4780dc786a2eb1a5452a1c66d94a8b442d411ab8/CONTRIBUTING.md).
