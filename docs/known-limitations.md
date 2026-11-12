<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->
# Known limitations
These are engineering facts about the current state of the tree, not
aspirations. Each item names the evidence and the gate that would close it.

## The persistent kernel's register capture is pending

The build captures the per-thread register count from the translation
unit that instantiates the kernel, via `-Xptxas -v`
(`cmake/GraphRegisters.cmake`). The probe that exists,
`cmake/GraphRegisterProbe.cu`, instantiates `RendezvousKernel`, the
rendezvous kernel. The pool kernel's own capture is a separate pending item.
Until it exists the launch geometry uses either the rendezvous kernel's
capture (20 registers per thread, measured on the A30 in
`docs/hal-acceptance.md`) or the 128-register fallback, and the current grid
may be conservative, or over-optimistic if the pool kernel uses more
registers. The `test_residency` CUDA case measures the rendezvous kernel, not
the pool kernel.

## Multi-warp-per-block is refused by design

`WarpBarrier::barrier()` maps to a block-wide barrier only when the
block holds exactly one warp. For more than one warp per block it refuses
(tripping a runtime flag the kernel must assert on), because a block-wide
barrier would also order the other warps in the block, a different,
co-residency-dependent schedule, and a bare vote orders nothing. The missing
prerequisite is a validated warp-local barrier.
`computeLaunchGeometry()` throws `std::invalid_argument` for a multi-warp
request unless the build defines `GRAPH_ENABLE_MULTI_WARP_BLOCKS`. The
measured headroom on the A30 is 3584 warps (64 warps per SM, 56 SMs) against
the 1792 workers the one-warp-per-block geometry uses.

## System-scope RMW on mapped memory is not contract-safe

`atomicAdd_system` on a mapped pinned allocation was measured working on
the A30: 256000 of 256000 unique results in the verification round-trip, even
though `cudaDevAttrHostNativeAtomicSupported` reported 0. The test family
`tests/unit/test_system_atomic.cpp` records that the A30 reports
`hostNativeAtomicSupported` as 0 and that the probe measured lost updates when
host and device RMW the same mapped word concurrently. The A30 reports
`concurrentManagedAccess=1`, so the eligible allocation for a concurrent RMW
is managed memory, not mapped pinned memory. That is why the ordered
transport, not the raw RMW, is the default. The ordered acquire/release
transport is the default (`IngressTransport::OrderedSystemAtomic`). The raw
RMW is reachable only through the explicit opt-in `atomicAddSystemNative`
after a live verification, and it costs about 597 ns per operation on the
A30, so it is unsuitable for per-job publication. No pool path uses it.

## Co-residency is arithmetic plus an empirical rendezvous

Co-residency is established by occupancy arithmetic over queried or
measured limits plus the startup rendezvous test. It is not a hardware
guarantee: alpaka has no occupancy API and no cooperative launch, and an
ordinary launch of an occupancy-sized grid does not promise that every block
is resident at the same time. The rendezvous test
(`tests/integration/test_residency.cpp`) is the gate that turns the arithmetic
into evidence: the host requires `completed == workerCount`, not merely
`arrived == workerCount`, because sequential waves leave earlier workers below
the expected count.

## The chunk-assembly publication guard cannot detect a reverted barrier

The cross-warp publication guard
(`tests/integration/test_chunk_publication_gpu.cpp`) runs 20 launches of
1000 iterations each (20000 published chunks, 640000 elements read from a
different warp) and checks for stale elements. It has observed 0 failures in
250+ patched launches on the measured GPU. A device-scope release fence
orders only the executing thread's own prior writes, so the guard cannot
detect a regression of the lane ordering to a vote or a bare leader fence on
this GPU: the reader warp may observe the payload through L2 even when the
lane stores are not ordered. The barrier rests on probe evidence
(`probes/warp_barrier_probe.cu`, `worstVoteInvisible=30`,
`worstBarrierInvisible=0`). This is a test-coverage gap, recorded here.

## `hal` sharing is `unknown`

The compute target reports `sharing=unknown`. The overlap and arrival
timings in `docs/hal-acceptance.md` (copy-queue overlap, residency wall time)
are presence and overlap evidence, not benchmarks. A benchmark run requires an
`exclusive` system.

## Acceptance gates still pending at this commit

The HAL acceptance record at `main` (`docs/hal-acceptance.md`) covers
the code at `52be5c72`. The CPU and CUDA CTest suites pass there (119/119 and
127/127). The following gates are not claimed at this commit:
- The particle-feedback example and its histogram. The target does not exist;
  `examples/particle_feedback/CMakeLists.txt` adds no target. Both example
  gates are `NOT-BUILT`.
- The end-to-end particle pipeline on the persistent pool and the
  exactly-once final-weight accounting. These arrive with the pool and example
  milestones. The worklist semantics are validated off-pool by
  `tests/integration/test_particle_semantics.cpp` against the committed
  fixtures.
- The GitLab runner path (`docs/ci.md`). No pipeline has run; the HAL jobs
  remain pending until the runners and the CUDA image are provisioned.
The green gates at this commit: CPU configure/build and CTest, CUDA
configure/build and CTest, format (clang-format 20.1.8), and REUSE
(reuse 6.3.0). The CUDA suite includes the residency rendezvous, the
copy-queue overlap, the chunk-assembly publication guard, the ready-ring
cross-warp visibility, and the particle-gpu kernel comparison; none is
skipped.
