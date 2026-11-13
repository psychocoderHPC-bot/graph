<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->
# Known limitations
Engineering facts about the current tree, each with its evidence and the gate that would close it.

## The register capture cannot see a user binder
`cmake/GraphRegisterProbe.cu` now instantiates the pool's own persistent entry point
(`detail::PoolKernel`) as well as `RendezvousKernel`, and `-Xptxas -v` reports the maximum. On the
A30 `PoolKernel<NullBinder>` compiles to 48 registers/thread (32 blocks/SM, the 1792-worker wave).
The binder is user code compiled into the same kernel: the test toy binder compiles to 90
registers/thread (20 blocks/SM, 1120 blocks). The capture therefore cannot bound the wave for an
arbitrary binder, so `startPool()` keeps the 128-register floor on top of the captured value
(896 blocks) unless `PoolConfig::registerHint` names a measured binder value. The open item is a
capture that includes a representative binder, or a per-instantiation occupancy check.

## Shuffle-based broadcast does not reconverge a split warp
The pinned alpaka implements `alpaka::onAcc::warp::shfl` as
`__shfl_sync(__activemask(), value, srcLane, width)`. `__activemask()` reports the lanes that are
converged *at that call site*; it does not reconverge the warp. After a leader-only region that the
compiler does not bracket with a reconvergence point, the leader and the remaining lanes stay in two
convergence groups and every later shuffle returns the caller's own value. Measured on the A30 with
a 32-thread block: immediately after `PacketPool::acquire` (whose leader-only CAS retry loop is the
trigger) lane 0 reports `__activemask() == 0x1` and lanes 1..31 report `0xfffffffe`; a following
`shfl` of a leader value delivers nothing to the other lanes, while the same shuffle before the
`acquire` works. That single defect explained the `imported < accepted` stall, the `errorWord 0x240`
abort, the all-wrong sink slots, and probe 3.

The pool no longer depends on a shuffle for any decision: the ready-ring claim result, the
packet-pool claim/release/configure result, and the importer/startup control values travel through
`graph::WarpHandoff` (one block-shared slot, written by lane 0, followed by `syncBlockThreads`, read
by every lane). The remaining shuffle users are functors and the chunk-assembly code, which call a
broadcast only in uniform control flow. The gate for reintroducing shuffles into a leader-then-
collective path is a broadcast primitive that passes an explicit full mask plus a warp convergence
point; the pinned alpaka does not provide one.

## The initializer role must stay unique
The pool used to abort at 256 workers and above (a power-of-two threshold) and at the full 896-block
wave, with `errorWord 0x240` and sink accounting `correct=0 missing=N`. The initializer role was
inferred from a block-index comparison, and on the full wave every block evaluated it as "I am the
initializer", so all of them ran `ReadyRing::initialize()` and `PacketPool::configure()`. A late
`configure()` rewrites each slot record to its initial free-list value, so a slot handed out by an
earlier `acquire()` read back as `{next slot, generation 0}` and the executing block dropped the job.
The wait path was never exercised by those blocks and formed a host-only `steady_clock` deadline in
device code.

Both are closed: the role is decided from the block index the kernel derives from the accelerator
and passes by value, the wait path no longer forms a host clock read on the device, and
`QuiescenceSnapshot::initializerClaims` is asserted to be one on every ladder rung and every point of
the 128..896 worker sweep. That counter is the regression gate: a second initializer means the pool
was re-configured underneath live owners.

## Multi-warp-per-block is refused by design
`WarpBarrier::barrier()` maps to a block-wide barrier only when the block holds exactly one warp.
For more than one warp per block it refuses (tripping a runtime flag the kernel must assert on),
because a block-wide barrier would also order the other warps in the block, a different,
co-residency-dependent schedule, and a bare vote orders nothing. The missing prerequisite is a
validated warp-local barrier. `computeLaunchGeometry()` throws `std::invalid_argument` for a
multi-warp request unless the build defines `GRAPH_ENABLE_MULTI_WARP_BLOCKS`. The measured headroom
on the A30 is 3584 warps (64 warps per SM, 56 SMs) against the 1792 workers the one-warp-per-block
geometry uses.

## System-scope RMW on mapped memory is not contract-safe
`atomicAdd_system` on a mapped pinned allocation measured working on the A30: 256000 of 256000
unique results in the verification round-trip, even though `cudaDevAttrHostNativeAtomicSupported`
reported 0. `tests/unit/test_system_atomic.cpp` records that the A30 reports
`hostNativeAtomicSupported` as 0 and that the probe measured lost updates when host and device RMW
the same mapped word concurrently. The A30 reports `concurrentManagedAccess=1`, so the eligible
allocation for a concurrent RMW is managed memory, not mapped pinned memory. The ordered
acquire/release transport is therefore the default (`IngressTransport::OrderedSystemAtomic`). The
raw RMW is reachable only through the explicit opt-in `atomicAddSystemNative` after a live
verification, and it costs about 597 ns per operation on the A30, so it is unsuitable for per-job
publication. No pool path uses it.

## Co-residency is arithmetic plus an empirical rendezvous
Occupancy arithmetic over queried or measured limits plus the startup rendezvous test establish
co-residency; it is not a hardware guarantee: alpaka has no occupancy API and no cooperative launch,
and an ordinary launch of an occupancy-sized grid does not promise every block is resident at the
same time. `tests/integration/test_residency.cpp` is the gate: the host requires `completed ==
workerCount`, not merely `arrived == workerCount`, because sequential waves leave earlier workers
below the expected count.

## The chunk-assembly publication guard cannot detect a reverted barrier
`tests/integration/test_chunk_publication_gpu.cpp` runs 20 launches of 1000 iterations (20000
published chunks, 640000 elements read from a different warp) and checks for stale elements; it
observed 0 failures in 250+ patched launches on the measured GPU. A device-scope release fence
orders only the executing thread's own prior writes, so the guard cannot detect a regression of the
lane ordering to a vote or a bare leader fence on this GPU: the reader warp may observe the payload
through L2 even when the lane stores are not ordered. The barrier rests on probe evidence
(`probes/warp_barrier_probe.cu`, `worstVoteInvisible=30`, `worstBarrierInvisible=0`); this is a
test-coverage gap.

## `hal` sharing is `unknown`
The compute target reports `sharing=unknown`. The overlap and arrival timings in
`docs/hal-acceptance.md` (copy-queue overlap, residency wall time) are presence and overlap
evidence, not benchmarks. A benchmark run requires an `exclusive` system.

## Acceptance gates still pending at this commit
This section is the record for `52be5c72`; the current branch status is in the two sections above.
`docs/hal-acceptance.md` covers `52be5c72`; the CPU and CUDA CTest suites pass there (119/119 and
127/127). The following gates are not claimed:
- The particle-feedback example and its histogram. The target does not exist;
  `examples/particle_feedback/CMakeLists.txt` adds no target, so both example gates are `NOT-BUILT`.
- The end-to-end particle pipeline on the persistent pool and the exactly-once final-weight
  accounting, which arrive with the pool and example milestones. Worklist semantics are validated
  off-pool by `tests/integration/test_particle_semantics.cpp` against the committed fixtures.
- The GitLab runner path (`docs/ci.md`). No pipeline has run; the HAL jobs stay pending until the
  runners and the CUDA image are provisioned.

The green gates: CPU configure/build and CTest, CUDA configure/build and CTest, format (clang-format
20.1.8), REUSE (reuse 6.3.0). The CUDA suite includes the residency rendezvous, copy-queue overlap,
chunk-assembly publication guard, ready-ring cross-warp visibility, and particle-gpu kernel
comparison; none is skipped.
