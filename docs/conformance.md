<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# Conformance map: design requirements to tree evidence

Every row maps one atomic requirement from
`graph-persistent-warp-pool-design.md` (the project design, kept outside this
repository) to the artifact in this tree that proves it. Section numbers refer
to the design document.

Statuses:

- `verified`: the proving artifact exists in the tree at
  `a12bfbe7c87497c91aaa8503a7fc5545c8c3d169`, is discoverable, and covers the
  stated claim.
- `partial`: a proving artifact exists, but only part of the claim is covered
  or the coverage is best-effort.
- `pending`: no artifact in the tree covers the claim.

## 1. Scope and deliverables

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-1.1 | One persistent worker kernel per pool lifetime; pool reusable, idle, stops after close and drain | `pool gpu one launch, two function pointers, many workers, clean drain`; `pool core accepts a late submission to an already running idle pool`; `pool core closes with zero work` | verified | All three CTest names resolve in the CUDA and CPU suites (W16 run, 161/161 and 143/143, 2026-11-13). |
| REQ-1.2 | A GPU worker is one hardware warp; all lanes share one indirect call | `launch_geometry multi-warp blocks are refused without the compile-time flag`; `probe 1: warp broadcast of the JobDescriptor field set`; `probe 2: ready-ring claim and broadcast of the entry` | verified | Multi-warp geometry is refused by construction; the probes run the real pool kernel on CUDA. |
| REQ-1.3 | Same functor structs on CPU and GPU, no per-backend copies | `graph vocabulary instantiates for the CUDA accelerator`; `particle_gpu kernel matches the host cores for the same particles`; `toy_ops operations instantiate for the active accelerator` | verified | The particle-functor table is a header shared by both configurations. |
| REQ-1.4 | `examples/particle_feedback/` with N, Y, seed; runs to termination; prints exact final-weight histogram | `examples/particle_feedback/cli.hpp`, `tests/unit/test_cli.cpp` | pending | The CLI header and its unit tests exist; no executable target and no CTest entry exist at this commit. `examples/particle_feedback/CMakeLists.txt` adds no target. |
| REQ-1.5 | Catch2 unit and integration tests registered with CTest; example executable registered too | `tests/CMakeLists.txt` (`catch_discover_tests` for both suites) | partial | Both Catch2 suites are registered. The example registration is absent because the example target is absent (REQ-1.4). |
| REQ-1.6 | alpaka pinned to the recorded revision and reproducible | `cmake/GraphDependencies.cmake`, `CMakeLists.txt`, `docs/build.md` | verified | Pin `4780dc786a2eb1a5452a1c66d94a8b442d411ab8` is the same revision named in the design. |
| REQ-1.7 | Release gate: Catch2 and CTest pass on CPU and on an NVIDIA GPU on `hal` | `docs/hal-acceptance.md` and the W16 artifact record for commit `a12bfbe7c87497c91aaa8503a7fc5545c8c3d169` (CPU 143/143, CUDA 161/161, 2026-11-13) | verified | The gate covers the test suites. The example portion of the gate is pending with REQ-1.4. |

## 3. Host API, registration, and device dispatch

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-3.1 | Nonblocking `tryEnqueue` with accepted-count result; blocking convenience wrapper; closed status after close | `detached pool accepts, closes, and rejects later submissions`; `ingress enqueueBlocking waits for the consumer and then accepts`; `ingress enqueueBlocking on a closed ring returns closed without waiting`; `ingress enqueueBlocking times out on a full ring without mutating it` | verified | |
| REQ-3.2 | A node registers named functor structs, input/output schemas, and routing; edges checked at build time | `graph_model accepts and compiles a valid linear graph`; `graph_model rejects a mismatched element type naming both ports`; `graph_model rejects a rank mismatch`; `graph_model rejects an extent mismatch`; `graph_model rejects a pitch that is not an element-size multiple`; `graph_model rejects a duplicate edge`; `graph_model rejects a missing source or sink and names the node`; `graph element schemas describe and check edges` | verified | |
| REQ-3.3 | Device jobs carry a real function pointer with one common ABI per accelerator instantiation; table formed in device code | `probe 1: warp broadcast of the JobDescriptor field set`; `pool gpu one launch, two function pointers, many workers, clean drain`; `probes/abi_probe.cu` (build-time check, `cmake/GraphRegisterProbe.cu` capture) | verified | P1 protocol probe p3 recorded the indirect-call cost and the host-table failure mode. |
| REQ-3.4 | Test at least two different function pointers with consecutive jobs in one pool | `pool gpu one launch, two function pointers, many workers, clean drain`; `pool core serves many jobs on multiple CPU workers with one launch` | verified | ToySource and ToySink run in the same resident kernel. |
| REQ-3.5 | One lane claims a job; the handle is broadcast uniformly; the packet slot stays valid through collective execution | `probe 2: ready-ring claim and broadcast of the entry`; `probe 3: packet-pool acquire broadcasts the handle`; `tests/unit/test_stream_id_unity.cpp` | verified | The broadcast travels through `graph::WarpHandoff` after a block sync (see `docs/known-limitations.md` on why shuffles are not used for this). |
| REQ-3.6 | Lanes visit `lane, lane+warpWidth, ...`; partial packet masks element access; CPU width-one fallback | `graph warp group implements the width-one fallback on the host`; `graph packet views and masked spans stay in bounds`; `particle_gpu kernel matches the host cores for the same particles` | verified | |
| REQ-3.7 | Concurrent CPU configuration for pool tests; a serial configuration for deterministic tests | `pool core serves many jobs on multiple CPU workers with one launch` (concurrent workers) | partial | The multi-worker CPU pool test exists. There is no separate named serial pool test at this commit. |

## 4. Launch and worker residency

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-4.1 | Launch size computed from the actual persistent entry point (registers, block size, shared memory) | `cmake/GraphRegisters.cmake` (build-time `-Xptxas -v` capture), `cmake/GraphRegisterProbe.cu`, `launch_geometry register-limited warps follow the measured A30 sub-partition rule` | partial | The capture probes `RendezvousKernel`/`PoolKernel<NullBinder>`, not a user binder. `docs/known-limitations.md` and `docs/architecture.md` record the 90-register toy binder, 20 blocks/SM, 1120 blocks, and the 128-register floor (896 blocks) that `startPool` keeps until `PoolConfig::registerHint` names a measured value. |
| REQ-4.2 | Most-resident-workers candidate is selected; geometry values are recorded | `launch_geometry geometry picks the most resident workers`; `launch_geometry resident blocks reproduce the measured A30 boundaries`; `launch_geometry thread, warp, block and shared-memory limits all bind`; `residency persistent workers rendezvous on CUDA within the deadlined single wave` (records blocks=1792, workers=1792, smCount=56, residentsPerSm=32, registersPerThread=48) | verified | The A30 points 48/40, 64/32, 82/20, 96/20, 112/16, 128/16 are checked against the P1 sweep. |
| REQ-4.3 | All configured workers enter the pool together, proven by a startup rendezvous with a deadline | `residency persistent workers rendezvous on CUDA within the deadlined single wave` (requires `completed == workerCount`, single-wave check) | verified | 1792/1792 completed in the W16 run. |
| REQ-4.4 | Polling backs off when idle; a bounded import keeps the importer in the ready queue | `pool gpu resumes from idle and reports the resume latency`; `PoolConfig::idleNanosleep` (default 128 ns, matching the P5 wake-up measurement) | verified | The P1 protocol probe p5 measured the backoff cost. |

## 5. Queues and system-scope atomic adapter

### 5.1 Host ingress

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-5.1.1 | SPSC ring; CPU producer owns `published`, GPU importer owns `consumed` | `ingress concurrent host producer and host importer keep order and exactly-once`; `ingress round trip preserves order, exactly once, and close last` | verified | |
| REQ-5.1.2 | Unsigned modular counters, capacity below half range, aligned and separate counter fields | `ingress setCountersForTest rejects an out-of-range occupancy`; `ingress peak occupancy and debug counters` | verified | |
| REQ-5.1.3 | Rollover validated with counters initialized near the maximum | `ingress counters initialised near the counter maximum roll over`; `ingress wraparound preserves order across many batches` | verified | |
| REQ-5.1.4 | Backpressure: a full ring reports without loss or hang | `ingress accepts exactly capacity then reports ring_full`; `ingress backpressure then drain then accept`; `ingress rejects batches larger than the capacity without a hang`; `ingress partial batch larger than free space` | verified | |
| REQ-5.1.5 | No overwrite before acknowledgement | `ingress does not overwrite unacknowledged slots` | verified | |
| REQ-5.1.6 | Ordered end-of-input marker; no host job accepted after it | `ingress close marker is ordered and blocks later submissions`; `ingress close marker is only accepted when it fits` | verified | |
| REQ-5.1.7 | A bounded scheduling opportunity for ingress while ready work exists | `pool core serves many jobs on multiple CPU workers with one launch`; `PoolConfig::maxImportBatch`/`importPerIteration` | partial | The import batch limits exist and the CPU pool test exercises interleaved import and execution, but there is no dedicated test asserting the bounded-opportunity rule under contention. |

### 5.2 System-scope atomic adapter

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-5.2.1 | Return value and add-zero behavior | `system_atomic: add-zero returns the old value and keeps memory unchanged` | verified | |
| REQ-5.2.2 | Concurrent CPU/GPU RMW on an eligible allocation | `system_atomic: concurrent CPU and device adds on an eligible allocation` | verified | On the A30 the eligible allocation is managed memory (`hostNativeAtomicSupported=0`, `concurrentManagedAccess=1`). The A30 measured fact for mapped pinned memory (lost updates under concurrent host/device RMW) is recorded in `docs/known-limitations.md` and in the P1 probe `results/p1_c.txt`. |
| REQ-5.2.3 | CPU path through `std::atomic_ref` | `system_atomic: CPU backend and host std::atomic_ref path` | verified | |
| REQ-5.2.4 | Capability reporting and transport selection; an unverified RMW is rejected | `system_atomic: capability probe, live verification and selected transport`; `system_atomic: transport selection rejects an unverified RMW` | verified | A30 record from the W16 run: `deviceSystemScopeRmwNative=0, hostNativeAtomicSupported=0, concurrentManagedAccess=1, mappedPinnedAllocation=1`, selected transport `OrderedSystemAtomic`. |
| REQ-5.2.5 | Both publication directions with descriptor sequence numbers and varied payloads | `system_atomic: host publishes descriptors the device consumes`; `system_atomic: device publishes completions the host consumes` | verified | |
| REQ-5.2.6 | The live-ingress integration path passes on the required GPU through a supported transport | `ingress round trip preserves order, exactly once, and close last` (CUDA build, W16 run) | verified | The round trip runs over the real ingress ring on the A30 through `OrderedSystemAtomic`. |

### 5.3 Device-ready ring

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-5.3.1 | Bounded MPMC ring with try-locked metadata transitions; bounded backoff on failure | `ready ring: exact-once claim over 10000 items, capacity 8`; `ready ring: empty and full, capacity 1`; `ready ring: empty and full, capacity 2`; `ready ring: single warp enqueue/dequeue, tiny capacity 1`; `ready ring: single warp enqueue/dequeue, tiny capacity 2` | verified | |
| REQ-5.3.2 | Sequence-numbered reuse; rollover | `ready ring: sequence rollover, capacity 2, 2048 items`; `ready ring: wrap-safe arithmetic, capacity 4, 4096 items` | verified | |
| REQ-5.3.3 | Entry copied out before slot reclaim; release/acquire publication between producer warp and consumer warp | `ready ring: cross-warp visibility, 8 blocks x 32 lanes, 200 items each`; `ready ring: cross-warp visibility, 16 blocks x 32 lanes, 128 items each` | verified | |
| REQ-5.3.4 | The same state transitions hold on a concurrent CPU executor | `pool core serves many jobs on multiple CPU workers with one launch` | verified | The CPU pool exercises the ready ring under concurrent workers. |

## 6. Packet storage and bin output

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-6.1 | Fixed-capacity packet pools with explicit slot ownership; no double hand-out | `packet pool: lifecycle, capacity 1`; `packet pool: lifecycle, capacity accounting and use-after-release, capacity 8`; `packet pool: no double hand-out, small pool, tight contention`; `packet pool: no double hand-out under multi-warp contention` | verified | |
| REQ-6.2 | Packet returned only after the last read; materialized routes release after all copies | `packet pool: lifecycle, capacity accounting and use-after-release, capacity 8`; `toy_ops pass-through copies unchanged to the routed bin` | verified | |
| REQ-6.3 | Two physical chunks A/B of capacity Y plus a logical counter per destination stream | `chunk_assembly all particles target one bin`; `chunk_assembly both bins receive data`; `chunk_assembly counter four plus input 128 yields full packet and four carried` | verified | The c+Y rotation is the A/B swap with counter decrement; the 4+128 case pins the carried-entry count. |
| REQ-6.4 | `0 <= c < Y` and an empty second chunk before an append; `Y=1` | `chunk_assembly capacity one rotates every append` | verified | |
| REQ-6.5 | Maximum occupancy after one append is `2Y-1`, so one rotation suffices | `chunk_assembly two Y minus one bound over randomized appends` | verified | |
| REQ-6.6 | Publish A when `c+k >= Y`, including the exactly-full case | `chunk_assembly exactly full rotates including the full chunk` | verified | |
| REQ-6.7 | Partial publication follows the same rule with the actual count | `chunk_assembly partial input stays in A`; `chunk_assembly flush partial publishes the real count` | verified | |
| REQ-6.8 | One writer warp owns a destination pair; competing writers get disjoint ranges | `chunk_assembly competing writers get disjoint ranges`; `chunk_assembly competing writers on CUDA warps get disjoint ranges`; `chunk_assembly multi-destination ranks split one CUDA warp`; `chunk_assembly destination group split for a full warp` | verified | |
| REQ-6.9 | A failed reservation leaves input and counters unchanged; stale commits fail without partial mutation | `chunk_assembly reservation failure leaves state unchanged`; `chunk_assembly stale commit fails without partial mutation` | verified | |
| REQ-6.10 | Non-warp-multiple chunk capacity | `chunk_assembly non-warp-multiple capacity` | verified | |
| REQ-6.11 | Safe rotation under repeated appends | `chunk_assembly safe rotation under repeated appends` | verified | |
| REQ-6.12 | Capacity budget `P + 2B + W*R + S` | `packet pool: budget formula requiredSlots(P,B,W,R,S)`; `packet pool: undersized budget fails startup loudly`; `pool config validation accepts a legal budget and names an undersized one` | verified | |
| REQ-6.13 | Deferred jobs retain their input and stay schedulable | `pool core re-queues a source continuation without flooding` | verified | |
| REQ-6.14 | Capacity and peak-usage exposure; a small-capacity progress test | `pool core survives full-capacity pressure on a deliberately small pool`; `ingress peak occupancy and debug counters`; `pool gpu full-capacity pressure does not deadlock` | verified | |

## 7. Completion with feedback

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-7.1 | Quiescence: completion only when ingress is closed and drained and all job, reservation, source, and partial state is empty | `pool gpu one launch, two function pointers, many workers, clean drain` (checks outstanding, pendingSources, queued, active all zero with stopReason drain) | verified | The counters come from `QuiescenceSnapshot`. |
| REQ-7.2 | Drain of nonempty partial buffers repeated until no work remains | `chunk_assembly flush partial publishes the real count`; `pool core closes with zero work`; `pool gpu one launch, two function pointers, many workers, clean drain` | verified | |
| REQ-7.3 | Stop flag only after one consistent observation; warp-uniform exit | `detached pool distinguishes drained from aborted` (SECTION "ordered close reports a drain"); `pool core closes with zero work` | verified | |
| REQ-7.4 | Abort is distinct from drain; a failing test cannot leave an unbounded launch | `pool core abort stops a pool with pending work and reports an abort`; `detached pool distinguishes drained from aborted` (SECTION "abort reports an abort, not a drain"); `toy_ops abort sets the diagnostic abort state` | verified | |
| REQ-7.5 | State and progress counters exposed for diagnostics | `QuiescenceSnapshot` counters printed by the ladder and pool tests (`[pool-gpu]` and `[ladder]` lines in the W16 artifact) | verified | |
| REQ-7.6 | Finite timeouts on CTest cases | `tests/CMakeLists.txt` (`catch_discover_tests ... PROPERTIES TIMEOUT 300`) | verified | |
| REQ-7.7 | Errors routed to the host as explicit failures; invalid request ranges and overflow covered | `toy_ops source reports request overflow and out-of-range ids`; `ingress reports invalid input as an error and mutates nothing`; `ingress import error is reported and leaves the ring usable`; `ingress import error acknowledges nothing and does not poison the ring` | verified | GPU launch failure propagation is covered by the error-word checks in the pool tests. |
| REQ-7.8 | Cycles with a deterministic scripted number of revisits | `toy_ops revisit increments visit exactly revisitCount times then sinks`; `graph_model accepts a declared feedback cycle`; `graph_model rejects an undeclared cycle`; `graph_model rejects a self edge without the feedback flag` | verified | |

## 8. Required particle feedback example

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-8.1 | CLI `--backend cpu|cuda --particles N --chunk-size Y --seed S`; Y=128 default | `cli accepts every flag`; `cli accepts the --flag=value form`; `cli defaults match the documented values` (Y=128, N=4096, seed=42); `cli rejects an unknown backend` | verified | CLI level. |
| REQ-8.2 | A chunk size that is not a multiple of the warp width is accepted | `cli accepts every flag` (chunk-size 17); `graph_model particle feedback graph validates for supported chunk sizes` (chunk 100); `chunk_assembly non-warp-multiple capacity` | verified | CLI and graph validation. The executable run at chunk 100 is pending with REQ-1.4. |
| REQ-8.3 | `N=0` produces an empty histogram | `cli accepts N=0`; `histogram empty input produces an empty result` | verified | CLI and histogram level. |
| REQ-8.4 | A documented default particle count and seed | `cli defaults match the documented values`; `examples/particle_feedback/cli.hpp` (defaults table) | verified | N=4096, seed=42. |
| REQ-8.5 | Reproducible integer RNG; no floats in decisions; uniform unsigned comparison at the 0.5 threshold | `rng key packing and documented constants`; `rng documented fixture draws`; `rng chi-square over 2^8 buckets for 10^7 draws`; `rng per-bit balance over 2^20 draws`; `rng lag-1 autocorrelation of the destroy bit`; `rng flag1 sequence has no period below 2^20`; `particle_semantics exact 0.5 boundary by construction` | verified | |
| REQ-8.6 | Functors: GenerateParticles, FilterByProperty, ProcessA, ProcessB, DestroyParticles | `tests/reference/oracle.cpp`, `tests/reference/oracle.hpp`, `tests/integration/test_particle_semantics.cpp`, `tests/integration/test_particle_gpu_kernel.cpp`, `include/graph/particle_functors.hpp` | verified | Host semantics against the scalar oracle, and CUDA lanes matched against the host cores. |
| REQ-8.7 | Property, ID, weight, and RNG position preserved across feedback rounds | `particle_semantics property preserved and visit incremented`; `particle_semantics worklist matches the committed scalar reference` | verified | |
| REQ-8.8 | Histogram: one final-weight slot per source ID, sink-only writes, exact `weight,count` rows, zero-weight row when present, counts sum to N, duplicates and missing terminations detected | `histogram counts must sum to N`; `histogram zero-weight row is present when a zero weight occurs`; `histogram a duplicate sink write is detected`; `histogram a missing sink termination is detected`; `histogram an unwritten sentinel slot is reported`; `histogram sink-visit counters must match the source count`; `histogram sparse and large weights need no dense weight-dependent allocation`; `histogram negative final weights are rejected`; `histogram single weight yields one row`; `histogram duplicate weights are merged and unsorted input renders deterministically`; `particle_semantics destroy writes the final weight exactly once` | verified | Host level. |
| REQ-8.9 | End-to-end: every final weight and the exact histogram match the scalar reference for zero, one, below-chunk, exact-chunk, and multi-chunk inputs with multiple seeds | `tests/reference/test-vectors.md` (N=0, 1, 7, 128, 1000, 4096 seeds 42 and 7, 65536 seeds 42 and 7); `tests/reference/oracle.cpp` (standalone) | partial | The fixtures and the oracle exist. No test runs the example pipeline against them, because the executable is not wired (REQ-1.4). |
| REQ-8.10 | CPU and NVIDIA results compared across chunk sizes | `particle_gpu kernel matches the host cores for the same particles` | partial | The functor-level CUDA-vs-CPU comparison exists. The end-to-end example comparison across chunk sizes is pending with REQ-1.4. |

## 9. Repository layout and verification

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-9.1 | Repository layout: CMake, README, include/graph, examples/particle_feedback, tests/unit, tests/integration, docs | tree at this commit | verified | |
| REQ-9.2 | Catch2 + `catch_discover_tests`; the example registered via `add_test` with backend arguments, result validation, and timeouts | `tests/CMakeLists.txt` | partial | Catch2 registration complete; example `add_test` absent with REQ-1.4. |
| REQ-9.3 | CPU build and CUDA build, same functor definitions in both | `docs/hal-acceptance.md` (both configurations built and tested on `a12bfbe7`) | verified | |
| REQ-9.4 | Test matrix: system atomic adapter | the `system_atomic:` cases in section 5.2 | verified | |
| REQ-9.5 | Test matrix: host ingress (empty/full, batches, backpressure, late submission, wraparound, no overwrite, ordered close, rejected submission after close) | the ingress cases in section 5.1 plus `ingress late submission works while the pool is idle` | verified | |
| REQ-9.6 | Test matrix: ready ring and packet pool | the ready-ring and packet-pool cases in sections 5.3 and 6.1 | verified | |
| REQ-9.7 | Test matrix: worker pool (many jobs, two pointers, startup, idle-resume, child enqueue, full-capacity pressure, zero-work close, shutdown, multiple worker IDs, one launch) | `pool core serves many jobs on multiple CPU workers with one launch`; `pool gpu one launch, two function pointers, many workers, clean drain`; `residency persistent workers rendezvous on CUDA within the deadlined single wave`; `pool gpu resumes from idle and reports the resume latency`; `pool gpu full-capacity pressure does not deadlock`; `pool core closes with zero work`; `pool core accepts a late submission to an already running idle pool` | verified | The `distinctWorkers >= 2` check and the `launchCount == 1` check are in the GPU case. |
| REQ-9.8 | Test matrix: chunk assembly | the `chunk_assembly` cases in section 6 | verified | |
| REQ-9.9 | Test matrix: graph drain (close while work is running, feedback during drain, cycles, long idle before close, no deadlock) | `pool core closes with zero work`; `pool gpu one launch, two function pointers, many workers, clean drain`; `pool gpu resumes from idle and reports the resume latency`; `toy_ops revisit increments visit exactly revisitCount times then sinks`; `pool core survives full-capacity pressure on a deliberately small pool` | partial | Close with pending work, cycles, and no-deadlock are covered. A long-idle-then-close case and feedback produced during drain are not separately covered on the GPU pool at this commit. |
| REQ-9.10 | Test matrix: particle functors (equality and 0.5, both increments, unchanged weight on destruction, property preservation, sink release) | `particle_semantics exact 0.5 boundary by construction`; `particle_semantics worklist matches the committed scalar reference`; `particle_semantics destroy writes the final weight exactly once`; `particle_semantics property preserved and visit incremented` | verified | |
| REQ-9.11 | Test matrix: end-to-end against the scalar reference | `tests/reference/test-vectors.md` | pending | No test consumes the fixtures end-to-end; see REQ-8.9. |
| REQ-9.12 | Test matrix: CTest example on CPU and NVIDIA with an exact expected histogram for a fixed seed | absent | pending | `examples/particle_feedback/CMakeLists.txt` adds no target. |
| REQ-9.13 | HAL record: commit, dependency revision, compiler, CUDA toolkit/driver, CPU/GPU model, test counts, commands, occupancy, transport, result summary | `docs/hal-acceptance.md`; W16 artifact record for `a12bfbe7` (hal container `c692209a108c`, A30 cc 8.0, driver 615.71.09, CUDA 13.4.92, g++ 13.3.0, CMake 3.28.3, CPU 143/143, CUDA 161/161, transport `OrderedSystemAtomic`) | verified | |
| REQ-9.14 | Automated CPU checks for changes | `.gitlab-ci.yml` (`test:cpu`) | partial | The job is committed. No pipeline has run: the GitLab project connection and runners are not provisioned (`docs/ci.md`). |
| REQ-9.15 | Automated HAL CPU and NVIDIA checks | `.gitlab-ci.yml` (`test:hal:cpu`, `test:hal:nvidia`); `docs/ci.md` | pending | The jobs are committed with placeholder runner tags. The hardware gate is visible as pending, as the design requires. |

## 11. Conventions, CI, REUSE

| ID | Requirement | Proving artifact | Status | Note |
| --- | --- | --- | --- | --- |
| REQ-11.1 | Pinned `.clang-format` from the upstream revision, formatter version pinned in CI | `.clang-format`; `scripts/check-format.sh`; `.gitlab-ci.yml` (`format`, clang-format 20.1.8) | verified | |
| REQ-11.2 | CI fails on formatting differences against tracked project files | `.gitlab-ci.yml` (`format`); `scripts/check-format.sh` | verified | Last run on `a12bfbe7`: 65 tracked files, no differences (W16 artifact). |
| REQ-11.3 | REUSE compliance: license text, SPDX headers, `REUSE.toml`, `reuse lint` in CI | `REUSE.toml`, `LICENSES/MPL-2.0.txt`, `LICENSE`, `scripts/reuse-lint.sh`, `.gitlab-ci.yml` (`reuse`) | verified | Last run on `a12bfbe7`: REUSE 3.3 compliant (W16 artifact). |
| REQ-11.4 | GitLab CI workflow committed, trigger mechanism documented, pending state visible | `.gitlab-ci.yml`, `docs/ci.md` | partial | The workflow is committed and the operator actions are listed in the file header. The trigger mechanism is not connected yet, so no pipeline result exists. |
| REQ-11.5 | CI artifacts: CTest logs, JUnit XML, HAL diagnostics, finite timeouts | `.gitlab-ci.yml` (artifacts and timeout blocks) | verified | |

## How this was derived

1. The requirement list comes from the sections named in the task, read
   atomically from `graph-persistent-warp-pool-design.md`.
2. The test inventory was extracted from every `TEST_CASE` in
   `tests/unit/*.cpp` and `tests/integration/*.cpp` at
   `a12bfbe7c87497c91aaa8503a7fc5545c8c3d169` (161 names). Each name below was
   checked against the `ctest` discovery output of the W16 HAL run on that
   exact commit (CPU 143, CUDA 161, zero skipped). A requirement is `verified`
   only where a test with that name exists, is discovered by CTest, and its
   assertions cover the stated claim.
3. Measured facts were taken from the W16 artifact record
   (`/tmp/W16-artifacts`, commit `a12bfbe7`, run 2026-11-13) and from the P1
   protocol probes in
   `/tmp/alpaka-orchestration/graph-warp-pool-20261112/artifacts/p1-probes-hal/results/`.
4. Where the executable example is absent, the row is marked `pending` or
   `partial` and says so. No requirement was upgraded on the basis of a
   planned change.

## Known gaps

- The example executable and its CTest registration do not exist
  (REQ-1.4, REQ-8.9, REQ-8.10, REQ-9.11, REQ-9.12). The CLI header, the
  fixtures in `tests/reference/test-vectors.md`, and the standalone oracle are
  in place; the wiring node adds the target.
- The register capture cannot see a user binder; the 896-block wave comes from
  the 128-register floor, not from a measurement of the pool kernel with a
  real binder (REQ-4.1).
- The chunk-publication barrier has no automated regression guard that would
  fail on this GPU if the `WarpBarrier` were reverted to a fence plus
  `warp.all`; `tests/integration/test_chunk_publication_gpu.cpp` is a
  best-effort probe, as `docs/known-limitations.md` records.
- The GitLab pipeline has never run; the HAL hardware gate is unfulfilled by
  CI and is covered by the manual W16 run instead (REQ-9.14, REQ-9.15,
  REQ-11.4).
- The graph-drain matrix lacks a dedicated long-idle-then-close case and a
  feedback-during-drain case on the GPU pool (REQ-9.9).
- The `2Y-1` bound is tested by randomized appends, not by a constructed
  worst case.
- There is no separate named serial-configuration pool test (REQ-3.7).
