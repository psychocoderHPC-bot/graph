<!-- SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0 -->
# Architecture
`graph` runs a task graph on a persistent accelerator worker pool on alpaka3.
This document maps the split onto the headers, the data path onto its ordering
rules, and the launch geometry onto the measured numbers. Design:
[`design.md`](design.md). Hardware: [`hal-acceptance.md`](hal-acceptance.md).
Numerics: [`numerics.md`](numerics.md). Gaps:
[`known-limitations.md`](known-limitations.md).

## Host and device split

- The pool host object owns the queue, the mapped ingress ring, the packet
  pools, the `CopyQueue`, and the operation table (`include/graph/pool.hpp`).
- One persistent kernel per pool lifetime (`detail::PoolKernel` in
  `include/graph/detail/scheduler.hpp`), resident while idle, stopping only
  after an ordered close and a complete drain.
- One hardware warp is one worker; bring-up is one warp per block. All lanes
  execute the same indirect device-function call on one packet.
- Single resident wave: the grid is sized to the resident capacity and
  validated by `validateSingleWave()`.

## Modules

| File | Role |
| --- | --- |
| `include/graph/ingress_ring.hpp` | Mapped pinned host-to-device SPSC ring. |
| `include/graph/ready_ring.hpp` | Device-side bounded MP/MC work ring. |
| `include/graph/detail/rendezvous.hpp` | Startup rendezvous kernel. |
| `include/graph/detail/warp_barrier.hpp` | Warp-scoped barrier; refuses multi-warp. |
| `include/graph/detail/warp_group.hpp` | Lane index, leader test, shuffle broadcast. |
| `include/graph/detail/launch.hpp` | Geometry, `validateSingleWave()`. |
| `include/graph/detail/copy_queue.hpp` | Transfer queue, pinned staging. |
| `include/graph/system_atomic.hpp` | System-scope atomic adapter. |
| `include/graph/detail/atomic_cpu.hpp`, `atomic_cuda.hpp` | Backend atomics. |
| `include/graph/detail/capabilities.hpp` | Capability probe, RMW verification. |
| `include/graph/chunk_assembly.hpp` | Two chunks, one counter, one rotation. |
| `include/graph/detail/operation.hpp` | `OperationFn<Acc>`, `OperationTable`. |
| `include/graph/detail/packet.hpp` | `PacketHandle`, `PacketRecord`. |
| `include/graph/detail/job.hpp` | `JobDescriptor`, routing, request. |
| `include/graph/detail/views.hpp` | Typed, bounds-masked packet views. |
| `include/graph/detail/concepts.hpp` | Element-schema concepts. |
| `include/graph/detail/config.hpp` | Version, feature, register fallback. |
| `include/graph/detail/stream_id.hpp` | Strong `StreamId` type and stream-endpoint helpers. |
| `include/graph/packet_pool.hpp` | Packet slot pool, reuse generation. |
| `include/graph/graph_model.hpp` | Host-side graph builder. |
| `include/graph/particle_feedback_graph.hpp` | Example graph description. |
| `include/graph/rng.hpp` | Counter-based integer RNG. |
| `include/graph/particle_functors.hpp` | Example functors, pure cores. |
| `include/graph/histogram.hpp` | Host-side final-weight histogram. |
The host `Pool` object (`include/graph/pool.hpp`), the persistent pool kernel,
and the scheduler (`include/graph/detail/scheduler.hpp`) are implemented on the
`warp-pool` branch.

## Data path

```
host producer -> ingress ring (mapped pinned, release store)
   -> ready ring (device acquire, warp leader dequeue, block-shared ClaimResult)
   -> OperationFn<Acc>(acc, job, accessor)  [all lanes]
   -> chunk assembly (StreamReservation: reserve, write, commit)
   -> sink; input packet retired exactly once after the commit
```

1. Host to device, ingress: the host writes the batch, then publishes the new
   head with a system-scope release store (`graph::host::hostStoreRelease`).
   The device observes the head with a system-scope acquire load
   (`graph::systemLoadAcquire`) and reads the batch. The release store is the
   publication; nothing depends on `__threadfence_system`. No RMW handshake:
   concurrent host/device RMWs on mapped memory lose updates on the target GPU
   (measured on the A30), so the protocol is pure load/store ordering.
2. Device, ready ring: producer lanes write the entry, the caller runs
   `WarpBarrier::barrier()`, and the leader's `enqueue()` publishes the slot
   with a device-scope release fence before the counter advance. The consumer
   leader acquires under the lock, copies the entry out, and advances the
   claim counter. It writes the whole `ClaimResult` (descriptor, ring slot,
   claim stamp, operation id, packet handle, status, valid flag) into one
   block-shared slot. The ring then runs a block barrier and every lane reads
   the same bytes. A warp vote orders nothing: it observes predicates, it does
   not order one lane's stores against another lane's reads. The probe
   `probes/warp_barrier_probe.cu` contrasts vote and barrier; values in
   [`known-limitations.md`](known-limitations.md).
3. Leader to lanes: never by shuffle. The pinned alpaka implements `shfl` as
   `__shfl_sync(__activemask(), ...)`, and `__activemask()` reports the lanes
   that happen to be converged at the call site; it does not reconverge a warp
   that a leader-only region split. After `PacketPool::acquire`'s CAS retry
   loop the leader and the other 31 lanes stay in two convergence groups, so
   every later shuffle silently returns the caller's own value. Measured on the
   A30: the mask is `0x1` on lane 0 and `0xfffffffe` on the rest immediately
   after `acquire`. The pool therefore moves every claim/status value through
   `WarpHandoff` (`include/graph/detail/warp_handoff.hpp`): the leader writes a
   block-shared slot, all lanes run `syncBlockThreads`, then read. Shuffles
   remain available for code that never crosses a leader-only region, but no
   pool decision depends on one.
4. Functors: all lanes enter the same `OperationFn` call. Functors use only
   warp-local synchronization; no block-wide or grid-wide barriers, no
   allocation, no exceptions.

## Copies

The worker kernel occupies the worker queue for the whole run, so every
transfer goes through a dedicated `CopyQueue` created separately from the
worker queue. alpaka does not guarantee two `makeQueue()` calls return
independent backend queues; `tests/integration/test_copy_queue_overlap.cpp`
verifies that a copy on the copy queue completes while the worker queue is
still busy. Correctness never depends on the overlap; only latency does.
Pinned/mapped staging is mandatory: a copy into pageable host memory blocks
until the stream drains, destroying the overlap, so the copy helpers reject a
pageable buffer with `std::invalid_argument`. `orderAfter(event)` enqueues a
stream dependency; the copy queue has no implicit ordering against the worker
queue. Numbers: [`hal-acceptance.md`](hal-acceptance.md).

## Operation ABI

```cpp
template<typename T_Acc>
using OperationFn = void (*)(T_Acc const&, JobDescriptor const&, PacketAccessor const&);
```

- One function-pointer type per accelerator instantiation; all lanes invoke
  it. `JobDescriptor` is immutable and scheduler-owned; `PacketAccessor`
  borrows packet-pool storage valid for the call.
- `OperationTable` is a fixed-capacity registry. On CUDA it holds device
  function addresses populated in device code; the probe `probes/abi_probe.cu`
  stores an `OperationFn` pointer in a `__device__` array, reloads it, and
  calls it from device code.
- The pool kernel binds host-registered ids to device addresses in a startup
  phase and then rendezvous (`startupRendezvous` in
  `include/graph/detail/scheduler.hpp`). The `RendezvousKernel` remains the
  geometry co-residency test.

## Launch geometry

```
blocks  = visible_multiprocessors * resident_blocks_per_multiprocessor
workers = blocks * warps_per_block
```

`computeLaunchGeometry()` picks the legal candidate with the most resident
workers; the default policy is one warp per block. `validateSingleWave()`
re-derives the resident block count and fails a grid larger than one wave.
The per-thread register count of the persistent kernel cannot be queried
without a kernel handle, so the build captures it from the compiler
(`-Xptxas -v`, `cmake/GraphRegisters.cmake`) as `GRAPH_KERNEL_REGISTERS`. The
probe `cmake/GraphRegisterProbe.cu` instantiates `detail::PoolKernel` (the
pool's own persistent entry point) as well as `RendezvousKernel` and reports
the maximum, so the number comes from the real entry point. Without a capture,
`config.hpp` falls back to 128 registers per thread, the highest A30 occupancy
step measured in P1-PROTOCOL-PROBES.

Measured A30 result on the `warp-pool` branch: `PoolKernel<NullBinder>` compiles
to 48 registers/thread. A user binder is compiled into the same kernel and
raises the count: the test toy binder compiles to 90 registers/thread. At 48
the measured rule allows the 32-blocks/SM cap and therefore 1792 workers; at 90
it allows 20 blocks/SM and 1120 blocks. Because the capture cannot see the
binder, `startPool()` keeps the 128-register floor on top of the captured value
(896 blocks, 16 residents/SM), and a caller that has measured its binder passes
`PoolConfig::registerHint`. The rendezvous record in `docs/hal-acceptance.md`
measured `RendezvousKernel` (20 registers), not the pool kernel.

## Pool guarantees and their gates

The persistent pool makes the following claims. Each names the test that gates
it; `tests/integration/test_pool_core.cpp` runs the host backend and
`tests/integration/test_pool_gpu.cpp` runs one real CUDA launch.

- **Exactly-once sink writes.** Every accepted item reaches the sink exactly
  once; the sink increments `sinkVisits` and the test requires `sinkVisits == 1`
  for every id and zero mismatches. Gate: `checkResults` in
  `test_pool_gpu.cpp` and the toy-worklist checks in `test_pool_core.cpp`.
- **Distinct worker ids.** The kernel records the participating block indices
  in a worker bitmap; the drain test requires `distinctWorkers >= 2`, so more
  than one warp consumed work. Gate: `distinctWorkers` in the drain snapshot.
- **Launch once.** One kernel serves every batch of a lifetime;
  `launchCount == 1` after drain. Gate: the drain and idle-resume cases.
- **Idle then resume.** The kernel stays resident while idle and serves a late
  submission without a relaunch; the resume case measures the enqueue-to-retire
  latency and reports it. Gate: the idle-resume case.
- **Pressure.** A deliberately small pool (`maxReadyInFlight = 2`,
  `maxImportBatch = 4`) consumes a full-capacity batch without deadlock and ends
  in an ordered drain. Gate: the pressure case.
- **Abort.** A host abort stops a pool with pending work and reports
  `stoppedByAbort` with the `aborted` error bit. Gate: the abort cases in
  `test_pool_core.cpp`.

Verification status: the host-backend cases pass. The CUDA repeat of this branch
does not pass; see `docs/known-limitations.md`, which records the open defect and
the evidence that it is a pool free-list visibility problem, not the packet
handle.

## Co-residency

Co-residency is occupancy arithmetic over queried or measured limits plus the
startup rendezvous test (`tests/integration/test_residency.cpp`). It is not a
hardware guarantee: alpaka has no occupancy API and no cooperative launch, and
an ordinary launch of an occupancy-sized grid does not promise every block is
resident at once. The rendezvous test is the empirical gate: every warp leader
arrives once and increments `completed` only when it leaves the spin having
observed the full arrival count. Sequential waves cannot fake that, because a
later wave arrives after the earlier blocks gave up at the deadline. The host
requires `completed == workerCount`, not `arrived == workerCount`.
