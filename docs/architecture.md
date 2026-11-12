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
  pools, the `CopyQueue`, and the operation table. The `Pool` class is pending;
  it does not exist at this commit.
- One persistent kernel per pool lifetime, resident while idle, stopping only
  after an ordered close and a complete drain. The pool kernel is pending; the
  implemented parts are the rendezvous kernel and the launch-geometry code.
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
| `include/graph/packet_pool.hpp` | Packet slot pool, reuse generation. |
| `include/graph/graph_model.hpp` | Host-side graph builder. |
| `include/graph/particle_feedback_graph.hpp` | Example graph description. |
| `include/graph/rng.hpp` | Counter-based integer RNG. |
| `include/graph/particle_functors.hpp` | Example functors, pure cores. |
| `include/graph/histogram.hpp` | Host-side final-weight histogram. |
The host `Pool` object, the persistent pool kernel, and the scheduler are
pending; none exists as a header at this commit.

## Data path

```
host producer -> ingress ring (mapped pinned, release store)
   -> ready ring (device acquire, warp leader dequeue, WarpBarrier, shuffle broadcast)
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
   claim counter; the caller runs `WarpBarrier::barrier()` before any lane
   reads the broadcast entry. A warp vote orders nothing: it observes
   predicates, it does not order one lane's stores against another lane's
   reads. The probe `probes/warp_barrier_probe.cu` contrasts vote and barrier;
   values in [`known-limitations.md`](known-limitations.md).
3. Leader to lanes: the claimed `JobDescriptor` is broadcast field by field by
   warp shuffle, so every lane holds an identical copy.
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
  phase and then rendezvous. That rendezvous with deadline is not in the tree
  yet; pending. The `RendezvousKernel` is the geometry co-residency test.

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
(`-Xptxas -v`, `cmake/GraphRegisters.cmake`) as `GRAPH_KERNEL_REGISTERS`.
Without a capture, `config.hpp` falls back to 128 registers per thread, the
highest A30 occupancy step measured in P1-PROTOCOL-PROBES.
Measured A30 result (`docs/hal-acceptance.md`): 1792 blocks of 32 threads, 56
SMs, 32 residents per SM, 20 registers per thread captured, 1792 of 1792
workers arriving at the rendezvous, `launchCount` 1, `singleWave` 1. That
capture compiled `RendezvousKernel`, not the pool kernel. The pool kernel's
register capture is a separate pending item; until it exists the grid may be
conservative, or over-optimistic if the pool kernel uses more registers.

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
