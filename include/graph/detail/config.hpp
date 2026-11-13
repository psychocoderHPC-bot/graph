// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Keep in sync with project(graph VERSION ...) in the top-level CMakeLists.txt.
#define GRAPH_VERSION_MAJOR 0
#define GRAPH_VERSION_MINOR 1
#define GRAPH_VERSION_PATCH 0

// Backend feature macros are injected by the graph CMake target. The defaults
// keep the header usable when it is compiled outside of a CMake build.
#ifndef GRAPH_HAS_CUDA
#    define GRAPH_HAS_CUDA 0
#endif

#ifndef GRAPH_HAS_OPENMP
#    define GRAPH_HAS_OPENMP 0
#endif

// Multi-warp persistent blocks are an explicit opt-in. The missing prerequisite
// is a validated warp-local barrier: several warps of one block are scheduled
// independently, so a block-wide barrier cannot order their memory operations
// inside a persistent pool. The CUDA configuration may define this to 1; the
// default keeps the one-warp-per-block bring-up configuration.
#ifndef GRAPH_ENABLE_MULTI_WARP_BLOCKS
#    define GRAPH_ENABLE_MULTI_WARP_BLOCKS 0
#endif

// Register usage of the translation unit that instantiates the persistent
// kernel, captured from the build (`-Xptxas -v`). A value of 0 means the
// build-time capture was unavailable; the launch geometry then falls back to
// GRAPH_KERNEL_REGISTERS_FALLBACK. The capture is the only way to obtain the
// per-thread register count: alpaka and CUDA expose register file capacities,
// not the allocation of a compiled kernel.
#ifndef GRAPH_KERNEL_REGISTERS
#    define GRAPH_KERNEL_REGISTERS 0
#endif

// Conservative fallback when no register capture exists. 128 registers per
// thread is the highest A30 occupancy step measured in P1-PROTOCOL-PROBES and
// therefore never overstates the resident block count on that device. A CUDA
// build without capture uses this value and the rendezvous test still proves
// the resulting (smaller) wave empirically.
#ifndef GRAPH_KERNEL_REGISTERS_FALLBACK
#    define GRAPH_KERNEL_REGISTERS_FALLBACK 128
#endif
