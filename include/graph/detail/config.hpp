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
