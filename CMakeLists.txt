# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.25)

project(
    graph
    VERSION 0.1.0
    DESCRIPTION "Persistent warp-worker task graph on alpaka3"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GraphOptions)

# CUDA is only required by the CUDA configuration. The CPU configuration must
# configure and build without a CUDA toolkit being installed.
if(GRAPH_ENABLE_CUDA)
    enable_language(CUDA)
endif()

include(GraphDependencies)
include(GraphTargets)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(examples)

# Device-function-pointer ABI probe (CUDA-only target inside).
add_subdirectory(probes)
