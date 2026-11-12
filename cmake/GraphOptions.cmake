# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

option(GRAPH_ENABLE_CUDA "Enable the NVIDIA CUDA backend" OFF)

find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    set(_graph_openmp_default ON)
else()
    set(_graph_openmp_default OFF)
endif()
option(GRAPH_ENABLE_OPENMP "Enable the OpenMP CPU executor if OpenMP is available" ${_graph_openmp_default})
if(GRAPH_ENABLE_OPENMP AND NOT OpenMP_CXX_FOUND)
    message(WARNING "GRAPH_ENABLE_OPENMP is ON but no OpenMP C++ support was found; disabling OpenMP.")
    set(GRAPH_ENABLE_OPENMP OFF CACHE BOOL "Enable the OpenMP CPU executor if OpenMP is available" FORCE)
endif()

set(
    GRAPH_ALPAKA_SOURCE_DIR
    ""
    CACHE PATH "Use an existing alpaka checkout instead of fetching the pinned revision"
)
set(
    GRAPH_ALPAKA_GIT_TAG
    "4780dc786a2eb1a5452a1c66d94a8b442d411ab8"
    CACHE STRING "alpaka Git revision to fetch (never a branch name)"
)

if(GRAPH_ENABLE_CUDA)
    include(GraphCuda)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        graph_detect_cuda_arch(_graph_cuda_arch)
        set(
            CMAKE_CUDA_ARCHITECTURES
            "${_graph_cuda_arch}"
            CACHE STRING "CUDA architectures to build for"
            FORCE
        )
        message(STATUS "graph: CMAKE_CUDA_ARCHITECTURES was not set, using ${_graph_cuda_arch}")
    endif()
endif()
