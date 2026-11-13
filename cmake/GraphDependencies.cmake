# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

include(FetchContent)

# Drive alpaka's own configuration from the graph options before alpaka is
# configured, whether it is fetched or added from a local checkout.
set(alpaka_DEP_CUDA ${GRAPH_ENABLE_CUDA} CACHE BOOL "Enable the CUDA dependency" FORCE)
if(GRAPH_ENABLE_CUDA)
    # The tests and the particle kernel are compiled against the CUDA API
    # directly (alpaka::exec::gpuCuda), so the execution backend must be
    # enabled even though alpaka itself is a header-only interface here.
    set(alpaka_EXEC_GPU_CUDA ON CACHE BOOL "Enable the CUDA execution backend" FORCE)
endif()
set(alpaka_DEP_OMP ${GRAPH_ENABLE_OPENMP} CACHE BOOL "Enable the OpenMP dependency" FORCE)
set(alpaka_DEP_TBB OFF CACHE BOOL "Enable the oneTBB dependency" FORCE)
set(alpaka_DEP_HIP OFF CACHE BOOL "Enable the HIP dependency" FORCE)
set(alpaka_DEP_ONEAPI OFF CACHE BOOL "Enable the oneAPI dependency" FORCE)
if(GRAPH_ENABLE_CUDA)
    # Later milestones pass device function pointers through the worker pool.
    set(alpaka_RELOCATABLE_DEVICE_CODE ON CACHE STRING "Enable relocatable device code" FORCE)
endif()

# The bootstrap project never builds alpaka's own tests, examples, or tools.
set(alpaka_TESTS OFF CACHE BOOL "Disable alpaka tests" FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "Disable alpaka examples" FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "Disable alpaka benchmarks" FORCE)
set(alpaka_DOCS OFF CACHE BOOL "Disable alpaka documentation snippets" FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "Disable alpaka header checks" FORCE)
set(alpaka_CLANG_TIDY OFF CACHE BOOL "Disable alpaka clang-tidy checks" FORCE)

if(GRAPH_ALPAKA_SOURCE_DIR)
    if(NOT EXISTS "${GRAPH_ALPAKA_SOURCE_DIR}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "GRAPH_ALPAKA_SOURCE_DIR='${GRAPH_ALPAKA_SOURCE_DIR}' is not an alpaka checkout"
        )
    endif()
    set(alpaka_SOURCE_DIR "${GRAPH_ALPAKA_SOURCE_DIR}" CACHE PATH "alpaka source directory" FORCE)
    add_subdirectory("${GRAPH_ALPAKA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/alpaka-build")
else()
    FetchContent_Declare(
        alpaka
        GIT_REPOSITORY https://github.com/alpaka-group/alpaka
        GIT_TAG ${GRAPH_ALPAKA_GIT_TAG}
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(alpaka)
endif()

# Catch2 revision used by the pinned alpaka test tree.
set(CATCH_CONFIG_NO_POSIX_SIGNALS ON CACHE BOOL "Avoid POSIX signal handlers in Catch2" FORCE)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.15.0
)
FetchContent_MakeAvailable(Catch2)
include(Catch)
