# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

add_library(graph INTERFACE)
add_library(graph::graph ALIAS graph)

target_include_directories(
    graph
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(graph INTERFACE cxx_std_20)
target_link_libraries(graph INTERFACE alpaka::alpaka)
target_compile_definitions(
    graph
    INTERFACE GRAPH_HAS_CUDA=$<BOOL:${GRAPH_ENABLE_CUDA}>
              GRAPH_HAS_OPENMP=$<BOOL:${GRAPH_ENABLE_OPENMP}>
)

# Consumers must call this after linking graph::graph. It is the project-wide
# entry point for the mandatory alpaka_finalize() call.
function(graph_finalize target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "graph_finalize(): target '${target}' does not exist")
    endif()
    get_target_property(_graph_target_type ${target} TYPE)
    if(_graph_target_type STREQUAL "INTERFACE_LIBRARY")
        message(
            FATAL_ERROR
            "graph_finalize(): '${target}' is an INTERFACE library; call it on a target with sources"
        )
    endif()
    alpaka_finalize(${target})
endfunction()
