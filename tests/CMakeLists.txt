# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Canonical test registration. Every `tests/unit/*.cpp` and
# `tests/integration/*.cpp` is globbed at configure time; CONFIGURE_DEPENDS
# re-runs the glob at build time, so adding a test file needs no CMake edit.
# Each directory gets exactly one Catch2 executable (Catch2::Catch2WithMain)
# and one catch_discover_tests() registration covering all of its sources.
# No explicit per-file or historical target names remain.
file(GLOB GRAPH_UNIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
file(GLOB GRAPH_INTEGRATION_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp")

function(graph_add_test_target directory sources)
    if(NOT sources)
        return()
    endif()

    set(_graph_test_target "graph_${directory}_tests")
    add_executable(${_graph_test_target} ${sources})
    target_link_libraries(${_graph_test_target} PRIVATE graph::graph Catch2::Catch2WithMain)

    # Tests share headers across directories: the unit tests use the scripted
    # integration fixture (toy_ops.hpp) and the integration tests use the
    # scalar reference oracle (oracle.hpp). The example tree is header-only and
    # has no target of its own.
    target_include_directories(
        ${_graph_test_target}
        PRIVATE
            "${PROJECT_SOURCE_DIR}/examples"
            "${CMAKE_CURRENT_SOURCE_DIR}/unit"
            "${CMAKE_CURRENT_SOURCE_DIR}/integration"
            "${CMAKE_CURRENT_SOURCE_DIR}/reference"
    )
    graph_finalize(${_graph_test_target})

    catch_discover_tests(${_graph_test_target} PROPERTIES TIMEOUT 300)
endfunction()

graph_add_test_target(unit "${GRAPH_UNIT_TEST_SOURCES}")
graph_add_test_target(integration "${GRAPH_INTEGRATION_TEST_SOURCES}")

# Independent scalar reference (oracle) of the particle-feedback semantics.
# Built as its own executable so the acceptance fixtures can be regenerated
# and checked without any project dependency; it depends on the standard
# library only. It is not a Catch2 test and therefore has no CTest entry.
add_executable(graph_particle_reference reference/oracle.cpp)
target_compile_features(graph_particle_reference PRIVATE cxx_std_20)
target_include_directories(graph_particle_reference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reference")
