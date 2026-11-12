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
    target_include_directories(${_graph_test_target} PRIVATE "${PROJECT_SOURCE_DIR}/examples")
    graph_finalize(${_graph_test_target})

    catch_discover_tests(${_graph_test_target} PROPERTIES TIMEOUT 300)
endfunction()

graph_add_test_target(unit "${GRAPH_UNIT_TEST_SOURCES}")
graph_add_test_target(integration "${GRAPH_INTEGRATION_TEST_SOURCES}")
