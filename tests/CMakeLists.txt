# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Register one Catch2 executable per test source file. CONFIGURE_DEPENDS
# re-globs when a later milestone adds a file, so tests never need a CMake edit.
file(GLOB GRAPH_UNIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
file(GLOB GRAPH_INTEGRATION_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp")

set(GRAPH_TEST_SOURCES ${GRAPH_UNIT_TEST_SOURCES} ${GRAPH_INTEGRATION_TEST_SOURCES})

foreach(_graph_test_source IN LISTS GRAPH_TEST_SOURCES)
    get_filename_component(_graph_test_stem "${_graph_test_source}" NAME_WE)
    string(REGEX REPLACE "^test_" "" _graph_test_name "${_graph_test_stem}")
    set(_graph_test_target "graph_${_graph_test_name}")

    add_executable(${_graph_test_target} "${_graph_test_source}")
    target_link_libraries(${_graph_test_target} PRIVATE graph::graph Catch2::Catch2WithMain)
    graph_finalize(${_graph_test_target})

    catch_discover_tests(${_graph_test_target} PROPERTIES TIMEOUT 300)
endforeach()
