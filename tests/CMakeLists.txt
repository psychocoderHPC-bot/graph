# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Register one Catch2 executable per unit test source file. CONFIGURE_DEPENDS
# re-globs when a later milestone adds a file, so tests never need a CMake edit.
file(GLOB GRAPH_UNIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")

foreach(_graph_unit_source IN LISTS GRAPH_UNIT_TEST_SOURCES)
    get_filename_component(_graph_unit_stem "${_graph_unit_source}" NAME_WE)
    string(REGEX REPLACE "^test_" "" _graph_unit_name "${_graph_unit_stem}")
    set(_graph_unit_target "graph_${_graph_unit_name}")

    add_executable(${_graph_unit_target} "${_graph_unit_source}")
    target_link_libraries(${_graph_unit_target} PRIVATE graph::graph Catch2::Catch2WithMain)
    graph_finalize(${_graph_unit_target})

    catch_discover_tests(${_graph_unit_target} PROPERTIES TIMEOUT 300)
endforeach()
