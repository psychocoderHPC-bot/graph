# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Drop-in test registration: every `tests/unit/test_*.cpp` and
# `tests/integration/test_*.cpp` becomes its own Catch2 executable and CTest
# entry, so adding a test file needs no CMake edit. CONFIGURE_DEPENDS re-runs
# the glob at build time. The example tree is on the include path because the
# CLI header is header-only and has no target of its own yet. The reference
# tree is on the include path because the oracle header is standalone and has
# no target of its own.
file(
    GLOB GRAPH_TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_*.cpp"
)

foreach(graph_test_source IN LISTS GRAPH_TEST_SOURCES)
    get_filename_component(graph_test_name "${graph_test_source}" NAME_WE)
    add_executable(${graph_test_name} "${graph_test_source}")
    target_link_libraries(${graph_test_name} PRIVATE graph::graph Catch2::Catch2WithMain)
    target_include_directories(
        ${graph_test_name}
        PRIVATE
            "${PROJECT_SOURCE_DIR}/examples"
            "${PROJECT_SOURCE_DIR}/tests/reference"
    )
    graph_finalize(${graph_test_name})
    catch_discover_tests(${graph_test_name} PROPERTIES TIMEOUT 300)
endforeach()

# Independent scalar reference (oracle) of the particle-feedback semantics.
# Built as its own executable so the acceptance fixtures can be regenerated
# and checked without any project dependency; it depends on the standard
# library only.
add_executable(graph_particle_reference reference/oracle.cpp)
target_compile_features(graph_particle_reference PRIVATE cxx_std_20)
target_include_directories(graph_particle_reference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reference")
