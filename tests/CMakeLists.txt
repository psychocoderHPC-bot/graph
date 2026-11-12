# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

add_executable(graph_smoke unit/test_smoke.cpp)
target_link_libraries(graph_smoke PRIVATE graph::graph Catch2::Catch2WithMain)
graph_finalize(graph_smoke)

catch_discover_tests(graph_smoke PROPERTIES TIMEOUT 120)

add_executable(test_vocabulary unit/test_vocabulary.cpp)
target_link_libraries(test_vocabulary PRIVATE graph::graph Catch2::Catch2WithMain)
graph_finalize(test_vocabulary)

catch_discover_tests(test_vocabulary PROPERTIES TIMEOUT 120)

add_executable(test_chunk_assembly unit/test_chunk_assembly.cpp)
target_link_libraries(test_chunk_assembly PRIVATE graph::graph Catch2::Catch2WithMain)
graph_finalize(test_chunk_assembly)

catch_discover_tests(test_chunk_assembly PROPERTIES TIMEOUT 300)
