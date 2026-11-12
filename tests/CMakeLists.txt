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

add_executable(test_ready_ring unit/test_ready_ring.cpp)
target_link_libraries(test_ready_ring PRIVATE graph::graph Catch2::Catch2WithMain)
graph_finalize(test_ready_ring)

catch_discover_tests(test_ready_ring PROPERTIES TIMEOUT 120)

add_executable(test_packet_pool unit/test_packet_pool.cpp)
target_link_libraries(test_packet_pool PRIVATE graph::graph Catch2::Catch2WithMain)
graph_finalize(test_packet_pool)

catch_discover_tests(test_packet_pool PROPERTIES TIMEOUT 120)

add_executable(test_ready_ring_minimal unit/test_ready_ring_minimal.cpp)
target_link_libraries(test_ready_ring_minimal PRIVATE graph::graph Catch2::Catch2WithMain)
graph_finalize(test_ready_ring_minimal)

catch_discover_tests(test_ready_ring_minimal PROPERTIES TIMEOUT 120)
