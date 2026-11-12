# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Device-function-pointer ABI probe. Only meaningful for the CUDA backend, so
# the CPU configuration does not build it.
if(GRAPH_ENABLE_CUDA)
    add_executable(graph_abi_probe abi_probe.cu)
    target_link_libraries(graph_abi_probe PRIVATE graph::graph)
    graph_finalize(graph_abi_probe)

    # ReadyRing root-cause diagnostic (T7b): a warp vote is not a memory
    # barrier; syncBlockThreads is. CUDA only.
    add_executable(graph_warp_barrier_probe warp_barrier_probe.cu)
    target_link_libraries(graph_warp_barrier_probe PRIVATE graph::graph)
    graph_finalize(graph_warp_barrier_probe)
endif()
