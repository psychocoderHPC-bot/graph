# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Device-function-pointer ABI probe. Only meaningful for the CUDA backend, so
# the CPU configuration does not build it.
if(GRAPH_ENABLE_CUDA)
    add_executable(graph_abi_probe abi_probe.cu)
    target_link_libraries(graph_abi_probe PRIVATE graph::graph)
    graph_finalize(graph_abi_probe)
endif()
