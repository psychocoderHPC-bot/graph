# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

add_executable(graph_particle_feedback main.cpp)
target_link_libraries(graph_particle_feedback PRIVATE graph::graph)
# The CUDA build compiles a copy of the source tree under
# `alpaka_build_files/cuda/`; the quoted example include therefore needs an
# explicit search path.
target_include_directories(graph_particle_feedback PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
graph_finalize(graph_particle_feedback)

if(BUILD_TESTING)
    set(_graph_example_fixture "${PROJECT_SOURCE_DIR}/tests/reference/test-vectors.md")
    set(_graph_example_helper "${CMAKE_CURRENT_SOURCE_DIR}/run_example.cmake")

    function(graph_add_example_case name backend particles chunk_size seed)
        add_test(
            NAME ${name}
            COMMAND
                ${CMAKE_COMMAND}
                -DEXECUTABLE=$<TARGET_FILE:graph_particle_feedback>
                -DBACKEND=${backend}
                -DPARTICLES=${particles}
                -DCHUNK_SIZE=${chunk_size}
                -DSEED=${seed}
                -DFIXTURE_MD=${_graph_example_fixture}
                -P ${_graph_example_helper}
        )
        # One pool run owns the whole device; serializing it keeps a parallel
        # `ctest -j` invocation from oversubscribing the persistent launch.
        set_tests_properties(${name} PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
    endfunction()

    graph_add_example_case(graph_particle_feedback_cpu cpu 4096 128 42)
    graph_add_example_case(graph_particle_feedback_cpu_zero cpu 0 128 42)
    graph_add_example_case(graph_particle_feedback_cpu_y100 cpu 4096 100 42)

    if(GRAPH_ENABLE_CUDA)
        graph_add_example_case(graph_particle_feedback_cuda cuda 4096 128 42)
        graph_add_example_case(graph_particle_feedback_cuda_chunk100 cuda 4096 100 42)
        add_test(
            NAME graph_particle_feedback_cuda_chunk_invariance
            COMMAND
                ${CMAKE_COMMAND}
                -DEXECUTABLE=$<TARGET_FILE:graph_particle_feedback>
                -DBACKEND=cuda
                -DPARTICLES=4096
                -DCHUNK_SIZE=128
                -DCOMPARE_CHUNK_SIZE=100
                -DSEED=42
                -DFIXTURE_MD=${_graph_example_fixture}
                -P ${_graph_example_helper}
        )
        set_tests_properties(
            graph_particle_feedback_cuda
            graph_particle_feedback_cuda_chunk100
            graph_particle_feedback_cuda_chunk_invariance
            PROPERTIES
                TIMEOUT 300
                RUN_SERIAL TRUE
                RESOURCE_LOCK graph_gpu_device
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
        )
    endif()
endif()
