# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Detect the compute capability of the first visible NVIDIA GPU. Returns a
# CMAKE_CUDA_ARCHITECTURES compatible value such as "80" for an A30.
function(graph_detect_cuda_arch out_var)
    find_program(_graph_nvidia_smi nvidia-smi)
    if(_graph_nvidia_smi)
        execute_process(
            COMMAND "${_graph_nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _graph_compute_caps
            ERROR_QUIET
            RESULT_VARIABLE _graph_nvidia_smi_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_graph_nvidia_smi_result EQUAL 0 AND _graph_compute_caps)
            string(REPLACE "\n" ";" _graph_compute_caps "${_graph_compute_caps}")
            list(GET _graph_compute_caps 0 _graph_compute_cap)
            string(STRIP "${_graph_compute_cap}" _graph_compute_cap)
            string(REPLACE "." "" _graph_compute_cap "${_graph_compute_cap}")
            if(_graph_compute_cap MATCHES "^[0-9]+$")
                set(${out_var} "${_graph_compute_cap}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # ampera/datacenter fallback (NVIDIA A30, A100)
    set(${out_var} "80" PARENT_SCOPE)
endfunction()
