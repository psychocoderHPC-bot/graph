# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

include_guard(GLOBAL)

# Capture the per-thread register allocation of the translation unit that
# instantiates the persistent kernel. Registers per thread cannot be queried
# without a kernel handle, so the value is taken from the compiler's
# `-Xptxas -v` output at configure time and injected as a compile definition.
option(
    GRAPH_CAPTURE_KERNEL_REGISTERS
    "Capture persistent-kernel register usage at configure time (-Xptxas -v)"
    ON
)

# Value used when the capture is unavailable (CPU-only build, no CUDA compiler,
# compile failure, or no ptxas report). It must be conservative: a value that is
# too high only lowers the computed wave, never raises it.
set(
    GRAPH_KERNEL_REGISTERS_FALLBACK
    "128"
    CACHE STRING "Registers per thread used when the build-time capture is unavailable"
)

# Captures `-Xptxas -v` register usage for `cmake/GraphRegisterProbe.cu`, which
# instantiates `graph::RendezvousKernel` through a real alpaka CUDA launch.
#
# Returns the maximum reported register count, or 0 when no capture is
# available. The caller turns 0 into `GRAPH_KERNEL_REGISTERS=0`, and the headers
# then use GRAPH_KERNEL_REGISTERS_FALLBACK.
function(graph_capture_kernel_registers out_var)
    set(${out_var} 0 PARENT_SCOPE)

    if(NOT GRAPH_ENABLE_CUDA)
        message(
            STATUS
            "graph: CUDA is disabled; persistent-kernel register count uses the ${GRAPH_KERNEL_REGISTERS_FALLBACK}-register fallback"
        )
        return()
    endif()
    if(NOT GRAPH_CAPTURE_KERNEL_REGISTERS)
        message(
            STATUS
            "graph: GRAPH_CAPTURE_KERNEL_REGISTERS is OFF; using the ${GRAPH_KERNEL_REGISTERS_FALLBACK}-register fallback"
        )
        return()
    endif()
    if(NOT CMAKE_CUDA_COMPILER)
        message(
            WARNING
            "graph: no CUDA compiler at configure time; using the ${GRAPH_KERNEL_REGISTERS_FALLBACK}-register fallback"
        )
        return()
    endif()

    set(_probe "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GraphRegisterProbe.cu")
    if(NOT EXISTS "${_probe}")
        message(WARNING "graph: register probe '${_probe}' is missing; using the fallback")
        return()
    endif()

    set(_arch_args "")
    if(CMAKE_CUDA_ARCHITECTURES)
        list(GET CMAKE_CUDA_ARCHITECTURES 0 _arch)
        if(_arch MATCHES "^[0-9]+$")
            list(APPEND _arch_args "-arch=sm_${_arch}")
        elseif(_arch STREQUAL "native")
            list(APPEND _arch_args "-arch=native")
        else()
            message(
                WARNING
                "graph: CMAKE_CUDA_ARCHITECTURES='${_arch}' is not a plain compute capability; the register probe lets nvcc choose"
            )
        endif()
    endif()

    set(_cuda_flags "")
    separate_arguments(_cuda_flags NATIVE_COMMAND "${CMAKE_CUDA_FLAGS}")
    set(_config_flags "")
    if(CMAKE_BUILD_TYPE AND CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE})
        separate_arguments(_config_flags NATIVE_COMMAND "${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}}")
    endif()

    set(_include_args "-I${PROJECT_SOURCE_DIR}/include")
    if(alpaka_SOURCE_DIR AND EXISTS "${alpaka_SOURCE_DIR}/include")
        list(APPEND _include_args "-I${alpaka_SOURCE_DIR}/include")
    else()
        message(
            WARNING
            "graph: alpaka include directory is unknown; cannot capture the persistent-kernel register count"
        )
        return()
    endif()

    execute_process(
        COMMAND
            "${CMAKE_CUDA_COMPILER}" -c "${_probe}" -o "${CMAKE_BINARY_DIR}/graph_register_probe.o"
            ${_arch_args}
            ${_cuda_flags}
            ${_config_flags}
            -std=c++20
            --expt-relaxed-constexpr
            --extended-lambda
            -DGRAPH_HAS_CUDA=1
            ${_include_args}
            -Xptxas
            -v
        RESULT_VARIABLE _compile_result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
        TIMEOUT 600
    )

    if(NOT _compile_result EQUAL 0)
        message(
            WARNING
            "graph: persistent-kernel register probe failed (${_compile_result}); using the ${GRAPH_KERNEL_REGISTERS_FALLBACK}-register fallback"
        )
        return()
    endif()

    # ptxas reports one "Used N registers" line per compiled kernel. Taking the
    # maximum is conservative for a translation unit with more than one kernel.
    string(REGEX MATCHALL "Used [0-9]+ registers" _matches "${_stderr}${_stdout}")
    set(_max 0)
    foreach(_match IN LISTS _matches)
        string(REGEX REPLACE "Used ([0-9]+) registers" "\\1" _count "${_match}")
        if(_count GREATER _max)
            set(_max "${_count}")
        endif()
    endforeach()

    if(_max GREATER 0)
        message(STATUS "graph: captured persistent-kernel register usage: ${_max} registers/thread")
        set(${out_var} "${_max}" PARENT_SCOPE)
    else()
        message(
            WARNING
            "graph: the register probe produced no 'Used N registers' line; using the ${GRAPH_KERNEL_REGISTERS_FALLBACK}-register fallback"
        )
    endif()
endfunction()
