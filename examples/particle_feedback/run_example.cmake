# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0

# Validate one `graph_particle_feedback` run against the committed fixture and,
# optionally, compare two chunk sizes byte-for-byte.
#
# Required:
#   EXECUTABLE    graph_particle_feedback binary
#   BACKEND       cpu | cuda
#   PARTICLES     N
#   CHUNK_SIZE    Y
#   SEED          S
#   FIXTURE_MD    tests/reference/test-vectors.md
# Optional:
#   COMPARE_CHUNK_SIZE   run this Y as well and require identical stdout

cmake_minimum_required(VERSION 3.25)

foreach(required EXECUTABLE BACKEND PARTICLES CHUNK_SIZE SEED FIXTURE_MD)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "run_example.cmake: missing -D${required}=...")
    endif()
endforeach()

if(NOT EXISTS "${EXECUTABLE}")
    message(FATAL_ERROR "run_example.cmake: executable '${EXECUTABLE}' does not exist")
endif()
if(NOT EXISTS "${FIXTURE_MD}")
    message(FATAL_ERROR "run_example.cmake: fixture '${FIXTURE_MD}' does not exist")
endif()

function(run_case chunk_size out_var)
    execute_process(
        COMMAND "${EXECUTABLE}"
            --backend "${BACKEND}"
            --particles "${PARTICLES}"
            --chunk-size "${chunk_size}"
            --seed "${SEED}"
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE output
        ERROR_VARIABLE diagnostics
        TIMEOUT 240
    )
    if(NOT exit_code EQUAL 0)
        message(
            FATAL_ERROR
            "run_example.cmake: Y=${chunk_size} exited ${exit_code}\n--- stderr ---\n${diagnostics}"
        )
    endif()
    set(${out_var} "${output}" PARENT_SCOPE)
endfunction()

function(split_rows output out_var)
    string(REPLACE "\r\n" "\n" cleaned "${output}")
    string(REPLACE "\n" ";" lines "${cleaned}")
    set(rows "")
    set(total 0)
    foreach(line IN LISTS lines)
        if(line STREQUAL "")
            continue()
        endif()
        if(NOT line MATCHES "^[0-9]+ [0-9]+$")
            message(FATAL_ERROR "run_example.cmake: unexpected stdout line '${line}'")
        endif()
        string(REGEX MATCH "^([0-9]+) ([0-9]+)$" match "${line}")
        math(EXPR total "${total} + ${CMAKE_MATCH_2}")
        list(APPEND rows "${line}")
    endforeach()
    set(${out_var} "${rows}" PARENT_SCOPE)
    set(GRAPH_EXAMPLE_TOTAL "${total}" PARENT_SCOPE)
endfunction()

run_case("${CHUNK_SIZE}" output)
split_rows("${output}" rows)

if(NOT GRAPH_EXAMPLE_TOTAL EQUAL PARTICLES)
    message(
        FATAL_ERROR
        "run_example.cmake: histogram total ${GRAPH_EXAMPLE_TOTAL} does not equal N=${PARTICLES}"
    )
endif()

# The fixture section is `### N=<N> seed=<S>` followed by `weight count` rows
# until the next heading.
file(STRINGS "${FIXTURE_MD}" fixture_lines)
set(header "### N=${PARTICLES} seed=${SEED}")
set(in_section FALSE)
set(expected "")
foreach(line IN LISTS fixture_lines)
    if(line STREQUAL header)
        set(in_section TRUE)
        continue()
    endif()
    if(in_section AND line MATCHES "^##")
        break()
    endif()
    if(in_section AND line MATCHES "^[0-9]+ [0-9]+$")
        list(APPEND expected "${line}")
    endif()
endforeach()

if("${rows}" STREQUAL "")
    set(rows_text "")
else()
    set(rows_text "${rows}")
endif()
if("${expected}" STREQUAL "")
    set(expected_text "")
else()
    set(expected_text "${expected}")
endif()

if(NOT "${rows_text}" STREQUAL "${expected_text}")
    message(
        FATAL_ERROR
        "run_example.cmake: Y=${CHUNK_SIZE} histogram differs from the N=${PARTICLES} seed=${SEED} fixture\n"
        "--- observed ---\n${rows_text}\n--- expected ---\n${expected_text}"
    )
endif()

if(DEFINED COMPARE_CHUNK_SIZE)
    run_case("${COMPARE_CHUNK_SIZE}" compared_output)
    split_rows("${compared_output}" compared_rows)
    if(NOT GRAPH_EXAMPLE_TOTAL EQUAL PARTICLES)
        message(
            FATAL_ERROR
            "run_example.cmake: Y=${COMPARE_CHUNK_SIZE} histogram total ${GRAPH_EXAMPLE_TOTAL} does not equal N=${PARTICLES}"
        )
    endif()
    if(NOT "${compared_output}" STREQUAL "${output}")
        message(
            FATAL_ERROR
            "run_example.cmake: chunk-size invariance failed between Y=${CHUNK_SIZE} and Y=${COMPARE_CHUNK_SIZE}"
        )
    endif()
endif()

message(STATUS "run_example.cmake: N=${PARTICLES} seed=${SEED} Y=${CHUNK_SIZE} rows validated")
