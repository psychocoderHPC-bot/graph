#!/bin/sh
# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0
#
# HAL acceptance harness for the graph repository.
#
# One command reproduces the hardware acceptance evidence of design section 9
# and 10: environment capture, a fresh CPU and CUDA configure/build, the CTest
# suites with JUnit output, the particle-feedback example (when it exists), the
# format and REUSE gates, and a filled acceptance record.
#
# The hal login profile exports LD_LIBRARY_PATH and CMAKE_PREFIX_PATH from a
# oneAPI setup. Those values select a different C++ runtime and a different
# CMake package set, so every configure, build, test, and example command is
# run with the same scrub prefix documented in docs/build.md:
#
#   env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
#       -u CPATH -u CPLUS_INCLUDE_PATH
#
# Usage: scripts/hal-acceptance.sh --help

# The literal `$ARTIFACT` in generated documentation text is intentional.
# shellcheck disable=SC2016

set -eu

# SC1007: `CDPATH= cd` is the repository's canonical way to resolve a path
# without inheriting CDPATH.
# shellcheck disable=SC1007
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# shellcheck disable=SC1007
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
template="$repo_root/docs/hal-acceptance.template.md"

usage()
{
    cat <<'EOF'
Usage: scripts/hal-acceptance.sh [options]

Run the HAL acceptance gates and write a filled acceptance record into a new
artifact directory. The committed docs/hal-acceptance.md is only overwritten
with --write-doc.

Options:
  --artifact-dir DIR        Output directory (default:
                            $TMPDIR/graph-hal-acceptance/<utc>-<pid>)
  --cuda-arch N             CMAKE_CUDA_ARCHITECTURES for the CUDA build
                            (default: 80). It must match the target device;
                            sm_80 runs on the A30.
  --cuda-visible-devices L  CUDA_VISIBLE_DEVICES for CUDA tests and examples
                            (default: $CUDA_VISIBLE_DEVICES, else 0)
  --jobs N                  Parallel build jobs (default: nproc)
  --timeout SEC             Per-test ctest timeout for both configurations
                            (default: 600)
  --cpu-timeout SEC         Per-test ctest timeout for the CPU suite
                            (default: 600)
  --cuda-timeout SEC        Per-test ctest timeout for the CUDA suite
                            (default: 900)
  --repeat N                Repeat every ctest case until a failure or N runs
                            (default: 3); the executed case count is recorded
                            as cases x N.
  --pool-repeat N           Repeat the CUDA ctest cases matching -R pool up to
                            N times (default: 20), with an overall wall-clock
                            bound from --pool-repeat-timeout.
  --pool-repeat-timeout SEC Wall-clock timeout for the pool repeat gate
                            (default: 3600)
  --sanitizer-timeout SEC   Wall-clock timeout for each compute-sanitizer group
                            (default: 600)
  --sanitize                Run the CUDA memcheck groups (default: on when
                            compute-sanitizer is on PATH)
  --no-sanitize             Do not run the CUDA memcheck groups (pending gate)
  --example-timeout SEC     Wall-clock timeout for one example run
                            (default: 300)
  --configure-timeout SEC   Wall-clock timeout for one cmake configure
                            (default: 1800)
  --build-timeout SEC       Wall-clock timeout for one cmake build
                            (default: 3600)
  --lint-timeout SEC        Wall-clock timeout for one lint gate
                            (default: 600)
  --particles N             Example particle count (default: 65536)
  --chunk-size Y            Example chunk size (default: 128)
  --seed S                  Example seed (default: 42)
  --alpaka-dir DIR          Use an existing alpaka checkout (offline builds).
                            Defaults to $FETCHCONTENT_SOURCE_DIR_ALPAKA.
  --catch2-dir DIR          Use an existing Catch2 checkout.
                            Defaults to $FETCHCONTENT_SOURCE_DIR_CATCH2.
  --skip-cpu                Do not run the CPU configuration (pending gate)
  --skip-cuda               Do not run the CUDA configuration (pending gate)
  --allow-dirty             Proceed with a dirty tree (recorded as dirty)
  --allow-pending           Exit 0 when gates are pending but none failed
  --write-doc               Overwrite docs/hal-acceptance.md with the record
  -h, --help                Show this help

Gates:
  CPU build/test, CUDA build/test, "no skipped CUDA tests", the CUDA pool
  repeat, the two compute-sanitizer memcheck groups, the example on both
  backends, scripts/check-format.sh, and scripts/reuse-lint.sh.

Every gate row names the command that produced it and the artifact that holds
its output. A skipped test on the CUDA configuration fails the run: a skipped
hardware test is not a pass. A skipped configuration (no nvcc, no NVIDIA
device, --skip-*) is recorded as pending, not failed. A missing example target
is recorded as NOT-BUILT and does not fail the run; the gate becomes an
executed pass/fail gate automatically once examples/particle_feedback builds
a target. A missing compute-sanitizer is pending and is never recorded as a
pass. A sanitizer gate passes only when its ERROR SUMMARY reports 0 errors.

Exit status:
  0  every executed gate passed and no gate is pending
  1  at least one gate failed
  2  no gate failed, but at least one gate is pending

Example output contract validated here: one "weight count" integer pair per
line, sorted ascending. The expected rows come from the committed fixture in
tests/reference/test-vectors.md for the same (N, seed); chunk size must not
change the histogram.
EOF
}

die()
{
    printf 'error: %s\n' "$*" >&2
    exit 1
}

have()
{
    command -v "$1" >/dev/null 2>&1
}

need_arg()
{
    [ "$#" -ge 2 ] || die "option '$1' requires a value"
}

require_uint()
{
    case "$2" in
        '' | *[!0-9]*) die "$1 expects a non-negative integer, got '$2'" ;;
    esac
}

scrubbed()
{
    env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH -u CPATH \
        -u CPLUS_INCLUDE_PATH "$@"
}

scrubbed_cuda()
{
    env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH -u CPATH \
        -u CPLUS_INCLUDE_PATH CUDA_VISIBLE_DEVICES="$cuda_visible" "$@"
}

# ---------------------------------------------------------------- options ---

artifact_dir=""
cuda_arch=80
jobs=""
cpu_timeout=600
cuda_timeout=900
repeat=3
pool_repeat=20
pool_repeat_timeout=3600
sanitizer_timeout=600
sanitize=auto
example_timeout=300
configure_timeout=1800
build_timeout=3600
lint_timeout=600
particles=65536
chunk_size=128
seed=42
alpaka_dir="${FETCHCONTENT_SOURCE_DIR_ALPAKA:-}"
catch2_dir="${FETCHCONTENT_SOURCE_DIR_CATCH2:-}"
cuda_visible="${CUDA_VISIBLE_DEVICES:-0}"
run_cpu=1
run_cuda=1
allow_dirty=0
allow_pending=0
write_doc=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --artifact-dir)
            need_arg "$@"
            artifact_dir=$2
            shift 2
            ;;
        --artifact-dir=*)
            artifact_dir=${1#*=}
            shift
            ;;
        --cuda-arch)
            need_arg "$@"
            cuda_arch=$2
            shift 2
            ;;
        --cuda-arch=*)
            cuda_arch=${1#*=}
            shift
            ;;
        --cuda-visible-devices)
            need_arg "$@"
            cuda_visible=$2
            shift 2
            ;;
        --cuda-visible-devices=*)
            cuda_visible=${1#*=}
            shift
            ;;
        --jobs)
            need_arg "$@"
            jobs=$2
            shift 2
            ;;
        --jobs=*)
            jobs=${1#*=}
            shift
            ;;
        --timeout)
            need_arg "$@"
            cpu_timeout=$2
            cuda_timeout=$2
            shift 2
            ;;
        --timeout=*)
            cpu_timeout=${1#*=}
            cuda_timeout=${1#*=}
            shift
            ;;
        --cpu-timeout)
            need_arg "$@"
            cpu_timeout=$2
            shift 2
            ;;
        --cpu-timeout=*)
            cpu_timeout=${1#*=}
            shift
            ;;
        --cuda-timeout)
            need_arg "$@"
            cuda_timeout=$2
            shift 2
            ;;
        --cuda-timeout=*)
            cuda_timeout=${1#*=}
            shift
            ;;
        --repeat)
            need_arg "$@"
            repeat=$2
            shift 2
            ;;
        --repeat=*)
            repeat=${1#*=}
            shift
            ;;
        --pool-repeat)
            need_arg "$@"
            pool_repeat=$2
            shift 2
            ;;
        --pool-repeat=*)
            pool_repeat=${1#*=}
            shift
            ;;
        --pool-repeat-timeout)
            need_arg "$@"
            pool_repeat_timeout=$2
            shift 2
            ;;
        --pool-repeat-timeout=*)
            pool_repeat_timeout=${1#*=}
            shift
            ;;
        --sanitizer-timeout)
            need_arg "$@"
            sanitizer_timeout=$2
            shift 2
            ;;
        --sanitizer-timeout=*)
            sanitizer_timeout=${1#*=}
            shift
            ;;
        --sanitize)
            sanitize=on
            shift
            ;;
        --no-sanitize)
            sanitize=off
            shift
            ;;
        --example-timeout)
            need_arg "$@"
            example_timeout=$2
            shift 2
            ;;
        --example-timeout=*)
            example_timeout=${1#*=}
            shift
            ;;
        --configure-timeout)
            need_arg "$@"
            configure_timeout=$2
            shift 2
            ;;
        --configure-timeout=*)
            configure_timeout=${1#*=}
            shift
            ;;
        --build-timeout)
            need_arg "$@"
            build_timeout=$2
            shift 2
            ;;
        --build-timeout=*)
            build_timeout=${1#*=}
            shift
            ;;
        --lint-timeout)
            need_arg "$@"
            lint_timeout=$2
            shift 2
            ;;
        --lint-timeout=*)
            lint_timeout=${1#*=}
            shift
            ;;
        --particles)
            need_arg "$@"
            particles=$2
            shift 2
            ;;
        --particles=*)
            particles=${1#*=}
            shift
            ;;
        --chunk-size)
            need_arg "$@"
            chunk_size=$2
            shift 2
            ;;
        --chunk-size=*)
            chunk_size=${1#*=}
            shift
            ;;
        --seed)
            need_arg "$@"
            seed=$2
            shift 2
            ;;
        --seed=*)
            seed=${1#*=}
            shift
            ;;
        --alpaka-dir)
            need_arg "$@"
            alpaka_dir=$2
            shift 2
            ;;
        --alpaka-dir=*)
            alpaka_dir=${1#*=}
            shift
            ;;
        --catch2-dir)
            need_arg "$@"
            catch2_dir=$2
            shift 2
            ;;
        --catch2-dir=*)
            catch2_dir=${1#*=}
            shift
            ;;
        --skip-cpu)
            run_cpu=0
            shift
            ;;
        --skip-cuda)
            run_cuda=0
            shift
            ;;
        --allow-dirty)
            allow_dirty=1
            shift
            ;;
        --allow-pending)
            allow_pending=1
            shift
            ;;
        --write-doc)
            write_doc=1
            shift
            ;;
        --)
            shift
            [ "$#" -eq 0 ] || die "unexpected argument '$1'"
            ;;
        -*)
            die "unknown option '$1' (try --help)"
            ;;
        *)
            die "unexpected argument '$1'"
            ;;
    esac
done

require_uint --cuda-arch "$cuda_arch"
require_uint --cpu-timeout "$cpu_timeout"
require_uint --cuda-timeout "$cuda_timeout"
require_uint --repeat "$repeat"
require_uint --pool-repeat "$pool_repeat"
require_uint --pool-repeat-timeout "$pool_repeat_timeout"
require_uint --sanitizer-timeout "$sanitizer_timeout"
require_uint --example-timeout "$example_timeout"
require_uint --configure-timeout "$configure_timeout"
require_uint --build-timeout "$build_timeout"
require_uint --lint-timeout "$lint_timeout"
require_uint --particles "$particles"
require_uint --chunk-size "$chunk_size"
require_uint --seed "$seed"
[ "$cuda_arch" -gt 0 ] || die "--cuda-arch must be a positive compute capability"
[ "$cpu_timeout" -gt 0 ] || die "--cpu-timeout must be positive"
[ "$cuda_timeout" -gt 0 ] || die "--cuda-timeout must be positive"
[ "$repeat" -gt 0 ] || die "--repeat must be positive"
[ "$pool_repeat" -gt 0 ] || die "--pool-repeat must be positive"
[ "$pool_repeat_timeout" -gt 0 ] || die "--pool-repeat-timeout must be positive"
[ "$sanitizer_timeout" -gt 0 ] || die "--sanitizer-timeout must be positive"
[ "$example_timeout" -gt 0 ] || die "--example-timeout must be positive"
[ "$configure_timeout" -gt 0 ] || die "--configure-timeout must be positive"
[ "$build_timeout" -gt 0 ] || die "--build-timeout must be positive"
[ "$lint_timeout" -gt 0 ] || die "--lint-timeout must be positive"
[ "$chunk_size" -gt 0 ] || die "--chunk-size must be at least 1"

if [ -z "$jobs" ]; then
    jobs=$(nproc 2>/dev/null || printf '4')
fi
require_uint --jobs "$jobs"
[ "$jobs" -gt 0 ] || die "--jobs must be positive"

[ -n "$cuda_visible" ] || cuda_visible=0
[ -f "$template" ] || die "missing acceptance template: $template"

# ----------------------------------------------------------- repository ---

have git || die "git is required"
have timeout || die "timeout(1) is required for the bounded configure, build, and lint gates"
git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1 || die "$repo_root is not a git checkout"

git_commit=$(git -C "$repo_root" rev-parse HEAD)
git_status=$(git -C "$repo_root" status --porcelain)
if [ -n "$git_status" ] && [ "$allow_dirty" -eq 0 ]; then
    printf '%s\n' "$git_status" >&2
    die "working tree is dirty; commit the changes or pass --allow-dirty"
fi
if [ -n "$git_status" ]; then
    git_dirty=yes
else
    git_dirty=no
fi

# ------------------------------------------------------------- artifact ---

generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ -z "$artifact_dir" ]; then
    artifact_dir="${TMPDIR:-/tmp}/graph-hal-acceptance/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
mkdir -p "$artifact_dir"
# shellcheck disable=SC1007
artifact_dir=$(CDPATH= cd -- "$artifact_dir" && pwd)

case "$artifact_dir" in
    "$repo_root" | "$repo_root"/*)
        printf 'warning: artifact directory is inside the checkout; it will affect reuse lint\n' >&2
        ;;
esac

results="$artifact_dir/results.tsv"
commands="$artifact_dir/commands.txt"
env_file="$artifact_dir/environment.txt"
details="$artifact_dir/details.md"
example_summary="$artifact_dir/example-summary.md"
diagnostics="$artifact_dir/diagnostics.md"
artifacts_list="$artifact_dir/artifacts.md"
verdict_file="$artifact_dir/verdict.md"
summary="$artifact_dir/summary.md"
test_counts="$artifact_dir/test-counts.md"

: > "$results"
: > "$commands"
: > "$env_file"
: > "$details"
: > "$example_summary"
: > "$diagnostics"
: > "$artifacts_list"
: > "$test_counts"

any_fail=0
any_pending=0

add_result()
{
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" \
        "$(printf '%s' "$3" | tr '\n\t' '  ')" \
        "$(printf '%s' "$4" | tr '\n\t' '  ')" \
        "$(printf '%s' "$5" | tr '\n\t' '  ')" >>"$results"
}

log_cmd()
{
    printf '$ %s\n' "$(printf '%s' "$*" | sed -e "s|$artifact_dir|\$ARTIFACT|g" -e "s|$repo_root|.|g")" >> "$commands"
}

log_header()
{
    header_file=$1
    header_title=$2
    {
        printf '# HAL acceptance log\n'
        printf '# commit: %s\n' "$git_commit"
        printf '# artifact: %s\n' "$artifact_dir"
        printf '# generated: %s\n' "$generated_at"
        printf '# gate: %s\n' "$header_title"
        printf '\n'
    } >"$header_file"
}

log_header "$commands" "command log"

record()
{
    record_key=$1
    shift
    record_value=$(printf '%s' "$*" | tr '\n\t' '  ')
    printf '%s=%s\n' "$record_key" "$record_value" >> "$env_file"
}

# ---------------------------------------------------------- environment ---

record hostname "$(hostname 2>/dev/null || printf 'unknown')"
record uname "$(uname -a 2>/dev/null || printf 'unknown')"
record git_commit "$git_commit"
record git_dirty "$git_dirty"
record git_status_porcelain "$git_status"
record generated_at "$generated_at"
record artifact_dir "$artifact_dir"
record cuda_arch "$cuda_arch"
record cuda_visible_devices "$cuda_visible"
record build_type Release
record build_jobs "$jobs"
record cpu_timeout_s "$cpu_timeout"
record cuda_timeout_s "$cuda_timeout"
record repeat "$repeat"
record pool_repeat "$pool_repeat"
record pool_repeat_timeout_s "$pool_repeat_timeout"
record sanitizer_timeout_s "$sanitizer_timeout"
record sanitize "$sanitize"
record example_timeout_s "$example_timeout"
record configure_timeout_s "$configure_timeout"
record build_timeout_s "$build_timeout"
record lint_timeout_s "$lint_timeout"
record example_particles "$particles"
record example_chunk_size "$chunk_size"
record example_seed "$seed"

record cmake "$(cmake --version 2>/dev/null | head -1 || printf 'unavailable')"
record ninja "$(ninja --version 2>/dev/null | head -1 || printf 'unavailable')"
if have g++; then
    record gxx "$(g++ --version 2>/dev/null | head -1)"
else
    record gxx "unavailable"
fi
record nproc "$(nproc 2>/dev/null || printf 'unknown')"
record cpu_model "$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1 || true)"
record lscpu_head5 "$(lscpu 2>/dev/null | head -5 | tr '\n' ';' || true)"

nvidia_smi_list="$artifact_dir/nvidia-smi-L.txt"
nvidia_smi_query="$artifact_dir/nvidia-smi-query.csv"
if have nvidia-smi; then
    if nvidia-smi -L >"$nvidia_smi_list" 2>&1; then :; else printf 'unavailable\n' >"$nvidia_smi_list"; fi
    if nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv \
        >"$nvidia_smi_query" 2>&1; then
        :
    else
        printf 'unavailable\n' >"$nvidia_smi_query"
    fi
else
    printf 'unavailable\n' >"$nvidia_smi_list"
    printf 'unavailable\n' >"$nvidia_smi_query"
fi

gpu_count=$(grep -c '^GPU ' "$nvidia_smi_list" 2>/dev/null || true)
[ -n "$gpu_count" ] || gpu_count=0
record gpu_count "$gpu_count"
gpu_line=$(sed -n '2p' "$nvidia_smi_query" 2>/dev/null || true)
if [ -n "$gpu_line" ] && [ "$gpu_line" != "unavailable" ]; then
    field()
    {
        printf '%s\n' "$gpu_line" | cut -d, -f"$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    }
    record gpu0_name "$(field 1)"
    record gpu0_compute_cap "$(field 2)"
    record gpu0_driver "$(field 3)"
    record gpu0_memory "$(field 4)"
else
    record gpu0_name "unavailable"
    record gpu0_compute_cap "unavailable"
    record gpu0_driver "unavailable"
    record gpu0_memory "unavailable"
fi

if have nvcc; then
    record nvcc_release "$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1 || true)"
    record nvcc "$(nvcc --version 2>/dev/null | grep 'release' | head -1 | sed 's/^[[:space:]]*//' || true)"
else
    record nvcc_release "unavailable"
    record nvcc "unavailable"
fi

default_alpaka_pin=$(grep -oE '"[0-9a-f]{40}"' "$repo_root/cmake/GraphOptions.cmake" | head -1 | tr -d '"' || true)
record alpaka_pin "$default_alpaka_pin"
catch2_tag=$(sed -n 's/.*GIT_TAG[[:space:]]*\(v[0-9][0-9.]*\).*/\1/p' "$repo_root/cmake/GraphDependencies.cmake" | head -1 || true)
record catch2_tag "$catch2_tag"
if [ -n "$alpaka_dir" ]; then
    record alpaka_source "local:$alpaka_dir"
    record alpaka_source_commit "$(git -C "$alpaka_dir" rev-parse HEAD 2>/dev/null || printf 'unknown')"
else
    record alpaka_source "fetchcontent"
    record alpaka_source_commit "$default_alpaka_pin"
fi
if [ -n "$catch2_dir" ]; then
    record catch2_source "local:$catch2_dir"
else
    record catch2_source "fetchcontent"
fi

# --------------------------------------------------------------- helpers ---

junit_attr()
{
    sed -n "s/.*$2=\"\([0-9][0-9]*\)\".*/\1/p" "$1" | head -1
}

list_skipped()
{
    tr '\n' ' ' <"$1" | sed 's/<testcase/\n<testcase/g' \
        | grep '<skipped' \
        | sed -n 's/.*name="\([^"]*\)".*/\1/p' \
        | paste -sd ';' - 2>/dev/null || true
}

extract_fixture()
{
    awk -v header="### N=$1 seed=$2" '
        $0 == header { found = 1; next }
        found && /^##/ { exit }
        found && /^[0-9]+ [0-9]+$/ { print }
        END { if (found == 0) exit 1 }
    ' "$repo_root/tests/reference/test-vectors.md" >"$3"
}

extract_diagnostics()
{
    jump=$1
    out=$2
    awk '
        /<system-out>/ { inout = 1 }
        inout { print }
        /<\/system-out>/ { inout = 0 }
    ' "$jump" 2>/dev/null \
        | sed -e 's/<system-out>//' -e 's|</system-out>||' \
            -e 's/&quot;/"/g' -e 's/&amp;/\&/g' -e 's/&lt;/</g' -e 's/&gt;/>/g' \
        | grep -v 'Filters:' \
        | grep -E 'system_atomic capability record|\[residency\]|\[particle_gpu\]|geometry|occupancy|registers/thread|selected=' \
        | sed 's/[[:space:]]*$//' | sort -u >"$out" || true
}

# ---------------------------------------------------------- build gates ---

configure_and_build()
{
    cfg=$1
    cuda_flag=$2
    label=$3
    build="$artifact_dir/build-$cfg"
    conf_log="$artifact_dir/configure-$cfg.log"
    build_log="$artifact_dir/build-$cfg.log"

    rm -rf "$build"

    set -- cmake -S "$repo_root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGRAPH_ENABLE_CUDA="$cuda_flag"
    if [ "$cuda_flag" = ON ]; then
        set -- "$@" -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
    fi
    if [ -n "$alpaka_dir" ]; then
        set -- "$@" -DGRAPH_ALPAKA_SOURCE_DIR="$alpaka_dir" \
            -DFETCHCONTENT_SOURCE_DIR_ALPAKA="$alpaka_dir"
    fi
    if [ -n "$catch2_dir" ]; then
        set -- "$@" -DFETCHCONTENT_SOURCE_DIR_CATCH2="$catch2_dir"
    fi
    log_cmd timeout "$configure_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH "$@"

    log_header "$conf_log" "$label configure"
    if timeout "$configure_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH "$@" >>"$conf_log" 2>&1; then
        conf_rc=0
    else
        conf_rc=$?
    fi
    configure_cmd="cmake -S . -B \$ARTIFACT/build-$cfg -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRAPH_ENABLE_CUDA=$cuda_flag"
    if [ "$cuda_flag" = ON ]; then
        configure_cmd="$configure_cmd -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
    fi
    if [ "$conf_rc" -ne 0 ]; then
        add_result "$label build" FAIL "configure failed (rc=$conf_rc)" \
            "timeout $configure_timeout $configure_cmd" "\$ARTIFACT/configure-$cfg.log"
        any_fail=1
        return 1
    fi

    set -- cmake --build "$build" -j "$jobs"
    log_cmd timeout "$build_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH "$@"
    log_header "$build_log" "$label build"
    if timeout "$build_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH "$@" >>"$build_log" 2>&1; then
        build_rc=0
    else
        build_rc=$?
    fi
    build_cmd="cmake --build \$ARTIFACT/build-$cfg -j $jobs"
    if [ "$build_rc" -ne 0 ]; then
        add_result "$label build" FAIL "build failed (rc=$build_rc)" \
            "timeout $build_timeout $build_cmd" "\$ARTIFACT/build-$cfg.log"
        any_fail=1
        return 1
    fi

    add_result "$label build" PASS "Release, Ninja, -j $jobs" \
        "$configure_cmd; $build_cmd" "\$ARTIFACT/build-$cfg.log"
    {
        printf '### %s build\n\n' "$label"
        printf -- '- build directory: `$ARTIFACT/build-%s`\n' "$cfg"
        printf -- '- configure log: `$ARTIFACT/configure-%s.log`\n' "$cfg"
        printf -- '- build log: `$ARTIFACT/build-%s.log`\n\n' "$cfg"
    } >>"$details"
    return 0
}

ctest_list_count()
{
    list_cfg=$1
    list_build=$2
    list_log="$artifact_dir/ctest-list-$list_cfg.log"
    set -- ctest --test-dir "$list_build" -N
    log_cmd "$@"
    log_header "$list_log" "$list_cfg registered tests"
    if scrubbed "$@" >>"$list_log" 2>&1; then
        list_rc=0
    else
        list_rc=$?
    fi
    list_count=$(sed -n 's/^Total Tests:[[:space:]]*//p' "$list_log" | tail -1)
    case "$list_count" in
        '' | *[!0-9]*)
            if [ "$list_rc" -ne 0 ]; then
                printf '0\n'
            else
                printf '0\n'
            fi
            ;;
        *) printf '%s\n' "$list_count" ;;
    esac
}

ctest_runs()
{
    runs=$(grep -c 'Test[[:space:]]*#[0-9][0-9]*:' "$1" 2>/dev/null || true)
    case "$runs" in
        '' | *[!0-9]*) runs=0 ;;
    esac
    printf '%s\n' "$runs"
}

run_tests()
{
    cfg=$1
    build=$2
    timeout_s=$3
    label=$4
    junit="$artifact_dir/junit-$cfg.xml"
    log="$artifact_dir/ctest-$cfg.log"

    rm -f "$junit"
    registered=$(ctest_list_count "$cfg" "$build")
    [ -n "$registered" ] || registered=0

    if [ "$cfg" = cuda ]; then
        set -- ctest --test-dir "$build" --output-on-failure --output-junit "$junit" \
            --timeout "$timeout_s" --repeat until-fail:"$repeat" --no-tests=error
        log_cmd "$@"
        log_header "$log" "$label tests, repeat $repeat"
        if scrubbed_cuda "$@" >>"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    else
        set -- ctest --test-dir "$build" --output-on-failure --output-junit "$junit" \
            --timeout "$timeout_s" --repeat until-fail:"$repeat" --no-tests=error
        log_cmd "$@"
        log_header "$log" "$label tests, repeat $repeat"
        if scrubbed "$@" >>"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi

    ctest_cmd="ctest --test-dir \$ARTIFACT/build-$cfg --output-on-failure --output-junit \$ARTIFACT/junit-$cfg.xml --timeout $timeout_s --repeat until-fail:$repeat --no-tests=error"

    executed=$(ctest_runs "$log")
    [ "$executed" -gt 0 ] || executed=$((registered * repeat))

    if [ ! -f "$junit" ]; then
        add_result "$label tests" FAIL \
            "ctest failed (rc=$rc), produced no JUnit report, executed $executed case runs" \
            "$ctest_cmd" "\$ARTIFACT/ctest-$cfg.log"
        any_fail=1
        return 1
    fi

    total=$(junit_attr "$junit" tests)
    failures=$(junit_attr "$junit" failures)
    errors=$(junit_attr "$junit" errors)
    skipped=$(junit_attr "$junit" skipped)
    disabled=$(junit_attr "$junit" disabled)
    [ -n "$total" ] || total=0
    [ -n "$failures" ] || failures=0
    [ -n "$errors" ] || errors=0
    [ -n "$skipped" ] || skipped=0
    [ -n "$disabled" ] || disabled=0
    passed=$((total - failures - errors - skipped - disabled))
    [ "$passed" -ge 0 ] || passed=0
    if [ "$registered" -eq 0 ]; then
        registered=$total
    fi

    planned=$((registered * repeat))
    if [ "$executed" -le 0 ]; then
        executed=$planned
    fi
    printf 'registered=%s\nrepeat=%s\nplanned_case_runs=%s\nexecuted_case_runs=%s\nfailures=%s\nerrors=%s\nskipped=%s\ndisabled=%s\npassed_unique=%s\n' \
        "$registered" "$repeat" "$planned" "$executed" \
        "$failures" "$errors" "$skipped" "$disabled" "$passed" \
        >"$artifact_dir/counts-$cfg.txt"

    detail="$passed/$total passed, $failures failed, $skipped skipped; executed $executed/$planned case runs (cases x repeat)"
    if [ "$rc" -ne 0 ]; then
        add_result "$label tests" FAIL "ctest rc=$rc: $detail" \
            "$ctest_cmd" "\$ARTIFACT/ctest-$cfg.log"
        any_fail=1
    else
        add_result "$label tests" PASS "$detail" \
            "$ctest_cmd" "\$ARTIFACT/ctest-$cfg.log"
    fi

    if [ "$cfg" = cuda ]; then
        skipped_total=$((skipped + disabled))
        if [ "$skipped_total" -gt 0 ]; then
            names=$(list_skipped "$junit")
            add_result "CUDA skips" FAIL "skipped: ${names:-<unnamed>}" \
                "$ctest_cmd" "\$ARTIFACT/junit-$cfg.xml"
            any_fail=1
        else
            add_result "CUDA skips" PASS "none in the executed CUDA configuration" \
                "$ctest_cmd" "\$ARTIFACT/junit-$cfg.xml"
        fi
    fi

    {
        printf '### %s tests\n\n' "$label"
        printf '```\n'
        printf '%s\n' "registered=$registered repeat=$repeat planned_case_runs=$planned executed_case_runs=$executed"
        printf '%s\n' "unique: passed=$passed failed=$failures errors=$errors skipped=$skipped disabled=$disabled"
        printf '```\n\n'
        printf -- '- ctest log: `$ARTIFACT/ctest-%s.log`\n' "$cfg"
        printf -- '- JUnit report: `$ARTIFACT/junit-%s.xml`\n' "$cfg"
        printf -- '- registered test list: `$ARTIFACT/ctest-list-%s.log`\n' "$cfg"
        printf -- '- last ctest lines:\n\n```\n'
        tail -n 4 "$log" || true
        printf '```\n\n'
    } >>"$details"

    if [ "$rc" -ne 0 ]; then
        return 1
    fi
    return 0
}

run_repo_gate()
{
    gate=$1
    script=$2
    log=$3
    gate_cmd="timeout $lint_timeout scripts/$script"
    log_cmd timeout "$lint_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH scripts/"$script"
    log_header "$log" "$gate"
    if timeout "$lint_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
        "$repo_root/scripts/$script" >>"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    detail=$(grep -v '^[[:space:]]*$' "$log" | grep -v '^# ' | tail -1 || true)
    if [ "$rc" -eq 0 ]; then
        add_result "$gate" PASS "$detail" "$gate_cmd" "\$ARTIFACT/$(basename "$log")"
    elif [ "$rc" -eq 127 ]; then
        add_result "$gate" FAIL "required tool is missing (rc=127): $detail" \
            "$gate_cmd" "\$ARTIFACT/$(basename "$log")"
        any_fail=1
    else
        add_result "$gate" FAIL "rc=$rc: $detail" \
            "$gate_cmd" "\$ARTIFACT/$(basename "$log")"
        any_fail=1
    fi
}

# ---------------------------------------------------------- example gate ---

run_example_config()
{
    cfg=$1
    build=$2
    backend=$3
    exe=""

    for cand in \
        "$build/examples/particle_feedback/graph_particle_feedback" \
        "$build/examples/graph_particle_feedback" \
        "$build/bin/graph_particle_feedback"; do
        if [ -x "$cand" ]; then
            exe=$cand
            break
        fi
    done
    if [ -z "$exe" ]; then
        exe=$(find "$build" -type f -name graph_particle_feedback -perm -u+x 2>/dev/null | head -1 || true)
    fi
    if [ -z "$exe" ]; then
        add_result "Example ($cfg)" NOT-BUILT \
            "target graph_particle_feedback is absent in the $cfg build" \
            "find \$ARTIFACT/build-$cfg -type f -name graph_particle_feedback -perm -u+x" \
            "\$ARTIFACT/example-summary.md"
        {
            printf '### %s example\n\n' "$cfg"
            printf 'not-built: no `graph_particle_feedback` target exists in the %s build.\n\n' "$cfg"
        } >>"$example_summary"
        return 0
    fi

    example_failed=0
    example_cases=0
    example_detail=""

    for case_name in default zero chunk100; do
        case "$case_name" in
            default)
                set -- "$exe" --backend "$backend" --particles "$particles" \
                    --chunk-size "$chunk_size" --seed "$seed"
                expected_total=$particles
                fixture_seed=$seed
                ;;
            zero)
                set -- "$exe" --backend "$backend" --particles 0 \
                    --chunk-size "$chunk_size" --seed "$seed"
                expected_total=0
                fixture_seed=$seed
                ;;
            chunk100)
                set -- "$exe" --backend "$backend" --particles "$particles" \
                    --chunk-size 100 --seed "$seed"
                expected_total=$particles
                fixture_seed=$seed
                ;;
        esac

        outlog="$artifact_dir/example-$cfg-$case_name.txt"
        rows="$artifact_dir/example-$cfg-$case_name.rows"
        log_cmd "$@"
        log_header "$outlog" "example $cfg $case_name"
        if [ "$cfg" = cuda ]; then
            if timeout "$example_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
                -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
                CUDA_VISIBLE_DEVICES="$cuda_visible" "$@" >>"$outlog" 2>&1; then
                rc=0
            else
                rc=$?
            fi
        else
            if timeout "$example_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
                -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
                "$@" >>"$outlog" 2>&1; then
                rc=0
            else
                rc=$?
            fi
        fi

        grep -E '^[0-9]+ [0-9]+$' "$outlog" >"$rows" 2>/dev/null || true
        row_count=$(wc -l <"$rows" | tr -d ' ')
        total_rows=$(awk '{ sum += $2 } END { print sum + 0 }' "$rows")
        case_status=PASS
        case_detail="rc=$rc rows=$row_count total=$total_rows expected_total=$expected_total"

        if [ "$rc" -ne 0 ]; then
            case_status=FAIL
            case_detail="$case_detail exit-code"
        fi
        if [ "$total_rows" -ne "$expected_total" ]; then
            case_status=FAIL
            case_detail="$case_detail histogram-total-mismatch"
        fi

        expected_rows="$artifact_dir/expected-$cfg-$case_name.rows"
        if extract_fixture "$expected_total" "$fixture_seed" "$expected_rows"; then
            if cmp -s "$rows" "$expected_rows"; then
                case_detail="$case_detail exact=match"
            else
                case_status=FAIL
                case_detail="$case_detail exact=mismatch"
            fi
        else
            case_detail="$case_detail exact=no-fixture"
        fi

        example_cases=$((example_cases + 1))
        if [ "$case_status" = FAIL ]; then
            example_failed=1
        fi
        example_detail="$example_detail $case_name:$case_status"

        {
            printf '### %s example: %s\n\n' "$cfg" "$case_name"
            printf '```\n%s\n```\n\n' "case=$case_name $case_detail"
            printf -- '- output: `$ARTIFACT/example-%s-%s.txt`\n\n' "$cfg" "$case_name"
        } >>"$example_summary"
    done

    example_cmd="graph_particle_feedback --backend $backend (default, zero, chunk100)"
    if [ "$example_failed" -ne 0 ]; then
        add_result "Example ($cfg)" FAIL "$example_cases cases:$example_detail" \
            "$example_cmd" "\$ARTIFACT/example-$cfg-default.txt"
        any_fail=1
    else
        add_result "Example ($cfg)" PASS "$example_cases cases:$example_detail" \
            "$example_cmd" "\$ARTIFACT/example-$cfg-default.txt"
    fi
}

find_cuda_test_binary()
{
    search_build=$1
    for candidate in \
        "$search_build/tests/graph_integration_tests" \
        "$search_build/bin/graph_integration_tests"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    find "$search_build" -type f -name graph_integration_tests -perm -u+x 2>/dev/null | head -1 || true
}

run_pool_repeat()
{
    repeat_build=$1
    list_log="$artifact_dir/ctest-list-pool-repeat.log"
    pool_log="$artifact_dir/pool-repeat.log"

    set -- ctest --test-dir "$repeat_build" -N -R pool
    log_cmd "$@"
    log_header "$list_log" "pool repeat registered tests"
    if scrubbed_cuda "$@" >>"$list_log" 2>&1; then
        list_rc=0
    else
        list_rc=$?
    fi
    pool_cases=$(sed -n 's/^Total Tests:[[:space:]]*//p' "$list_log" | tail -1)
    case "$pool_cases" in
        '' | *[!0-9]*) pool_cases=0 ;;
    esac

    pool_repeat_cmd="ctest --test-dir \$ARTIFACT/build-cuda -R pool --output-on-failure --timeout $cuda_timeout --repeat until-fail:$pool_repeat --no-tests=error"
    if [ "$pool_cases" -eq 0 ]; then
        add_result "Pool repeat" FAIL "no CUDA test matched -R pool (list rc=$list_rc)" \
            "timeout $pool_repeat_timeout $pool_repeat_cmd" "\$ARTIFACT/ctest-list-pool-repeat.log"
        any_fail=1
        return 1
    fi

    set -- ctest --test-dir "$repeat_build" -R pool --output-on-failure \
        --timeout "$cuda_timeout" --repeat until-fail:"$pool_repeat" --no-tests=error
    log_cmd timeout "$pool_repeat_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
        CUDA_VISIBLE_DEVICES="$cuda_visible" "$@"
    log_header "$pool_log" "CUDA pool repeat, $pool_cases cases x $pool_repeat"
    if timeout "$pool_repeat_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
        CUDA_VISIBLE_DEVICES="$cuda_visible" "$@" >>"$pool_log" 2>&1; then
        pool_rc=0
    else
        pool_rc=$?
    fi

    pool_planned=$((pool_cases * pool_repeat))
    pool_executed=$(ctest_runs "$pool_log")
    [ "$pool_executed" -gt 0 ] || pool_executed=$pool_planned
    pool_detail="$pool_executed/$pool_planned case runs ($pool_cases cases x $pool_repeat)"
    if [ "$pool_rc" -ne 0 ]; then
        add_result "Pool repeat" FAIL "rc=$pool_rc: $pool_detail" \
            "timeout $pool_repeat_timeout $pool_repeat_cmd" "\$ARTIFACT/pool-repeat.log"
        any_fail=1
        return 1
    fi
    add_result "Pool repeat" PASS "$pool_detail" \
        "timeout $pool_repeat_timeout $pool_repeat_cmd" "\$ARTIFACT/pool-repeat.log"
    return 0
}

run_sanitizer_group()
{
    san_group=$1
    san_filter=$2
    san_expected=$3
    san_binary=$4
    san_log="$artifact_dir/sanitizer-memcheck-$san_group.log"
    san_list_log="$artifact_dir/sanitizer-filter-$san_group-list.log"
    san_binary_ref=$(printf '%s' "$san_binary" | sed "s|$artifact_dir|\$ARTIFACT|g")
    san_cmd="timeout $sanitizer_timeout compute-sanitizer --tool memcheck --target-processes all --print-limit 20 $san_binary_ref '$san_filter'"

    if [ -z "$san_binary" ] || [ ! -x "$san_binary" ]; then
        log_header "$san_log" "compute-sanitizer $san_group"
        printf 'test binary not found: graph_integration_tests\n' >>"$san_log"
        add_result "Sanitizer ($san_group)" FAIL "graph_integration_tests is missing from the CUDA build" \
            "$san_cmd" "\$ARTIFACT/sanitizer-memcheck-$san_group.log"
        any_fail=1
        return 1
    fi

    set -- "$san_binary" --list-tests "$san_filter"
    log_cmd "$@"
    log_header "$san_list_log" "Catch2 filter list for $san_group"
    if scrubbed_cuda "$@" >>"$san_list_log" 2>&1; then
        list_rc=0
    else
        list_rc=$?
    fi
    if [ "$list_rc" -ne 0 ] || ! grep -Eq "$san_expected" "$san_list_log"; then
        log_header "$san_log" "compute-sanitizer $san_group"
        printf 'Catch2 filter did not list the expected tests: %s\n' "$san_filter" >>"$san_log"
        add_result "Sanitizer ($san_group)" FAIL "Catch2 filter has no matching test (list rc=$list_rc)" \
            "graph_integration_tests --list-tests '$san_filter'" "\$ARTIFACT/sanitizer-filter-$san_group-list.log"
        any_fail=1
        return 1
    fi

    set -- compute-sanitizer --tool memcheck --target-processes all --print-limit 20 \
        "$san_binary" "$san_filter"
    log_cmd timeout "$sanitizer_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
        CUDA_VISIBLE_DEVICES="$cuda_visible" "$@"
    log_header "$san_log" "compute-sanitizer memcheck $san_group"
    if timeout "$sanitizer_timeout" env -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CMAKE_PREFIX_PATH -u CPATH -u CPLUS_INCLUDE_PATH \
        CUDA_VISIBLE_DEVICES="$cuda_visible" "$@" >>"$san_log" 2>&1; then
        san_rc=0
    else
        san_rc=$?
    fi

    san_summary=$(sed -n 's/.*ERROR SUMMARY:/ERROR SUMMARY:/p' "$san_log" | tail -1)
    if [ -z "$san_summary" ]; then
        add_result "Sanitizer ($san_group)" FAIL "no ERROR SUMMARY line (test rc=$san_rc)" \
            "$san_cmd" "\$ARTIFACT/sanitizer-memcheck-$san_group.log"
        any_fail=1
        return 1
    fi
    if [ "$san_rc" -ne 0 ] || ! printf '%s\n' "$san_summary" | grep -q 'ERROR SUMMARY: 0 errors'; then
        add_result "Sanitizer ($san_group)" FAIL "$san_summary (test rc=$san_rc)" \
            "$san_cmd" "\$ARTIFACT/sanitizer-memcheck-$san_group.log"
        any_fail=1
        return 1
    fi
    add_result "Sanitizer ($san_group)" PASS "$san_summary (test rc=0)" \
        "$san_cmd" "\$ARTIFACT/sanitizer-memcheck-$san_group.log"
    return 0
}

# ------------------------------------------------------------ execution ---

printf 'HAL acceptance: repository %s at %s (dirty=%s)\n' "$repo_root" "$git_commit" "$git_dirty"
printf 'HAL acceptance: artifacts in %s\n' "$artifact_dir"

cpu_test_cmd="ctest --test-dir \$ARTIFACT/build-cpu --output-on-failure --output-junit \$ARTIFACT/junit-cpu.xml --timeout $cpu_timeout --repeat until-fail:$repeat --no-tests=error"
if [ "$run_cpu" -eq 1 ]; then
    if configure_and_build cpu OFF CPU; then
        run_tests cpu "$artifact_dir/build-cpu" "$cpu_timeout" CPU || true
        run_example_config cpu "$artifact_dir/build-cpu" cpu
    else
        add_result "CPU tests" FAIL "not run after a CPU build failure" \
            "$cpu_test_cmd" "\$ARTIFACT/build-cpu.log"
        add_result "Example (cpu)" NOT-RUN "not run after a CPU build failure" \
            "graph_particle_feedback --backend cpu" "\$ARTIFACT/example-summary.md"
        any_fail=1
    fi
else
    add_result "CPU build" PENDING "skipped by --skip-cpu" "not run" "-"
    add_result "CPU tests" PENDING "skipped by --skip-cpu" "not run" "-"
    add_result "Example (cpu)" PENDING "skipped by --skip-cpu" "not run" "-"
    any_pending=1
fi

pool_repeat_cmd="ctest --test-dir \$ARTIFACT/build-cuda -R pool --output-on-failure --timeout $cuda_timeout --repeat until-fail:$pool_repeat --no-tests=error"
cuda_test_cmd="ctest --test-dir \$ARTIFACT/build-cuda --output-on-failure --output-junit \$ARTIFACT/junit-cuda.xml --timeout $cuda_timeout --repeat until-fail:$repeat --no-tests=error"
san_pool_cmd="compute-sanitizer --tool memcheck --target-processes all --print-limit 20 \$ARTIFACT/build-cuda/tests/graph_integration_tests '[graph][pool][ladder],[graph][pool][probe]'"
san_chunk_cmd="compute-sanitizer --tool memcheck --target-processes all --print-limit 20 \$ARTIFACT/build-cuda/tests/graph_integration_tests '[chunk_assembly][gpu]'"

if [ "$run_cuda" -eq 1 ]; then
    cuda_reason=""
    if ! have nvcc; then
        cuda_reason="nvcc is not on PATH"
    elif [ "$gpu_count" -eq 0 ]; then
        cuda_reason="no NVIDIA device reported by nvidia-smi"
    fi

    if [ -n "$cuda_reason" ]; then
        add_result "CUDA build" PENDING "not run: $cuda_reason" "not run" "-"
        add_result "CUDA tests" PENDING "not run: $cuda_reason" "$cuda_test_cmd" "-"
        add_result "CUDA skips" PENDING "not run: $cuda_reason" "$cuda_test_cmd" "-"
        add_result "Pool repeat" PENDING "not run: $cuda_reason" \
            "timeout $pool_repeat_timeout $pool_repeat_cmd" "-"
        add_result "Sanitizer (pool-ladder-probe)" PENDING "not run: $cuda_reason" "$san_pool_cmd" "-"
        add_result "Sanitizer (chunk-publication)" PENDING "not run: $cuda_reason" "$san_chunk_cmd" "-"
        add_result "Example (cuda)" PENDING "not run: $cuda_reason" "not run" "-"
        any_pending=1
    elif configure_and_build cuda ON CUDA; then
        run_tests cuda "$artifact_dir/build-cuda" "$cuda_timeout" CUDA || true
        run_example_config cuda "$artifact_dir/build-cuda" cuda
        run_pool_repeat "$artifact_dir/build-cuda" || true

        cuda_test_binary=$(find_cuda_test_binary "$artifact_dir/build-cuda")
        if [ "$sanitize" = off ]; then
            add_result "Sanitizer (pool-ladder-probe)" PENDING "disabled by --no-sanitize" \
                "$san_pool_cmd" "-"
            add_result "Sanitizer (chunk-publication)" PENDING "disabled by --no-sanitize" \
                "$san_chunk_cmd" "-"
            any_pending=1
        elif ! have compute-sanitizer; then
            add_result "Sanitizer (pool-ladder-probe)" PENDING "compute-sanitizer is not on PATH" \
                "$san_pool_cmd" "-"
            add_result "Sanitizer (chunk-publication)" PENDING "compute-sanitizer is not on PATH" \
                "$san_chunk_cmd" "-"
            any_pending=1
        else
            run_sanitizer_group pool-ladder-probe \
                '[graph][pool][ladder],[graph][pool][probe]' \
                'pool gpu correctness ladder|probe 1: warp broadcast' \
                "$cuda_test_binary" || true
            run_sanitizer_group chunk-publication \
                '[chunk_assembly][gpu]' \
                'chunk_assembly publication guard' \
                "$cuda_test_binary" || true
        fi
    else
        add_result "CUDA tests" FAIL "not run after a CUDA build failure" \
            "$cuda_test_cmd" "\$ARTIFACT/build-cuda.log"
        add_result "CUDA skips" FAIL "not run after a CUDA build failure" \
            "$cuda_test_cmd" "\$ARTIFACT/build-cuda.log"
        add_result "Pool repeat" FAIL "not run after a CUDA build failure" \
            "timeout $pool_repeat_timeout $pool_repeat_cmd" "\$ARTIFACT/build-cuda.log"
        add_result "Sanitizer (pool-ladder-probe)" FAIL "not run after a CUDA build failure" \
            "$san_pool_cmd" "\$ARTIFACT/build-cuda.log"
        add_result "Sanitizer (chunk-publication)" FAIL "not run after a CUDA build failure" \
            "$san_chunk_cmd" "\$ARTIFACT/build-cuda.log"
        add_result "Example (cuda)" NOT-RUN "not run after a CUDA build failure" \
            "graph_particle_feedback --backend cuda" "\$ARTIFACT/example-summary.md"
        any_fail=1
    fi
else
    add_result "CUDA build" PENDING "skipped by --skip-cuda" "not run" "-"
    add_result "CUDA tests" PENDING "skipped by --skip-cuda" "not run" "-"
    add_result "CUDA skips" PENDING "skipped by --skip-cuda" "not run" "-"
    add_result "Pool repeat" PENDING "skipped by --skip-cuda" "not run" "-"
    add_result "Sanitizer (pool-ladder-probe)" PENDING "skipped by --skip-cuda" "not run" "-"
    add_result "Sanitizer (chunk-publication)" PENDING "skipped by --skip-cuda" "not run" "-"
    add_result "Example (cuda)" PENDING "skipped by --skip-cuda" "not run" "-"
    any_pending=1
fi

run_repo_gate Format check-format.sh "$artifact_dir/format.log"
run_repo_gate REUSE reuse-lint.sh "$artifact_dir/reuse.log"

# Diagnostics are collected from the JUnit system-out blocks and the configure
# logs. Only lines that carry geometry, occupancy, residency, register, or
# transport information are kept.
for cfg in cpu cuda; do
    diag="$artifact_dir/diagnostics-$cfg.txt"
    : >"$diag"
    if [ -f "$artifact_dir/junit-$cfg.xml" ]; then
        extract_diagnostics "$artifact_dir/junit-$cfg.xml" "$diag"
    fi
    if [ -f "$artifact_dir/configure-$cfg.log" ]; then
        grep -E 'captured persistent-kernel register usage|register count uses|CUDA is disabled' \
            "$artifact_dir/configure-$cfg.log" >>"$diag" 2>/dev/null || true
    fi
    if [ -s "$diag" ]; then
        {
            printf '### %s\n\n```\n' "$cfg"
            cat "$diag"
            printf '```\n\n'
        } >>"$diagnostics"
    fi
done
if [ ! -s "$diagnostics" ]; then
    printf 'No geometry, occupancy, residency, register, or transport lines were reported.\n' \
        >>"$diagnostics"
fi
if [ ! -s "$details" ]; then
    printf 'No configuration was executed.\n' >>"$details"
fi
if [ ! -s "$example_summary" ]; then
    printf 'No example run was executed.\n' >>"$example_summary"
fi

{
    printf '| Configuration | Registered cases | Repeat | Planned case runs | Executed case runs |\n'
    printf '| --- | --- | --- | --- | --- |\n'
    for count_cfg in cpu cuda; do
        count_file="$artifact_dir/counts-$count_cfg.txt"
        [ -f "$count_file" ] || continue
        count_registered=$(sed -n 's/^registered=//p' "$count_file")
        count_repeat=$(sed -n 's/^repeat=//p' "$count_file")
        count_planned=$(sed -n 's/^planned_case_runs=//p' "$count_file")
        count_executed=$(sed -n 's/^executed_case_runs=//p' "$count_file")
        printf '| %s | %s | %s | %s | %s |\n' \
            "$count_cfg" "$count_registered" "$count_repeat" "$count_planned" "$count_executed"
    done
} >"$test_counts"

# ------------------------------------------------------------- verdict ---

if [ "$any_fail" -ne 0 ]; then
    overall=FAIL
    status=1
    verdict_text="**FAIL.** At least one executed gate failed. See the gate table and the per-gate logs."
elif [ "$any_pending" -ne 0 ]; then
    overall=PENDING
    status=2
    verdict_text="**PENDING.** No executed gate failed, but at least one gate did not run. The hardware gate is not claimed."
else
    overall=PASS
    status=0
    verdict_text="**PASS.** Every executed gate passed."
fi

if [ "$overall" = PASS ] && grep -q 'NOT-BUILT' "$results"; then
    verdict_text="$verdict_text The example gate is NOT-BUILT and is not claimed as verified."
fi

if [ "$allow_pending" -eq 1 ] && [ "$status" -eq 2 ]; then
    status=0
fi

harness_cmd="scripts/hal-acceptance.sh --artifact-dir \$ARTIFACT --repeat $repeat --pool-repeat $pool_repeat"
if [ "$sanitize" = off ]; then
    harness_cmd="$harness_cmd --no-sanitize"
else
    harness_cmd="$harness_cmd --sanitize"
fi
add_result Overall "$overall" "$verdict_text" "$harness_cmd" "\$ARTIFACT/hal-acceptance.md"
{
    printf '%s\n' "$verdict_text"
    printf '\nCode commit: `%s` (dirty: %s). Environment: `environment.txt` in the artifact directory.\n' \
        "$git_commit" "$git_dirty"
    printf 'Exit status: %s (`0` pass, `1` fail, `2` pending).\n' "$status"
} >"$verdict_file"

tab=$(printf '\t')
{
    printf '| Gate | Result | Detail | Command | Artifact |\n'
    printf '| --- | --- | --- | --- | --- |\n'
    while IFS="$tab" read -r gate result detail command artifact; do
        printf '| %s | %s | %s | %s | %s |\n' \
            "$gate" "$result" \
            "$(printf '%s' "$detail" | sed 's/|/\\|/g')" \
            "$(printf '%s' "$command" | sed 's/|/\\|/g')" \
            "$artifact"
    done <"$results"
} >"$summary"

# The template has a single ENVIRONMENT placeholder; fold the raw dumps in.
{
    cat "$env_file"
    printf '\n# nvidia-smi -L\n'
    cat "$nvidia_smi_list"
    printf '# nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv\n'
    cat "$nvidia_smi_query"
} >"$artifact_dir/environment-full.txt"

{
    for cfg in cpu cuda; do
        if [ -f "$artifact_dir/counts-$cfg.txt" ]; then
            printf -- '- %s build directory: `$ARTIFACT/build-%s`\n' "$cfg" "$cfg"
            printf -- '- %s configure log: `$ARTIFACT/configure-%s.log`\n' "$cfg" "$cfg"
            printf -- '- %s build log: `$ARTIFACT/build-%s.log`\n' "$cfg" "$cfg"
        fi
    done
    printf -- '- Commands executed: `$ARTIFACT/commands.txt`\n'
    printf -- '- Per-config counts: `$ARTIFACT/counts-<cfg>.txt`\n'
    printf -- '- JUnit reports: `$ARTIFACT/junit-<cfg>.xml`\n'
    printf -- '- CTest logs: `$ARTIFACT/ctest-<cfg>.log`\n'
    printf -- '- Registered test lists: `$ARTIFACT/ctest-list-<cfg>.log`\n'
    printf -- '- Test counts: `$ARTIFACT/test-counts.md`\n'
    printf -- '- Pool repeat log: `$ARTIFACT/pool-repeat.log`\n'
    printf -- '- Sanitizer logs: `$ARTIFACT/sanitizer-memcheck-<group>.log`\n'
    printf -- '- Sanitizer filter lists: `$ARTIFACT/sanitizer-filter-<group>-list.log`\n'
    printf -- '- Example outputs: `$ARTIFACT/example-<cfg>-<case>.txt`\n'
    printf -- '- Environment: `$ARTIFACT/environment.txt`\n'
} >"$artifacts_list"

# --------------------------------------------------------------- record ---

awk \
    -v f_summary="$summary" \
    -v f_env="$artifact_dir/environment-full.txt" \
    -v f_commands="$commands" \
    -v f_details="$details" \
    -v f_example="$example_summary" \
    -v f_counts="$test_counts" \
    -v f_diag="$diagnostics" \
    -v f_artifacts="$artifacts_list" \
    -v f_verdict="$verdict_file" \
    -v generated="$generated_at" '
    function emit(path,    line) {
        while ((getline line < path) > 0) print line
        close(path)
    }
    /^\{\{GENERATED_AT\}\}$/ { print generated; next }
    /^\{\{VERDICT\}\}$/ { emit(f_verdict); next }
    /^\{\{ENVIRONMENT\}\}$/ { emit(f_env); next }
    /^\{\{SUMMARY_TABLE\}\}$/ { emit(f_summary); next }
    /^\{\{COMMANDS\}\}$/ { emit(f_commands); next }
    /^\{\{DETAILS\}\}$/ { emit(f_details); next }
    /^\{\{EXAMPLE\}\}$/ { emit(f_example); next }
    /^\{\{TEST_COUNTS\}\}$/ { emit(f_counts); next }
    /^\{\{DIAGNOSTICS\}\}$/ { emit(f_diag); next }
    /^\{\{ARTIFACTS\}\}$/ { emit(f_artifacts); next }
    { print }
' "$template" >"$artifact_dir/hal-acceptance.md"

if [ "$write_doc" -eq 1 ]; then
    cp "$artifact_dir/hal-acceptance.md" "$repo_root/docs/hal-acceptance.md"
    printf 'HAL acceptance: wrote %s\n' "$repo_root/docs/hal-acceptance.md"
fi

printf '\nHAL acceptance summary\n\n'
cat "$summary"
printf '\nRecord: %s\n' "$artifact_dir/hal-acceptance.md"
printf 'Artifacts: %s\n' "$artifact_dir"

exit "$status"
