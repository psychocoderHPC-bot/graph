#!/bin/sh
# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0
#
# Local equivalent of the GitLab `format` job: run the pinned clang-format in
# check mode over every tracked project C++/CUDA source and header file.
#
# The repository .clang-format is the alpaka file from the pinned revision and
# requires clang-format 20. Install the matching formatter with
#   pip install "clang-format==20.1.8"
#
# Usage: scripts/check-format.sh
# Exit status: 0 when all files match, 1 on a formatting difference or a
# formatter version mismatch, 127 when clang-format is missing.

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

required_major=20

if ! command -v clang-format >/dev/null 2>&1; then
    echo "error: clang-format not found; install clang-format ${required_major}" >&2
    echo "       pip install 'clang-format==20.1.8'" >&2
    exit 127
fi

version_output=$(clang-format --version)
version_major=$(printf '%s\n' "$version_output" | sed -n 's/.*version \([0-9][0-9]*\).*/\1/p')
if [ "$version_major" != "$required_major" ]; then
    echo "error: found ${version_output}" >&2
    echo "       .clang-format is pinned to clang-format ${required_major}" >&2
    echo "       install it with: pip install 'clang-format==20.1.8'" >&2
    exit 1
fi

file_list=$(mktemp "${TMPDIR:-/tmp}/graph-format-files.XXXXXX")
trap 'rm -f "$file_list"' EXIT HUP INT TERM

# Tracked project files only. Build outputs, fetched dependencies, and
# third-party trees are excluded explicitly even if they were ever tracked.
git ls-files -z -- \
    '*.h' '*.hh' '*.hpp' '*.hxx' '*.inl' \
    '*.c' '*.cc' '*.cpp' '*.cxx' '*.cu' '*.cuh' \
    ':(exclude,glob)build*/**' \
    ':(exclude,glob)**/build*/**' \
    ':(exclude,glob)_deps/**' \
    ':(exclude,glob)**/_deps/**' \
    ':(exclude,glob)third_party/**' \
    ':(exclude,glob)**/third_party/**' \
    ':(exclude,glob)external/**' \
    ':(exclude,glob)**/external/**' \
    >"$file_list"

file_count=$(tr -cd '\000' <"$file_list" | wc -c)
file_count=$(printf '%s' "$file_count" | tr -d ' ')
if [ "$file_count" -eq 0 ]; then
    echo "check-format: no tracked C++/CUDA sources; nothing to check"
    exit 0
fi

echo "check-format: ${version_output}; checking ${file_count} tracked files"
xargs -0 clang-format --dry-run --Werror <"$file_list"
