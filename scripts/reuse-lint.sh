#!/bin/sh
# SPDX-FileCopyrightText: 2026 René Widera and contributors
# SPDX-License-Identifier: MPL-2.0
#
# Local equivalent of the GitLab `reuse` job: run the pinned REUSE tool's lint
# over the repository. Install the matching tool with
#   pip install "reuse==6.3.0"
#
# Usage: scripts/reuse-lint.sh
# Exit status: 0 when the repository is compliant, non-zero on lint errors,
# 127 when the reuse tool is missing.

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if ! command -v reuse >/dev/null 2>&1; then
    echo "error: reuse not found; install the pinned tool" >&2
    echo "       pip install 'reuse==6.3.0'" >&2
    exit 127
fi

echo "reuse-lint: $(reuse --version 2>&1 | sed -n '1p')"
exec reuse lint
