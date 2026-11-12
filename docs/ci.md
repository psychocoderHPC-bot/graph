<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# Continuous integration

The repository ships a root `.gitlab-ci.yml`. GitHub stays the source
repository; GitLab runs the checks. This page records how a pipeline is
triggered, which commit it validates, what each job proves, and what is still
missing before the hardware gate is real.

## Current status

The workflow is committed. The GitLab project, the HAL runners, and the CUDA
image are not provisioned yet, so no pipeline has run against this repository.
Until they are, the HAL jobs stay queued and the hardware gate is visibly
unfulfilled. The committed jobs run real commands with no no-op fallback.

### Pending operator actions

1. Connect a GitLab CI project to
   `https://github.com/psychocoderHPC-bot/graph` with GitLab's GitHub
   integration, or with the established mirror plus trigger workflow. Enable
   commit status reporting so results appear on the GitHub commit.
2. Register the HAL CPU runner with tags `hal`, `cpuonly`, `x86_64`.
3. Register the HAL NVIDIA runner with tags `hal`, `cuda`, `nvidia`, with the
   A30 exposed as CUDA device 0.
4. Confirm or replace the placeholder CUDA image
   `nvidia/cuda:12.6.3-devel-ubuntu24.04`, and pin its digest.
5. Enable shared runners for `format`, `reuse`, and `test:cpu`, or give those
   jobs the HAL CPU tags. Without one of the two, the pipeline reports the
   three jobs as pending as well.
6. Mark `format`, `reuse`, `test:cpu`, and `test:hal:nvidia` as required checks
   in the GitHub branch protection once the integration reports statuses.

The tag names and the image are placeholders. When the operator registers
different tags, change the `tags:` blocks of `test:hal:cpu` and
`test:hal:nvidia` in `.gitlab-ci.yml` together with this list.

## Trigger and validated commit

The pipeline runs for:

- merge requests (`$CI_PIPELINE_SOURCE == "merge_request_event"`),
- pushes to the default branch (`main`),
- pushes to `warp-pool/*` branches,
- tags,
- manual and scheduled pipelines.

Every job starts with

```sh
test "$(git rev-parse HEAD)" = "$CI_COMMIT_SHA"
```

so the build fails when the checkout does not match the commit GitLab reports
for the pipeline. GitLab checks out `CI_COMMIT_SHA`, the head commit of the
source branch for a branch pipeline, and the source commit (or the merged-results
commit, when merged-results pipelines are enabled) for a merge request. The
explicit check keeps that contract visible instead of assumed.

`GIT_DEPTH` is 50. The runner fetches `CI_COMMIT_SHA` explicitly; the depth only
keeps the clone bounded. alpaka and Catch2 are fetched by CMake at their pinned
revisions, so build jobs need network access to GitHub.

## Visibility on the GitHub commit

With GitLab's GitHub integration (CI/CD for external repositories) the GitLab
project mirrors the GitHub repository and GitLab reports each pipeline result as
a commit status on the GitHub SHA. The status appears next to the commit and in
the pull request checks once the GitLab project is connected and the integration
has permission to write statuses.

If the instance does not offer that integration, the fallback is a GitHub
webhook or Actions job that starts a GitLab pipeline with
`CI_COMMIT_SHA=<pushed sha>` and a small status reporter that posts the pipeline
result back to the GitHub commit. Both routes must validate the pushed SHA, not
a locally rebuilt tree. Neither route is configured yet; action 1 above tracks
it.

## Jobs

| Job | Stage | What it checks |
| --- | --- | --- |
| `format` | `lint` | `scripts/check-format.sh` runs `clang-format --dry-run --Werror` over tracked project C++/CUDA sources and headers. The job installs `clang-format==20.1.8`, the version the pinned `.clang-format` requires. Build directories, fetched dependencies, and third-party trees are excluded. |
| `reuse` | `lint` | `scripts/reuse-lint.sh` runs `reuse lint` with `reuse==6.3.0` at the repository root and fails on any compliance error. |
| `test:cpu` | `test` | Configures a Release CPU build with Ninja and `-DGRAPH_ENABLE_CUDA=OFF`, builds it, and runs `ctest --output-on-failure --no-tests=error` with a per-test timeout. |
| `test:hal:cpu` | `test` | Same CPU build and test commands on the HAL CPU runner. Dumps `nvidia-smi -q`, `nvcc --version`, `cmake --version`, and `lscpu` to `hal-diagnostics.txt`. |
| `test:hal:nvidia` | `test` | Configures `-DGRAPH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`, builds, and runs the suite with `CUDA_VISIBLE_DEVICES=0`, so only the allocated A30 is visible and used. |

All test jobs scrub `LD_LIBRARY_PATH` and `CMAKE_PREFIX_PATH` on HAL
(`env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH`), because the oneAPI profile on
hal pollutes both and would otherwise change compiler and library selection.

The CPU job does not require a CUDA toolkit. The CUDA job is not run locally by
this repository's checks; it exists for the HAL NVIDIA runner.

## Artifacts and timeouts

Every test job keeps, on success and on failure:

- `junit.xml`, produced by `ctest --output-junit` and published as the GitLab
  JUnit report,
- `Testing/Temporary/LastTest.log` and `LastTestsFailed.log` from the build
  directory,
- `hal-diagnostics.txt` for the two HAL jobs.

Artifacts expire after one week. Job timeouts are finite: 20 minutes for the
lint jobs, 60 minutes for `test:cpu`, 90 minutes for `test:hal:cpu`, and 120
minutes for `test:hal:nvidia`. Test commands also carry their own timeouts
(600 s for CPU, 900 s for CUDA). `--no-tests=error` makes an empty test set a
failure, so a broken test registration cannot pass silently.

No tokens, credentials, or runner-specific paths are stored in the repository.
The HAL jobs contain no device addresses beyond `CUDA_VISIBLE_DEVICES=0`.

## Run the same checks locally

From the repository root:

```sh
# formatting; requires clang-format 20
pip install "clang-format==20.1.8"
scripts/check-format.sh

# REUSE compliance; requires the pinned tool
pip install "reuse==6.3.0"
scripts/reuse-lint.sh

# CPU build and tests, same commands as test:cpu
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRAPH_ENABLE_CUDA=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure --output-junit junit.xml \
  --timeout 600 --no-tests=error
```

The CUDA configuration is documented in [`build.md`](build.md); run it on a
machine with a supported NVIDIA GPU and toolkit, not in the lint container.

## Validating the workflow file locally

The file is parsed with a YAML loader before it is committed:

```sh
python3 -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))"
```

A full GitLab CI Lint requires the GitLab instance and is part of action 1. A
YAML parse proves syntax only; it does not prove that the runner tags exist.
