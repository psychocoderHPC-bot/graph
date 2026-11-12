<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# Build and test

These instructions reproduce the local validation on a machine like `hal`:
a Linux host with a C++20 compiler and, for the CUDA configuration, an NVIDIA
GPU with a matching CUDA toolkit. The commands match the jobs in
`.gitlab-ci.yml` so local results and CI results are comparable.

## Pinned dependencies

| Component | Pin | Notes |
| --- | --- | --- |
| alpaka | `4780dc786a2eb1a5452a1c66d94a8b442d411ab8` | `dev_v3.x`; architecture, code style, and the `.clang-format` reference |
| Catch2 | `v3.15.0` | the revision the pinned alpaka test tree uses |
| clang-format | `20.1.8` | required by the pinned `.clang-format` |
| reuse | `6.3.0` | REUSE lint tool |
| CMake | 3.25 or newer | `cmake_minimum_required(VERSION 3.25)` |
| Ninja | any recent version | the generator used by CI |

alpaka and Catch2 are fetched by CMake through `FetchContent` at the pins
above. The configure step therefore needs network access to GitHub on a clean
machine. Use a local alpaka checkout when the network is unavailable or when
developing against a modified alpaka tree.

## Host prerequisites

```sh
# Debian/Ubuntu example, CPU build
sudo apt install build-essential cmake ninja-build git

# CUDA build: install the vendor toolkit so nvcc is on PATH, or load the
# site's CUDA module. The pinned alpaka revision supports CUDA 12.x.
nvcc --version
nvidia-smi
```

Optional OpenMP: `GRAPH_ENABLE_OPENMP` defaults to `ON` when CMake finds OpenMP
C++ support and to `OFF` otherwise. Force it with
`-DGRAPH_ENABLE_OPENMP=ON` or `-DGRAPH_ENABLE_OPENMP=OFF`.

## Environment scrubbing on hal

hal loads a oneAPI profile that exports `LD_LIBRARY_PATH` and
`CMAKE_PREFIX_PATH`. Those entries can select a different C++ runtime or a
different CMake package set than the intended system toolchain. Scrub both
variables for every configure, build, and test command:

```sh
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH <command>
```

The HAL CI jobs use the same prefix. If a local command behaves differently
from CI, check these two variables first.

## CPU configuration

```sh
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH \
  cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRAPH_ENABLE_CUDA=OFF
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH cmake --build build-cpu --parallel
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH \
  ctest --test-dir build-cpu --output-on-failure --timeout 600 --no-tests=error
```

The CPU configuration does not enable the CUDA language and does not require a
CUDA toolkit. `ctest` runs the Catch2 unit tests and the CTest entries the
project registers, including the particle-feedback example once it exists.

To keep a JUnit report like CI:

```sh
ctest --test-dir build-cpu --output-on-failure --output-junit junit.xml \
  --timeout 600 --no-tests=error
```

## CUDA configuration

```sh
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH \
  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH cmake --build build-cuda --parallel
CUDA_VISIBLE_DEVICES=0 \
  env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH \
  ctest --test-dir build-cuda --output-on-failure --timeout 900 --no-tests=error
```

`GRAPH_ENABLE_CUDA=ON` enables the CUDA language and relocatable device code
(`alpaka_RELOCATABLE_DEVICE_CODE=ON`), which the indirect device-function calls
require. `CMAKE_CUDA_ARCHITECTURES=80` builds sm_80 code, which the A30
(compute capability 8.6) runs, and matches CI. When the variable is unset, the configure step
reads the compute capability of the first visible GPU through `nvidia-smi` and
falls back to `80`.

`CUDA_VISIBLE_DEVICES=0` makes the run use one device only. On a shared machine
this prevents the test suite from claiming every visible GPU and makes the
device choice reproducible.

Do not run the CUDA configuration in the lint container; it needs the toolkit,
the driver, and a physical GPU.

## Using a local alpaka checkout

```sh
env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH \
  cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPH_ENABLE_CUDA=OFF -DGRAPH_ALPAKA_SOURCE_DIR=/path/to/alpaka
```

The path must contain alpaka's `CMakeLists.txt`. The configure step fails
otherwise. `GRAPH_ALPAKA_GIT_TAG` overrides the fetched revision and must name a
commit, not a branch. For reproducible results keep the default pin.

## Formatting and licensing checks

```sh
pip install "clang-format==20.1.8"
scripts/check-format.sh

pip install "reuse==6.3.0"
scripts/reuse-lint.sh
```

Both scripts are the local equivalents of the `format` and `reuse` CI jobs and
return non-zero on failure. See [`ci.md`](ci.md) for the pipeline description.

## Records to keep for a hardware validation

For an acceptance run record the commit, the alpaka revision, the compiler and
CUDA toolkit versions, driver version, CPU and GPU model, the exact commands,
test counts, and the result summary. `hal-diagnostics.txt` from the HAL jobs
collects most of the environment facts. Skipped tests do not count as passing
hardware validation.
