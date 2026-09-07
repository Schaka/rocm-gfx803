#!/bin/sh
# Apply the gfx803 patches and build MIGraphX from source into /opt/rocm.
#
# Every patch file carries its own WHY header.
set -eu

SRC=/migraphx-src
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
REF="${MIGRAPHX_REF:?MIGRAPHX_REF is required}"
. /scripts/lib/build-jobs.sh

sh /patches/migraphx/gfx-default-rocblas-hipblaslt-off-build-failure.sh "$SRC"
sh /patches/migraphx/mlir-stub-missing-symbols.sh "$SRC"

# Neither composable_kernel nor rocMLIR has ever supported gfx8, and rbuild would
# otherwise spend hours building both.
sed -i '/composable_kernel/d; /rocMLIR/d' "$SRC/requirements.txt"
if grep -q 'composable_kernel\|rocMLIR' "$SRC/requirements.txt"; then
    echo "FATAL: requirements.txt still lists composable_kernel or rocMLIR." >&2
    exit 1
fi

jobs="$(resolve_build_jobs)"
echo "MIGraphX build: arch $ARCH, $jobs parallel jobs"
cd "$SRC"
ulimit -s unlimited

# PYTHON_DISABLE_VERSIONS and the interpreter directory on PATH both exist for
# one reason: MIGraphX's cmake/PythonModules.cmake ignores -DPython3_EXECUTABLE
# for its python-module target. find_python(version) does a bare find_program
# search for python<version>-config over a hardcoded 3.6 to 3.14 list, and it
# silently skips any version whose -config script is missing. A venv carries no
# python3.12-config, so without the real interpreter's directory on PATH,
# find_python(3.12) fails while find_python(3.14) succeeds against the base
# image, and the only migraphx.so built is cpython-314 tagged and unimportable
# from the 3.12 venv the final image uses.
py312_bin="$(dirname "$(uv python find 3.12)")"
CMAKE_BUILD_PARALLEL_LEVEL="$jobs" \
PATH="${py312_bin}:/rbuild-venv/bin:$PATH" \
/rbuild-venv/bin/rbuild build -d /migraphx-deps -B build -G Ninja \
    --cxx=/opt/rocm/llvm/bin/clang++ --cc=/opt/rocm/llvm/bin/clang \
    "-DGPU_TARGETS=$ARCH" \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release \
    -DMIGRAPHX_ENABLE_PYTHON=On \
    -DPython3_EXECUTABLE=/rbuild-venv/bin/python3 \
    -DPYTHON_DISABLE_VERSIONS=3.14 \
    -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off \
    -DMIGRAPHX_ENABLE_MLIR=Off \
    -DMIGRAPHX_USE_HIPBLASLT=Off \
    -DMIGRAPHX_USE_ROCBLAS=On \
    -DMIGRAPHX_USE_MIOPEN=On \
    -DBUILD_TESTING=Off \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS=-I/patches/migraphx/mlir-stub \
    -DCMAKE_CXX_FLAGS=-I/patches/migraphx/mlir-stub \
    -T install
rm -rf /migraphx-deps

echo "$REF $(git -C "$SRC" rev-parse HEAD)" > /opt/rocm/migraphx-version.txt

# A module with the wrong ABI tag is unimportable, and without this gate the
# failure surfaces two jobs later as an ImportError in the final stage.
if ! find /opt/rocm -iname "migraphx.cpython-312-*.so" | grep -q .; then
    echo "FATAL: no migraphx.cpython-312-*.so under /opt/rocm, so the python module was built for the wrong interpreter." >&2
    find /opt/rocm -iname "migraphx.cpython-*.so" >&2
    exit 1
fi
echo "OK: the migraphx python module is built for cpython-312."
