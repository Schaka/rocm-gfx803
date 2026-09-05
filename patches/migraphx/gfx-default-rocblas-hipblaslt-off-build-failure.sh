#!/bin/sh
# Apply gfx-default-rocblas-hipblaslt-off-build-failure.patch. That file gives the
# reason and the change. This driver uses `patch` and not `git apply`, for the
# reason stated in ../rocblas/small-gemm-assembly-miscompute.sh.
set -eu
SRC="${1:-/migraphx-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx-default-rocblas-hipblaslt-off-build-failure.patch"
HPP="$SRC/src/targets/gpu/include/migraphx/gpu/device_name.hpp"
CPP="$SRC/src/targets/gpu/device_name.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$CPP" ] || { echo "FATAL: $CPP does not exist -- upstream moved this file" >&2; exit 1; }

if grep -q 'GFX_DEFAULT_ROCBLAS_HIPBLASLT_OFF_PATCH' "$CPP"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --verbose < "$PATCH"

if grep -A2 '^#if MIGRAPHX_USE_HIPBLASLT$' "$HPP" | grep -q 'gfx_default_rocblas'; then
    echo "FATAL: gfx_default_rocblas() declaration still gated behind MIGRAPHX_USE_HIPBLASLT in $HPP" >&2
    exit 1
fi
if ! grep -q 'GFX_DEFAULT_ROCBLAS_HIPBLASLT_OFF_PATCH' "$CPP"; then
    echo "FATAL: marker not found in $CPP after patch reported success" >&2
    exit 1
fi
echo "gfx_default_rocblas hipBLASLt-off build-failure patch applied and verified in $HPP and $CPP"
