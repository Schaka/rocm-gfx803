#!/usr/bin/env bash
# Apply the gfx803 int8_gemm rocBLAS fallback to a PyTorch checkout.
#
# Usage:
#   ./apply-gfx803-int8-gemm-rocblas.sh /path/to/pytorch
#
# Verifies its own result: greps the patched file for the patch's marker and
# fails loudly if it is missing.

set -euo pipefail

SRC="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-int8-gemm-rocblas.patch"
TARGET="$SRC/aten/src/ATen/cuda/CUDABlas.cpp"
MARKER="GFX803_INT8_GEMM_ROCBLAS_PATCH"

if [[ -z "$SRC" || ! -f "$TARGET" ]]; then
    echo "usage: $0 /path/to/pytorch" >&2
    exit 1
fi

if grep -q "$MARKER" "$TARGET"; then
    echo "already patched in $TARGET, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --batch < "$PATCH"

if ! grep -q "$MARKER" "$TARGET"; then
    echo "FATAL: $MARKER marker not found after patch reported success" >&2
    exit 1
fi

echo "gfx803-int8-gemm-rocblas.patch applied and verified in $TARGET"
