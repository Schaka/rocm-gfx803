#!/usr/bin/env bash
# Apply the gfx803 DPP-broadcast warpReduce fix to a triton checkout.
#
# Usage:
#   ./apply-gfx803-dpp-broadcast-warpreduce.sh /path/to/triton
#
# Verifies its own result: greps the patched source for the GCN3 branch
# and fails loudly if missing. Requires gfx803-isa-family.patch already
# applied (this patch's context assumes ISAFamily::GCN3 exists).

set -euo pipefail

TRITON_DIR="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-dpp-broadcast-warpreduce.patch"
TARGET="$TRITON_DIR/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp"

if [[ -z "$TRITON_DIR" || ! -f "$TARGET" ]]; then
    echo "usage: $0 /path/to/triton (must contain $TARGET)" >&2
    exit 1
fi

git -C "$TRITON_DIR" apply --check "$PATCH"
git -C "$TRITON_DIR" apply "$PATCH"

if grep -q "getISAFamily() == ISAFamily::GCN3) {" "$TARGET"; then
    echo "gfx803-dpp-broadcast-warpreduce.patch applied and verified."
else
    echo "ERROR: patch applied but GCN3 DPP-broadcast branch not found" >&2
    exit 1
fi
