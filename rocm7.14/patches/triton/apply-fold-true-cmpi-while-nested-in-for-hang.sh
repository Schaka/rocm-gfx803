#!/usr/bin/env bash
# Apply the while-nested-in-for GPU-hang fix to a triton checkout.
#
# Usage:
#   ./apply-fold-true-cmpi-while-nested-in-for-hang.sh /path/to/triton
#
# Verifies its own result: greps the patched source for the guard function
# and fails loudly if missing.

set -euo pipefail

TRITON_DIR="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/fold-true-cmpi-while-nested-in-for-hang.patch"
TARGET="$TRITON_DIR/third_party/amd/lib/Analysis/RangeAnalysis.cpp"

if [[ -z "$TRITON_DIR" || ! -f "$TARGET" ]]; then
    echo "usage: $0 /path/to/triton (must contain $TARGET)" >&2
    exit 1
fi

git -C "$TRITON_DIR" apply --check "$PATCH"
git -C "$TRITON_DIR" apply "$PATCH"

if grep -q "isOwnWhileConditionOperand" "$TARGET"; then
    echo "fold-true-cmpi-while-nested-in-for-hang.patch applied and verified."
else
    echo "ERROR: patch applied but isOwnWhileConditionOperand guard not found" >&2
    exit 1
fi
