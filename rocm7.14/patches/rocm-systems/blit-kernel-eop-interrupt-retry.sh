#!/bin/sh
# Apply blit-kernel-eop-interrupt-retry.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunk actually landed.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/blit-kernel-eop-interrupt-retry.patch"
AQL_FILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"
BLIT_FILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_kernel.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if ! grep -q "Gfx8EopMitigationMaxAttempts" "$AQL_FILE" || ! grep -q "BlitKernel::RecreateQueue" "$BLIT_FILE"; then
    echo "FATAL: expected markers not found after git apply reported success" >&2
    exit 1
fi
echo "blit-kernel-eop-interrupt-retry patch applied and verified"
