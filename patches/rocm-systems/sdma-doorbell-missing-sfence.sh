#!/bin/sh
# Apply sdma-doorbell-missing-sfence.patch with `git apply`, then make sure that
# the hunk landed. That patch file gives the reason and the change.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/sdma-doorbell-missing-sfence.patch"
SDMA_FILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if ! grep -q "_mm_sfence();" "$SDMA_FILE" || ! grep -q "\*queue_doorbell_ = write_index;" "$SDMA_FILE"; then
    echo "FATAL: _mm_sfence() not found before doorbell write in $SDMA_FILE after git apply reported success" >&2
    exit 1
fi
echo "sdma-doorbell-missing-sfence patch applied and verified in $SDMA_FILE"
