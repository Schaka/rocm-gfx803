#!/bin/sh
# Apply diag-graph-replay-spin.patch with `git apply`, then make sure that the
# hunks landed. This patch is a diagnostic and nothing else. See the patch file.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/diag-graph-replay-spin.patch"
TARGET_FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.hpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if grep -q "DIAG WaitForQueueSlot" "$TARGET_FILE" && grep -q "DIAG WaitForSignal" "$TARGET_FILE"; then
    echo "diag-graph-replay-spin.patch applied and verified."
else
    echo "ERROR: patch applied but diagnostic markers not found" >&2
    exit 1
fi
