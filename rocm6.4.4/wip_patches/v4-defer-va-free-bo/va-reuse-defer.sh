#!/bin/sh
# Apply va-reuse-defer.patch (see that file for the full WHY/WHAT) via
# `git apply`, then verify the marker actually landed.
set -eu

SRC="${1:-/rocr-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/va-reuse-defer.patch"
FILE="$SRC/libhsakmt/src/fmm.c"
FLAGFILE="$SRC/runtime/hsa-runtime/core/util/flag.h"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -e "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'GFX803_VA_REUSE_DEFER_PATCH' "$FILE" || true)
if [ "$count" -lt 1 ]; then
    echo "FATAL: marker not found in $FILE after git apply reported success" >&2
    exit 1
fi
grep -q 'disable_fragment_alloc_ = (var == "0") ? false : true' "$FLAGFILE" || {
    echo "FATAL: fragment-allocator default flip not found in $FLAGFILE" >&2
    exit 1
}
echo "VA-reuse defer patch applied and verified in $FILE and $FLAGFILE"
