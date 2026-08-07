#!/bin/sh
# Apply conv-direct-fwd-grouped-oob.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunk actually landed.
# `git apply` on a tag that will never move again (rocm-6.4.4, gfx803 is
# abandoned upstream) either matches exactly or fails loud -- this wrapper
# exists to catch operator error (wrong clone, already-patched tree,
# patch/source mismatch), not upstream drift.
set -eu

SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/conv-direct-fwd-grouped-oob.patch"
FILE="$SRC/src/solver/conv/conv_ocl_dir2Dfwd.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'CONV_DIRECT_FWD_GROUPED_OOB_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after git apply reported success" >&2
    exit 1
fi
echo "Grouped-conv OOB patch applied and verified in $FILE"
