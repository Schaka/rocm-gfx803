#!/bin/sh
# Apply va-reuse-defer-mapping.patch (see that file for the full WHY/WHAT).
# Must run AFTER va-reuse-defer.sh on the same checkout -- this patch's
# context is the state va-reuse-defer.patch produces (the park branch and
# the fmm_defer_release forward declaration).
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/va-reuse-defer-mapping.patch"
FILE="$SRC/projects/rocr-runtime/libhsakmt/src/fmm.c"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing -- is va-reuse-defer applied?" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q '_fmm_map_to_gpu(ctx, aperture, object->start' "$FILE" || {
    echo "FATAL: re-map marker not found in $FILE after git apply reported success" >&2
    exit 1
}
echo "VA-reuse defer re-map patch applied and verified in $FILE"