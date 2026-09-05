#!/bin/sh
# Apply va-reuse-defer-mapping.patch with `git apply`, then make sure that the hunk
# landed. That patch file gives the reason and the change.
#
# Run this after va-reuse-defer.sh on the same checkout. This patch's context is the
# state that va-reuse-defer.patch produces, which is the park branch and the
# fmm_defer_release forward declaration.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/va-reuse-defer-mapping.patch"
FILE="$SRC/projects/rocr-runtime/libhsakmt/src/fmm.c"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing -- is va-reuse-defer applied?" >&2; exit 1; }

# Marker check, not `apply --check --reverse`: noremap/fmm-keep-userptr-map
# touch the same file after this one (noremap even removes the re-map CALL
# this patch adds), so detect on this patch's forward declaration -- the
# second occurrence of the signature line (pristine has only the definition
# below), untouched by every later patch in the chain.
if [ "$(grep -c '^static HSAKMT_STATUS _fmm_map_to_gpu(HsaKFDContext \*ctx,$' "$FILE")" -ge 2 ]; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q '_fmm_map_to_gpu(ctx, aperture, object->start' "$FILE" || {
    echo "FATAL: re-map marker not found in $FILE after git apply reported success" >&2
    exit 1
}
echo "VA-reuse defer re-map patch applied and verified in $FILE"