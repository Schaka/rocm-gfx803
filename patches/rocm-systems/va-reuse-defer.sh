#!/bin/sh
# Apply va-reuse-defer.patch with `git apply`, then make sure that the hunks
# landed. That patch file gives the reason and the change. SRC is the
# rocm-systems monorepo root, a real git checkout, so `git apply` works here.
# The rocblas, miopen, migraphx and pytorch patches in this tree target sparse
# checkouts and use `patch -p1` instead (see small-gemm-assembly-miscompute.sh).
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/va-reuse-defer.patch"
FILE="$SRC/projects/rocr-runtime/libhsakmt/src/fmm.c"
FLAGFILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

# Marker check, not `apply --check --reverse`: va-reuse-defer-mapping/noremap
# touch the same file after this one, so a clean reverse only holds on a tree
# that contains nothing past this patch.
if grep -q 'GFX803_VA_REUSE_DEFER_PATCH' "$FILE" \
   && grep -q 'disable_fragment_alloc_ = (var == "0") ? false : true' "$FLAGFILE"; then
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
