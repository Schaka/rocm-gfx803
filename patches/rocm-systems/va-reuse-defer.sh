set -eu

# See va-reuse-defer.patch (re-diffed from the shipped
# rocm6.4.4/patches/rocr/va-reuse-defer.patch) for the full WHY/WHAT and the
# 7.14 re-diff notes. Uses `git apply` -- SRC here is the rocm-systems
# monorepo root, a real git checkout, unlike the rocblas/miopen/migraphx/
# pytorch patches in this tree (see small-gemm-assembly-miscompute.sh for
# why those use `patch` instead).

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/va-reuse-defer.patch"
FILE="$SRC/projects/rocr-runtime/libhsakmt/src/fmm.c"
FLAGFILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

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
