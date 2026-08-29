set -eu

# See small-gemm-assembly-miscompute.patch for the full WHY (measurements,
# isolation method, 7.14 re-diff notes). Unchanged in substance from the
# 6.4.4-era rocm6.4.4/patches/rocblas/small-gemm-assembly-miscompute.sh;
# see ../../../KERNEL_BUGS.md for the original investigation.
#
# Uses `patch`, not the `git apply` every wrapper on the 6.4.4 line uses:
# on this rocBLAS source tree (rocm-libraries monorepo, git 2.55), `git
# apply` reproducibly reports "Skipped patch" and exits 0 without
# modifying anything, for a diff independently confirmed correct (content
# and placement checked by hand against the pinned source, and applying
# cleanly via `patch -p1`) -- a git-apply-specific quirk, not a defect in
# this patch.
#
# Which wrapper uses which, and why, in this directory:
#   `patch`     -- rocblas/, miopen/, migraphx/, pytorch/. The first two are
#                  checked out as a SUBDIRECTORY of a sparse-checkout
#                  monorepo, which is what git apply's path handling does
#                  not survive; the latter two follow suit so every diff in
#                  a re-diffed-for-7.14 tree is applied the same way.
#   `git apply` -- rocm-systems/ only, whose SRC is a real repo root and
#                  whose diffs are already root-relative (projects/...), so
#                  the 6.4.4 dialect works there unchanged.

SRC="${1:-/rocblas-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/small-gemm-assembly-miscompute.patch"
FILE="$SRC/library/src/tensile_host.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved rocBLAS's Tensile" >&2
    echo "       dispatch, so this patch would silently stop applying and the" >&2
    echo "       gfx803 small-GEMM miscompute would ship." >&2
    exit 1
}

if grep -q 'GFX803_SMALL_GEMM_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --verbose < "$PATCH"

count=$(grep -c 'GFX803_SMALL_GEMM_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi
echo "gfx803 small-GEMM assembly miscompute patch applied and verified in $FILE"
