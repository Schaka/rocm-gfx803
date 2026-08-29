#!/bin/sh
# See reduce-prod-wrong-identity.patch (copied unchanged from the
# 6.4.4-era rocm6.4.4/patches/miopen/) for the full WHY. Re-verified applying
# cleanly against the pinned 7.14 source (only a 1-line offset, no content
# drift) before being carried over here. Not gfx803-specific -- see the
# patch header.
#
# Uses `patch`, not `git apply` -- see
# ../rocblas/small-gemm-assembly-miscompute.sh for why (git-apply-specific
# "Skipped patch" quirk on this git version, not a defect in the patch).
set -eu
SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/reduce-prod-wrong-identity.patch"
FILE="$SRC/src/kernels/MIOpenReduceCalculation.cpp"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE does not exist -- upstream moved this kernel source" >&2; exit 1; }
if grep -q 'REDUCE_PROD_WRONG_IDENTITY_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi
patch -p1 -d "$SRC" --verbose < "$PATCH"
count=$(grep -c 'REDUCE_PROD_WRONG_IDENTITY_PATCH' "$FILE" || true)
if [ "$count" -ne 2 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi
echo "Reduce-Prod wrong-identity patch applied and verified in $FILE"
