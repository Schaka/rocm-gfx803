#!/bin/sh
# See winograd-fused-conv-miscompute.patch (copied unchanged from the
# 6.4.4-era rocm6.4.4/patches/miopen/) for the full WHY. Re-verified applying
# cleanly against the pinned 7.14 source (only a small line-number offset,
# no content drift) before being carried over here.
#
# Uses `patch`, not `git apply` -- see
# ../rocblas/small-gemm-assembly-miscompute.sh for why (git-apply-specific
# "Skipped patch" quirk on this git version, not a defect in the patch).
set -eu
SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/winograd-fused-conv-miscompute.patch"
FILE="$SRC/src/solver/conv_bin_winoRxS_fused.cpp"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE does not exist -- upstream moved/removed this solver" >&2; exit 1; }
if grep -q 'WINOGRAD_FUSED_CBA_MISCOMPUTE_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi
patch -p1 -d "$SRC" --verbose < "$PATCH"
count=$(grep -c 'WINOGRAD_FUSED_CBA_MISCOMPUTE_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi
echo "Winograd fused-CBA miscompute patch applied and verified in $FILE"
