#!/bin/sh
# Apply winograd-fused-conv-miscompute.patch. That file gives the full reason.
# The patch came unchanged from rocm6.4.4/patches/miopen/, and it was confirmed to
# apply cleanly against the pinned 7.14 source before it was carried here. Only
# line numbers moved, and no content drifted.
#
# This driver uses `patch` and not `git apply`. On this git version,
# `git apply --check` on a sparse checkout of a monorepo subdirectory reports
# success and changes nothing ("Skipped patch", exit 0). That is a quirk of
# `git apply` here, and not a defect in the patch. See
# ../rocblas/small-gemm-assembly-miscompute.sh.
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
