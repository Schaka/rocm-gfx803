#!/bin/sh
# Apply restore-gfx803-winograd.patch. That file gives the full reason. It puts
# back the gfx803 Winograd solvers that the pinned MIOpen does not carry, and
# winograd-fused-conv-miscompute.sh must run after it, because that patch edits
# a file that this one creates.
#
# This driver uses `patch` and not `git apply`. On this git version,
# `git apply --check` on a sparse checkout of a monorepo subdirectory reports
# success and changes nothing ("Skipped patch", exit 0). That is a quirk of
# `git apply` here, and not a defect in the patch. See
# ../rocblas/small-gemm-assembly-miscompute.sh.
set -eu
SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/restore-gfx803-winograd.patch"
FILE="$SRC/src/solver/conv/conv_bin_wino3x3U.cpp"
FUSED="$SRC/src/solver/conv_bin_winoRxS_fused.cpp"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE does not exist -- upstream moved/removed this solver" >&2; exit 1; }
if grep -q 'RESTORE_GFX803_WINOGRAD_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi
# The patch creates this file. If it is already there, the tree is not the one
# the patch was cut against (upstream may carry the solver again), and creating
# it a second time would fail halfway through the other files.
if [ -e "$FUSED" ]; then
    echo "FATAL: $FUSED already exists -- re-check whether upstream restored the gfx803 Winograd solvers before re-cutting this patch" >&2
    exit 1
fi
patch -p1 -d "$SRC" --verbose < "$PATCH"
count=$(grep -c 'RESTORE_GFX803_WINOGRAD_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi
[ -f "$FUSED" ] || { echo "FATAL: $FUSED missing after patch reported success" >&2; exit 1; }
grep -q 'ConvBinWinogradRxSFused' "$SRC/src/fusion.cpp" || {
    echo "FATAL: ConvBinWinogradRxSFused is not registered in $SRC/src/fusion.cpp" >&2; exit 1; }
echo "gfx803 Winograd solver restore applied and verified in $SRC"
