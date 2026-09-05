#!/bin/sh
# Apply reduce-prod-wrong-identity.patch. That file gives the full reason, and it
# states that the fault is not gfx803-specific. The patch came unchanged from
# rocm6.4.4/patches/miopen/, and it was confirmed to apply cleanly against the
# pinned 7.14 source before it was carried here. Only one line number moved, and no
# content drifted.
#
# This driver uses `patch` and not `git apply`. On this git version,
# `git apply --check` on a sparse checkout of a monorepo subdirectory reports
# success and changes nothing ("Skipped patch", exit 0). See
# ../rocblas/small-gemm-assembly-miscompute.sh.
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
