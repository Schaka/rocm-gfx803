#!/bin/sh
set -eu
SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/winograd-fused-conv-miscompute.patch"
FILE="$SRC/src/solver/conv_bin_winoRxS_fused.cpp"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'WINOGRAD_FUSED_CBA_MISCOMPUTE_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after git apply reported success" >&2
    exit 1
fi
echo "Winograd fused-CBA miscompute patch applied and verified in $FILE"
