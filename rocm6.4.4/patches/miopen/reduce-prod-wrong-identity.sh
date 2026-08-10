#!/bin/sh
# Apply reduce-prod-wrong-identity.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunk actually landed.
set -eu

SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/reduce-prod-wrong-identity.patch"
FILE="$SRC/src/kernels/MIOpenReduceCalculation.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'REDUCE_PROD_WRONG_IDENTITY_PATCH' "$FILE" || true)
if [ "$count" -ne 2 ]; then
    echo "FATAL: marker not found after git apply reported success" >&2
    exit 1
fi
echo "Reduce-Prod wrong-identity patch applied and verified in $FILE"
