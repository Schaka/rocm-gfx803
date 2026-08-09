#!/bin/sh
# ABANDONED -- not called from the Dockerfile, not shipped. See the
# "WHY THIS WAS ABANDONED" note at the top of reduce-kernel-cache-eviction.patch
# before wiring this back in: it passed every standalone repro but still
# failed ORT's real test suite intermittently, with a worse failure mode
# than the original bug. Kept only so the exact patch that was tried is
# reproducible if someone picks this investigation back up.
#
# Apply reduce-kernel-cache-eviction.patch via `git apply`, then verify the
# hunks actually landed.
set -eu

SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/reduce-kernel-cache-eviction.patch"
FILE="$SRC/src/reducetensor.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'REDUCE_CACHE_EVICT_PATCH' "$FILE" || true)
if [ "$count" -ne 4 ]; then
    echo "FATAL: marker not found after git apply reported success" >&2
    exit 1
fi
echo "Reduce kernel-cache eviction workaround applied and verified in $FILE"
