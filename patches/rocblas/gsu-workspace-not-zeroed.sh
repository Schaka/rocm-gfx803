#!/bin/sh
# Apply gsu-workspace-not-zeroed.patch (see that file for the full WHY/WHAT)
# via `git apply`, then verify both hunks actually landed. `git apply` on a
# tag that will never move again (rocm-6.4.4, gfx803 is abandoned upstream)
# either matches exactly or fails loud -- this wrapper exists to catch
# operator error (wrong clone, already-patched tree, patch/source mismatch),
# not upstream drift.
set -eu

SRC="${1:-/rocblas-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gsu-workspace-not-zeroed.patch"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

for marker in GSU_WORKSPACE_ZERO_PATCH GSU_WORKSPACE_SIZE_GUARD_PATCH; do
    count=$(grep -rc "$marker" "$SRC/library/src/tensile_host.cpp" \
                              "$SRC/library/src/include/handle.hpp" 2>/dev/null \
            | awk -F: '{s+=$2} END{print s+0}')
    if [ "$count" -lt 1 ]; then
        echo "FATAL: marker $marker not found after git apply reported success" >&2
        exit 1
    fi
done
echo "GSU workspace patches applied and verified in $SRC"
