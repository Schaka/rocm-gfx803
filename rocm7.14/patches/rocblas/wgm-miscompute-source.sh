#!/bin/sh
# Apply wgm-miscompute-source.patch (see that file for the full WHY/WHAT,
# including the real-hardware verification this replaced wgm-miscompute.sh
# on the strength of). Wired into the root Dockerfile as of this commit;
# wgm-miscompute.sh (the sed) stays in this directory for reference/rollback.
set -eu

# Takes the rocm-libraries checkout ROOT (parent of both shared/ and
# projects/rocblas), not ROCBLAS_SRC -- SolutionStructs.py lives in
# shared/tensile, a sibling of projects/rocblas, not under it.
ROOT="${1:-/rocblas-src-root}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/wgm-miscompute-source.patch"
FILE="$ROOT/shared/tensile/Tensile/SolutionStructs.py"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved Tensile's" >&2
    echo "       SolutionStructs.py, so this patch would silently stop applying" >&2
    echo "       and the gfx803 WGM miscompute would ship unfixed." >&2
    exit 1
}

if grep -q 'GFX803_WGM_MISCOMPUTE_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$ROOT" --verbose < "$PATCH"

count=$(grep -c 'GFX803_WGM_MISCOMPUTE_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi
echo "gfx803 WGM source-level patch applied and verified in $FILE"
