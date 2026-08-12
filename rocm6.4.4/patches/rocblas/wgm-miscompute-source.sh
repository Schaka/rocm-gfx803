#!/bin/sh
# Apply wgm-miscompute-source.patch (see that file for the full WHY/WHAT).
# NOT currently wired into rocm6.4.4/Dockerfile -- wgm-miscompute.sh (the
# sed patch) remains the one actually applied. Kept here, self-verifying
# like every other driver in this repo, ready to swap in once the
# same-mechanism equivalence check run against this line's own Tensile
# pin confirms it (see the patch header).
#
# Unlike the rocm7 line, Tensile isn't vendored in rocBLAS's own checkout
# here -- rocBLAS pip-installs it from ROCmSoftwarePlatform/Tensile at
# CMake configure time. This script clones the exact pinned Tensile
# commit into its own directory and patches THAT, so the Dockerfile can
# point rocBLAS's Tensile_TEST_LOCAL_PATH cache var at it instead of
# letting CMake pip-fetch the stock (unpatched) one -- see the Dockerfile
# comment at the call site for the --cmake-darg wiring.
set -eu

TENSILE_COMMIT="7449c7fefd3d208ba6a2705699cd1c13b654ad87"
OUT_DIR="${1:-/tensile-local}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/wgm-miscompute-source.patch"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }

if [ -d "$OUT_DIR" ] && grep -q 'GFX803_WGM_MISCOMPUTE_PATCH' "$OUT_DIR/Tensile/SolutionStructs.py" 2>/dev/null; then
    echo "already cloned and patched at $OUT_DIR, skipping"
    exit 0
fi

git clone --filter=blob:none https://github.com/ROCmSoftwarePlatform/Tensile.git "$OUT_DIR"
git -C "$OUT_DIR" checkout "$TENSILE_COMMIT"

FILE="$OUT_DIR/Tensile/SolutionStructs.py"
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved Tensile's" >&2
    echo "       SolutionStructs.py, so this patch would silently stop applying" >&2
    echo "       and the gfx803 WGM miscompute would ship unfixed." >&2
    exit 1
}

patch -p1 -d "$OUT_DIR" --verbose < "$PATCH"

count=$(grep -c 'GFX803_WGM_MISCOMPUTE_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi
echo "gfx803 WGM source-level patch applied and verified in $FILE"
echo "pass -DTensile_TEST_LOCAL_PATH=$OUT_DIR (rmake.py: --cmake-darg Tensile_TEST_LOCAL_PATH=$OUT_DIR) to use it"
