#!/bin/sh
# Apply d2h-staged-copy.patch (see that file for the full WHY/WHAT).
# gfx803 D2H churn page fault: force the GPU-staging path for D2H copies to
# host memory. Must run AFTER graph-replay-batch-chunk-deadlock.sh (the
# rocvirtual.cpp hunk offsets assume its chunk-size change).
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/d2h-staged-copy.patch"
FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q 'constexpr bool kEnablePin = false;' "$SRC/projects/clr/rocclr/device/rocm/rocblit.cpp" || {
    echo "FATAL: kEnablePin=false marker not found after git apply reported success" >&2
    exit 1
}
grep -q 'Route through the staged' "$FILE" || {
    echo "FATAL: staged-routing marker not found in $FILE after git apply reported success" >&2
    exit 1
}
echo "D2H staged-copy patch applied and verified"