#!/bin/sh
# Apply h2d-staged-copy.patch (see that file for the full WHY/WHAT).
# gfx803 H2D queue eviction / MMU notifier hang: force the GPU-staging path
# for Host-to-Device copies. Must run AFTER d2h-null-dsthost.sh.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/h2d-staged-copy.patch"
FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

if grep -q "GFX803: never pin the host source for an H2D copy" "$SRC/projects/clr/rocclr/device/rocm/rocblit.cpp"; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"
