#!/bin/sh
# Apply h2d-staged-copy.patch with `git apply`, then make sure that every hunk
# landed. That patch file gives the reason and the change. In short: on gfx803,
# pinning a pageable H2D source lets the KFD MMU notifier evict the process's
# queues, and an evicted idle queue can hang for good. The change sends every H2D
# copy through the CPU-filled staging buffer.
#
# Run this after d2h-null-dsthost.sh, because the copyMemory hunk extends the
# branch that patch rewrites.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/h2d-staged-copy.patch"
BLIT="$SRC/projects/clr/rocclr/device/rocm/rocblit.cpp"
FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$BLIT" ] || { echo "FATAL: $BLIT missing" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

# Marker check, not `apply --check --reverse`: later chain drivers touch
# rocvirtual.cpp, so a clean reverse only holds on a tree that contains nothing
# past this patch.
if grep -q 'GFX803: never pin the host source for an H2D copy' "$BLIT"; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q 'GFX803: never pin the host source for an H2D copy' "$BLIT" || {
    echo "FATAL: enablePin=false marker not found in $BLIT after git apply reported success" >&2
    exit 1
}
grep -q 'GFX803: do NOT read the (registered/locked) host buffer directly' "$FILE" || {
    echo "FATAL: submitWriteMemory staged-routing marker not found in $FILE after git apply reported success" >&2
    exit 1
}
grep -q 'srcDevMem->getDeviceMemory() + realSrcOrigin\[0\];' "$FILE" || {
    echo "FATAL: copyMemory source VA + realSrcOrigin marker not found in $FILE after git apply reported success" >&2
    exit 1
}
echo "H2D staged-copy patch applied and verified"
