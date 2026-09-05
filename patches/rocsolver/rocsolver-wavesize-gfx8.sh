#!/bin/sh
# Apply rocsolver-wavesize-gfx8.patch. That file gives the reason and the change,
# and this driver makes sure that the apply worked.
#
# gfx8xx is a wave64 ISA, like gfx9xx, but upstream's WarpSize gate only recognises
# __GFX9__. So every one-partial-per-wave reduction stores each wave's sum twice.
#
# This driver uses `patch` and not `git apply`, because the rocm-libraries tree is a
# sparse checkout of a monorepo subdirectory. On this git version, `git apply` there
# reports success and applies nothing. The same reasoning applies to ../miopen/*.sh.
set -eu

SRC="${1:-/rocsolver-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/rocsolver-wavesize-gfx8.patch"
FILE="$SRC/library/src/include/lib_device_helpers.hpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE does not exist -- wrong SRC or upstream moved the header" >&2; exit 1; }

if grep -q 'WAVESIZE_GFX8_WAVE64_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --verbose < "$PATCH"

count=$(grep -c 'WAVESIZE_GFX8_WAVE64_PATCH' "$FILE" || true)
[ "$count" -eq 1 ] || {
    echo "FATAL: marker found $count times (expected 1) in $FILE" >&2; exit 1; }
grep -q '#if defined(__GFX9__) || defined(__GFX8__)' "$FILE" || {
    echo "FATAL: WarpSize gate not updated in $FILE" >&2; exit 1; }
echo "rocSOLVER gfx8 wave64 patch applied and verified in $FILE"
