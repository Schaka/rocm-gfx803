#!/bin/sh
# Apply fmm-keep-userptr-map.patch (see that file for the full WHY/WHAT).
# libhsakmt's reserved_aperture_release() munmaps released host-VA ranges
# while the kernel-side KFD userptr for the same range is still alive and
# queued GPU work still reads it; on gfx803 the resulting VM fault wedges
# the CP (torch/ComfyUI checkpoint loads fault deterministically). Keep the
# mapping instead of dropping it.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/fmm-keep-userptr-map.patch"
FILE="$SRC/projects/rocr-runtime/libhsakmt/src/fmm.c"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q 'GFX803_FMM_KEEP_USERPTR_MAP' "$FILE" || {
    echo "FATAL: keep-userptr-map marker not found in $FILE after git apply reported success" >&2
    exit 1
}
grep -q 'Failed to reserve VA range' "$FILE" && {
    echo "FATAL: munmap/PROT_NONE re-reserve still present in $FILE" >&2
    exit 1
}
echo "fmm keep-userptr-map patch applied and verified"
