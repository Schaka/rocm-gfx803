#!/bin/sh
# Apply d2h-null-dsthost.patch (see that file for the full WHY/WHAT).
# D2H copy destination for host-accessible *device* allocations: pass the
# destination device VA (getHostMem() is NULL for a device allocation that
# happens to be host-accessible, which made readBuffer memcpy to address 0
# and crash vLLM 10.0 during inference). Must run AFTER d2h-staged-copy.sh
# (this patch edits the branch that patch introduced).
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/d2h-null-dsthost.patch"
FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q 'void\* dstHost = dstDevMem->getDeviceMemory();' "$FILE" || {
    echo "FATAL: dstDevMem->getDeviceMemory() marker not found in $FILE after git apply reported success" >&2
    exit 1
}
grep -q 'readBuffer(\*srcDevMem, dstMem.getHostMem(), realSrcOrigin' "$FILE" && {
    echo "FATAL: old getHostMem() destination still present in $FILE" >&2
    exit 1
}
echo "D2H null-dsthost patch applied and verified"
