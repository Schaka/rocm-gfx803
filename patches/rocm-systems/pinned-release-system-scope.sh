#!/bin/sh
# Apply pinned-release-system-scope.patch with `git apply`, then make sure that
# the hunk landed. That patch file gives the reason and the change. In short: the
# pinned-buffer release marker must be a system-scope barrier, because the buffer
# may only be unlocked after the copy shader's L2 writes have drained.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/pinned-release-system-scope.patch"
FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q 'marker->setCommandEntryScope(amd::Device::kCacheStateSystem)' "$FILE" || {
    echo "FATAL: system-scope marker line not found in $FILE after git apply reported success" >&2
    exit 1
}
echo "Pinned-release system-scope marker patch applied and verified in $FILE"