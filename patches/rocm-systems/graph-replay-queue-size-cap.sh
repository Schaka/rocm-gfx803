#!/bin/sh
# Apply graph-replay-queue-size-cap.patch with `git apply`, then make sure that the
# hunk landed. Read that patch file first. This patch needs a companion kernel patch
# that this repo does not ship. Apply only this half and the result is dangerous,
# and not merely ineffective.
#
# The main Dockerfile does not call this driver, on purpose. Run it by hand only
# after you build and load the kernel companion
# (patches/kernel/REFERENCE-amdkfd-gfx7-8-queue-size-writeback.patch) and test it
# with a real dispatch. A successful hsa_queue_create is not that test.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/graph-replay-queue-size-cap.patch"
ROCVIRTUAL_FILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if ! grep -q "reported_ring_size_bytes" "$ROCVIRTUAL_FILE"; then
    echo "FATAL: reported_ring_size_bytes not found in $ROCVIRTUAL_FILE after git apply reported success" >&2
    exit 1
fi
echo "graph-replay-queue-size-cap patch applied and verified in $ROCVIRTUAL_FILE"
echo "REMINDER: this requires the companion kernel patch to be safe -- see this patch's header."
