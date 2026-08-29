#!/bin/sh
# Apply graph-replay-queue-size-cap.patch (see that file for the full
# WHY/WHAT -- READ IT FIRST, this patch requires a companion kernel patch
# not shipped by this repo; applying only this half is dangerous, not just
# ineffective). NOT called from the main Dockerfile on purpose -- run this
# by hand only if you've already built and loaded the kernel companion
# (patches/kernel/REFERENCE-amdkfd-gfx7-8-queue-size-writeback.patch) and
# verified it with a real dispatch, not just hsa_queue_create succeeding.
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
