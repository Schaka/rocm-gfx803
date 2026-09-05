#!/bin/sh
# Apply graph-replay-batch-chunk-deadlock.patch with `git apply`, then make sure
# that the hunk landed. That patch file gives the reason and the change.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/graph-replay-batch-chunk-deadlock.patch"
ROCVIRTUAL_FILE="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if ! grep -q "std::min<size_t>(DEBUG_HIP_GRAPH_BATCH_SIZE, sw_queue_size)" "$ROCVIRTUAL_FILE"; then
    echo "FATAL: kPeriod clamp not found in $ROCVIRTUAL_FILE after git apply reported success" >&2
    exit 1
fi
echo "graph-replay-batch-chunk-deadlock patch applied and verified in $ROCVIRTUAL_FILE"
