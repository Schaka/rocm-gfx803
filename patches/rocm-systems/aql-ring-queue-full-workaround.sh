#!/bin/sh
# Apply aql-ring-queue-full-workaround.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunks actually landed.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/aql-ring-queue-full-workaround.patch"
R="$SRC/projects/rocr-runtime/runtime/hsa-runtime"
QUEUE_FILE="$R/core/runtime/amd_aql_queue.cpp"
DRIVER_FILE="$R/core/driver/kfd/amd_kfd_driver.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

# The flag has to be both requested from KFD and reflected in the size
# reported to CreateQueue -- either half alone is worse than neither (a
# doubled report with no mirror faults the GPU; a mirror the CP is never
# told about is the unfixed hang). Check for both, not just that the
# patch tool was happy.
if ! grep -q "ui32.AQLQueueMemory" "$DRIVER_FILE"; then
    echo "FATAL: AQLQueueMemory translation not found in $DRIVER_FILE after git apply reported success" >&2
    exit 1
fi
if ! grep -q "MemoryRegion::AllocateDoubleMap" "$QUEUE_FILE"; then
    echo "FATAL: ring buffer does not request AllocateDoubleMap in $QUEUE_FILE after git apply reported success" >&2
    exit 1
fi
if ! grep -q "queue_full_workaround_) ring_buf_alloc_bytes_ \*= 2;" "$QUEUE_FILE"; then
    echo "FATAL: doubled ring span not published in $QUEUE_FILE after git apply reported success" >&2
    exit 1
fi
echo "aql-ring-queue-full-workaround patch applied and verified in $QUEUE_FILE and $DRIVER_FILE"
