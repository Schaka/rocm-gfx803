#!/bin/sh
# Apply hsa-agent-rejects-legacy-doorbell.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunks actually landed.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/hsa-agent-rejects-legacy-doorbell.patch"
AGENT_FILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp"
QUEUE_FILE="$SRC/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if grep -q "DoorbellType != 2" "$AGENT_FILE"; then
    echo "FATAL: DoorbellType != 2 throw still present in $AGENT_FILE after git apply reported success" >&2
    exit 1
fi
if ! grep -q "doorbell_type_ == 0" "$QUEUE_FILE"; then
    echo "FATAL: legacy doorbell dispatch branch not found in $QUEUE_FILE after git apply reported success" >&2
    exit 1
fi
echo "Legacy-doorbell patch applied and verified in $AGENT_FILE and $QUEUE_FILE"
