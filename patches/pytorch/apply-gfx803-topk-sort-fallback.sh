#!/usr/bin/env bash
# Apply and verify gfx803 TopK sort fallback against a pinned PyTorch tree.
# Usage: apply-gfx803-topk-sort-fallback.sh /path/to/pytorch
set -euo pipefail

SRC="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-topk-sort-fallback.patch"
TARGET="$SRC/aten/src/ATen/native/cuda/TensorTopK.cpp"
MARKER="GFX803_TOPK_SORT_FALLBACK_PATCH"

if [[ -z "$SRC" || ! -f "$TARGET" ]]; then
    echo "usage: $0 /path/to/pytorch" >&2
    exit 1
fi

if grep -q "$MARKER" "$TARGET"; then
    # Existing source must include both the exact architecture helper and the
    # selected sort branch; a marker alone cannot admit a partial edit.
    grep -Fq 'getCUDAHooks().isGPUArch({"gfx803"})' "$TARGET" &&
    grep -Fq 'return true;' "$TARGET" || {
        echo "FATAL: TopK marker exists without required architecture dispatch" >&2
        exit 1
    }
    echo "gfx803 TopK patch already present; verified"
    exit 0
fi

patch -p1 -d "$SRC" --batch --fuzz=0 < "$PATCH"

grep -q "$MARKER" "$TARGET" &&
grep -Fq 'getCUDAHooks().isGPUArch({"gfx803"})' "$TARGET" || {
    echo "FATAL: patch returned success but verified markers are absent" >&2
    exit 1
}

echo "gfx803-topk-sort-fallback.patch applied and verified in $TARGET"
