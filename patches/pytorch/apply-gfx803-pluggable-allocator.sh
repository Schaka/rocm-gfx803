#!/usr/bin/env bash
# Apply the gfx803 pluggable-allocator patch to a PyTorch checkout and build
# the hipMalloc-backed allocator .so it loads at runtime.
#
# Usage:
#   ./apply-gfx803-pluggable-allocator.sh /path/to/pytorch [/path/to/hipcc]
#
# The patch itself is inert by default: the allocator only activates when
# GFX803_PLUGGABLE_ALLOCATOR is set at runtime (see the patch header).
# Verifies its own result: greps for the env-var hook in torch/cuda/__init__.py
# and the stats counters in CUDAPluggableAllocator.h, and confirms the .so's
# gfx803_alloc symbol exists -- fails loudly if any marker is missing.

set -euo pipefail

SRC="${1:-}"
HIPCC="${2:-hipcc}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-pluggable-allocator.patch"
HIP="$PATCH_DIR/gfx803_pluggable.hip"
SO_OUT="${GFX803_PLUGGABLE_SO:-/opt/rocm/lib/libgfx803_pluggable.so}"

if [[ -z "$SRC" || ! -d "$SRC" ]]; then
    echo "usage: $0 /path/to/pytorch" >&2
    exit 1
fi

if ! grep -q "GFX803_PLUGGABLE_ALLOCATOR" "$SRC/torch/cuda/__init__.py"; then
    patch -p1 -d "$SRC" --batch < "$PATCH"
fi

if ! grep -q "GFX803_PLUGGABLE_ALLOCATOR" "$SRC/torch/cuda/__init__.py" \
   || ! grep -q "current_bytes_" "$SRC/torch/csrc/cuda/CUDAPluggableAllocator.h"; then
    echo "FATAL: pluggable-allocator markers not found after patch reported success" >&2
    exit 1
fi

mkdir -p "$(dirname "$SO_OUT")"
if ! "$HIPCC" --offload-arch=gfx803 -O2 -fPIC -shared \
        "$HIP" -o "$SO_OUT" -lamdhip64; then
    echo "FATAL: hipcc build of $HIP failed" >&2
    exit 1
fi

if ! nm -D "$SO_OUT" | grep -q " T gfx803_alloc$"; then
    echo "FATAL: gfx803_alloc symbol missing from $SO_OUT" >&2
    exit 1
fi

echo "gfx803-pluggable-allocator.patch applied, verified, and $SO_OUT built"
