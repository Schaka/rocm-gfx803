#!/usr/bin/env bash
# Apply the gfx803 packed-dp4a patch to a llama.cpp checkout.
#
# Usage:
#   ./apply-packed-dp4a.sh /path/to/llama.cpp
#
# Verifies its own result: greps the patched common.cuh for the GCN4 branch
# and fails loudly if it's missing. Matches the repo convention that every
# apply driver proves the patch landed, not just that the tool exited 0.

set -euo pipefail

LLAMA_DIR="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-packed-dp4a.patch"

if [[ -z "$LLAMA_DIR" || ! -f "$LLAMA_DIR/ggml/src/ggml-cuda/common.cuh" ]]; then
    echo "usage: $0 /path/to/llama.cpp (must contain ggml/src/ggml-cuda/common.cuh)" >&2
    exit 1
fi

# Repo convention: `patch -p1` for anything that is not a full git clone.
# llama.cpp is a full git repo, so git apply is the correct dialect.
git -C "$LLAMA_DIR" apply --check "$PATCH"
git -C "$LLAMA_DIR" apply "$PATCH"

if grep -q "v_mul_i32_i24" "$LLAMA_DIR/ggml/src/ggml-cuda/common.cuh"; then
    echo "gfx803-packed-dp4a.patch applied and verified (GCN4 branch present)."
else
    echo "ERROR: patch applied but GCN4 branch not found in common.cuh" >&2
    exit 1
fi