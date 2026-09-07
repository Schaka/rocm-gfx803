#!/usr/bin/env bash
# Apply the gfx803 ISA-family wiring to a triton checkout.
#
# Usage:
#   ./apply-gfx803-isa-family.sh /path/to/triton
#
# Verifies its own result: greps the patched sources for the GCN3 family and
# fails loudly if missing.

set -euo pipefail

TRITON_DIR="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-isa-family.patch"
TARGET="$TRITON_DIR/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp"

if [[ -z "$TRITON_DIR" || ! -f "$TARGET" ]]; then
    echo "usage: $0 /path/to/triton (must contain $TARGET)" >&2
    exit 1
fi

git -C "$TRITON_DIR" apply --check "$PATCH"
git -C "$TRITON_DIR" apply "$PATCH"

if grep -q "return ISAFamily::GCN3;" "$TARGET" \
   && grep -q "case ISAFamily::GCN3:" "$TARGET"; then
    echo "gfx803-isa-family.patch applied and verified (GCN3 family wired)."
else
    echo "ERROR: patch applied but GCN3 wiring not found" >&2
    exit 1
fi
