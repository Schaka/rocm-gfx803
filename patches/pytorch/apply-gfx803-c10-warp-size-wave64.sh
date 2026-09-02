#!/usr/bin/env bash
# Apply the gfx803 wave64 C10_WARP_SIZE fix to a PyTorch checkout.
#
# Usage:
#   ./apply-gfx803-c10-warp-size-wave64.sh /path/to/pytorch
#
# Verifies its own result: greps the patched header for defined(__GFX8__) and
# fails loudly if missing.

set -euo pipefail

SRC="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-c10-warp-size-wave64.patch"

if [[ -z "$SRC" || ! -d "$SRC" ]]; then
    echo "usage: $0 /path/to/pytorch" >&2
    exit 1
fi

# 2.14 defines C10_WARP_SIZE in torch/headeronly/macros/Macros.h (c10/macros/
# Macros.h just includes it); older lines carry it directly in c10/macros/Macros.h.
TARGET=""
if [[ -f "$SRC/torch/headeronly/macros/Macros.h" ]] \
   && grep -q "C10_WARP_SIZE_INTERNAL" "$SRC/torch/headeronly/macros/Macros.h"; then
    TARGET="$SRC/torch/headeronly/macros/Macros.h"
elif [[ -f "$SRC/c10/macros/Macros.h" ]] \
     && grep -q "C10_WARP_SIZE_INTERNAL" "$SRC/c10/macros/Macros.h"; then
    TARGET="$SRC/c10/macros/Macros.h"
else
    echo "FATAL: could not locate Macros.h defining C10_WARP_SIZE_INTERNAL" >&2
    exit 1
fi

if grep -q 'defined(__GFX8__)' "$TARGET"; then
    echo "already patched in $TARGET, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --batch < "$PATCH"

if ! grep -q 'defined(__GFX8__)' "$TARGET"; then
    echo "FATAL: defined(__GFX8__) marker not found after patch reported success" >&2
    exit 1
fi

echo "gfx803-c10-warp-size-wave64.patch applied and verified in $TARGET"
