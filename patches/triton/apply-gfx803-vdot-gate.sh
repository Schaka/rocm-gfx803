#!/usr/bin/env bash
# Apply the gfx803 v_dot gate to a triton checkout.
#
# Usage:
#   ./apply-gfx803-vdot-gate.sh /path/to/triton
#
# Verifies its own result: greps the patched sources for the new predicate and
# for both of its call sites, and fails loudly if any is missing. Requires
# gfx803-isa-family.patch already applied: this patch's TargetFeatures.cpp hunk
# keeps `case ISAFamily::GCN3:` as context.

set -euo pipefail

TRITON_DIR="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gfx803-vdot-gate.patch"
FEATURES_H="$TRITON_DIR/third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h"
FEATURES_CPP="$TRITON_DIR/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp"
FMA_CPP="$TRITON_DIR/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp"
MATMUL_CPP="$TRITON_DIR/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp"

for f in "$FEATURES_H" "$FEATURES_CPP" "$FMA_CPP" "$MATMUL_CPP"; do
    if [[ -z "$TRITON_DIR" || ! -f "$f" ]]; then
        echo "usage: $0 /path/to/triton (must contain $f)" >&2
        exit 1
    fi
done

git -C "$TRITON_DIR" apply --check "$PATCH"
git -C "$TRITON_DIR" apply "$PATCH"

# The predicate must exist, be defined, and be consulted at both call sites. A
# patch that applies but leaves the fdot2 cell unguarded is exactly the state
# this patch exists to end, so check for the call, not for the declaration.
fail=0
grep -q "bool supportsVDot() const;" "$FEATURES_H" || { echo "ERROR: supportsVDot() not declared" >&2; fail=1; }
grep -q "bool TargetFeatures::supportsVDot() const {" "$FEATURES_CPP" || { echo "ERROR: supportsVDot() not defined" >&2; fail=1; }
grep -q "supportsVDot()) {" "$FMA_CPP" || { echo "ERROR: FMA.cpp does not gate its packed cells on supportsVDot()" >&2; fail=1; }
grep -q "if (targetFeatures.supportsVDot()) {" "$MATMUL_CPP" || { echo "ERROR: isLegalFMAForm does not gate the v_dot forms on supportsVDot()" >&2; fail=1; }
grep -q "if (!targetFeatures.supportsVDot())" "$MATMUL_CPP" || { echo "ERROR: tryAccelerateF16WithVDot guard missing" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1

echo "gfx803-vdot-gate.patch applied and verified (v_dot gated on the target)."
