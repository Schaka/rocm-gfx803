#!/bin/sh
# Apply tensile-gfx803-fp16-nond16.patch (see that file for the full
# WHY/WHAT, including the real-hardware verification). Enables non-HPA fp16
# GEMM assembly kernels on gfx803 (unpacked codegen) and fixes the
# gfx803-only WGM work-group division bug that made them miscompute.
set -eu

# Takes the rocm-libraries checkout ROOT (parent of shared/tensile), not
# ROCBLAS_SRC -- these files live in shared/tensile/Tensile.
ROOT="${1:-/rocblas-src-root}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/tensile-gfx803-fp16-nond16.patch"
FILE="$ROOT/shared/tensile/Tensile/KernelWriterAssembly.py"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved Tensile's" >&2
    echo "       KernelWriterAssembly.py, so this patch would silently" >&2
    echo "       stop applying and fp16 GEMM on gfx803 would stay broken." >&2
    exit 1
}

if grep -q 'GFX803_FP16_NOD16_PATCH' "$FILE"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$ROOT" --verbose < "$PATCH"

count=$(grep -c 'GFX803_FP16_NOD16_PATCH' "$FILE" || true)
if [ "$count" -eq 0 ]; then
    echo "FATAL: marker not found after patch reported success" >&2
    exit 1
fi

RFR="$ROOT/shared/tensile/Tensile/Source/client/source/ResultFileReporter.cpp"
if ! grep -q 'catch(std::invalid_argument const&)' "$RFR"; then
    echo "FATAL: client stoi fix (ResultFileReporter.cpp) not found" >&2
    exit 1
fi
echo "gfx803 fp16 non-d16 Tensile patch applied and verified in $FILE + $RFR"