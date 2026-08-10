set -eu

# MultiHeadAttention's ROCm "Generic" fallback pipeline (the only path
# available once Composable Kernel is compiled out, as it is in this image)
# refuses to run the plain three-separate-tensor MHA call ("MHA basic": Q, K,
# V passed unpermuted, no fused QKV projection) whenever query
# sequence_length > 1 -- an entirely ordinary shape, not an edge case. It was
# simply never given the transpose step its own GEMMs need. rocBLAS is never
# even reached; the rejection happens in ORT's own dispatch before any GPU
# kernel launches. See the patch for the full investigation and why the fix
# belongs in ORT, not rocBLAS/MIOpen.

SRC="${1:-/onnxruntime}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/mha-basic-mode-no-viable-op.patch"
FILE="$SRC/onnxruntime/contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved this pipeline file," >&2
    echo "       so this patch would silently stop applying and the MHA" >&2
    echo "       basic-mode no-viable-op bug would ship." >&2
    exit 1
}

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'MHA_BASIC_MODE_TRANSPOSE_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after git apply reported success" >&2
    exit 1
fi
echo "MHA basic-mode no-viable-op patch applied and verified in $FILE"
