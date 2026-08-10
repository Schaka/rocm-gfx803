set -eu

# ROCm's generic TopK kernel's RadixTopK path (used whenever K is small
# enough or the result is unsorted) allocates tied slots via hipCUB's
# BlockScan/BlockReduce -- rocPRIM-backed on ROCm, a different implementation
# from NVIDIA's own cub. With heavy exact ties in the input, it picks a
# different winner among tied candidates from run to run on bit-identical
# data: BeamSearchTest.GptBeamSearchFp32_DisableFastTopK failed ~50% of
# isolated single-test runs. Not a gfx803-specific bug and not upstream
# floating-point noise -- see the patch for the full investigation, including
# two other fixes that were tried and rejected (one crashes, one can't safely
# cover this shape).

SRC="${1:-/onnxruntime}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/topk-radix-tiebreak-nondeterministic.patch"
FILE="$SRC/onnxruntime/core/providers/cuda/math/topk_impl.cuh"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved this file, so this" >&2
    echo "       patch would silently stop applying and the TopK tie-break" >&2
    echo "       nondeterminism would ship." >&2
    exit 1
}

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'SAFE_SMALL_K_TOPK_PATCH' "$FILE" || true)
if [ "$count" -ne 2 ]; then
    echo "FATAL: expected 2 marker occurrences, found $count after git apply reported success" >&2
    exit 1
fi
echo "TopK radix tie-break nondeterminism patch applied and verified in $FILE"
