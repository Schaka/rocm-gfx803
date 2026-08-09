set -eu

# gfx803's tuned Tensile *assembly* GEMM kernels silently miscompute whenever
# every dimension of the problem is small. They are correct from roughly
# 48x48x48 upwards, and correct for thin/GEMV-shaped problems as long as one
# dimension is large -- the corruption needs *all* of m, n and k to be small.
# rocBLAS returns rocblas_status_success either way, so no caller can detect it.
#
# This is a rocBLAS-level fix on purpose. The bug is in rocBLAS, and every
# consumer on this image (ONNX Runtime, PyTorch, MIGraphX, CTranslate2) hits it.
# An earlier version of this work patched ONNX Runtime's ROCm EP instead, which
# fixed ORT's tests but left every other consumer silently wrong.
#
# Measured on an RX 470 (gfx803), ROCm 6.4.4, onnxruntime_test_all (4882 tests):
#
#   configuration                        failures
#   ----------------------------------   --------
#   stock                                      96
#   with this patch (initial gate)             16
#   + prob.strided_batch guard                 12
#
# The ~84 tests this fixes are Einsum, EinsumTransposeMatMulThreeInputs,
# Attention, DecoderAttention, MultiHeadAttention and FusedMatMul. Most of
# what remains is unrelated to GEMM (seven of the twelve are StringNormalizer,
# a string op).
#
# The strided_batch guard exists because the gate originally routed
# rocblas_gemm_batched_ex (array-of-device-pointers batching, as opposed to
# gemm_strided_batched_ex) into a source solution too. canSolve() validates
# problem shape, not pointer-array compatibility, so it said yes to a solution
# rocBLAS's own dispatch then rejected with rocblas_status_invalid_value --
# not a miscompute, an outright call failure. See KERNEL_BUGS.md, "Resolved:
# the four FusedMatMulOpTest failures...".
#
# How it was pinned down: an in-process verifier that snapshots operands on the
# GEMM's own stream and recomputes on the host reported miscomputing problems as
# small as **1x1x1** -- one output element, one multiply. That rules out
# GlobalSplitU, the LocalSplitU LDS reduction and barrier ordering as the cause;
# the generated assembly for the dispatched kernel was also checked directly and
# has correct ds_write / s_waitcnt / s_barrier / ds_read ordering, and switching
# EdgeType from ShiftPtr to Branch changed nothing. The defect inside the
# assembly kernels has NOT been isolated. This patch avoids those kernels for
# the shapes where they are wrong rather than fixing them.
#
# Nothing is given up: the source kernels are *faster* than the assembly ones at
# this size (0.63-0.71x the runtime at 2x2x1 and 4x4x4), and only the small-shape
# path changes -- every larger GEMM still goes through findBestSolution and the
# tuned assembly kernels.
#
# The solution is looked up per problem rather than hardcoded. Solution indices
# are specific to a transpose combination: the NN source solution is not
# applicable to a TN GEMM, and forcing it there returns
# rocblas_status_invalid_value and fails the call. An earlier attempt did
# exactly that and produced ~14 phantom "Attention regressions" that were really
# just rejected calls.
#
# See ../../KERNEL_BUGS.md for the full investigation.

SRC="${1:-/rocblas-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/small-gemm-assembly-miscompute.patch"
FILE="$SRC/library/src/tensile_host.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || {
    echo "FATAL: $FILE does not exist -- upstream moved rocBLAS's Tensile" >&2
    echo "       dispatch, so this patch would silently stop applying and the" >&2
    echo "       gfx803 small-GEMM miscompute would ship." >&2
    exit 1
}

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'GFX803_SMALL_GEMM_PATCH' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found after git apply reported success" >&2
    exit 1
fi
echo "gfx803 small-GEMM assembly miscompute patch applied and verified in $FILE"
