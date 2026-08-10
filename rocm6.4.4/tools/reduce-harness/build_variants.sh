#!/bin/bash
# Build the CK reduction kernel code objects (codeobj/*.hsaco) the harness needs.
# Run once per target ROCm. Adjust CK_SRC/CK_INC to your MIOpen CK checkout.
set -u
CK_SRC="${CK_SRC:-<path-to-MIOpen>/src/composable_kernel/composable_kernel/src/kernel_wrapper}"
CK_INC="${CK_INC:-<path-to-MIOpen>/src/composable_kernel/composable_kernel/include}"
OUT="$(dirname "$0")/codeobj"
mkdir -p "$OUT"
INCS="-I. "
for d in $(find "$CK_INC" -type d); do INCS="$INCS -I$d"; done
# also need MIOpen bootstrap headers (bfloat16_dev.hpp, miopen_cstdint.hpp, ...)
# -> add -I pointing at MIOpen's src/kernels if not already in CK_INC's tree.

BUILD() { # prefix method srcpad dstpad outbase
  P=$1; M=$2; SP=$3; DP=$4; BASE="$OUT/$5"
  SRC="$CK_SRC/gridwise_generic_reduction_${P}_${M}_reduce_partial_dims.cpp"
  [ -f "$SRC" ] || { echo "MISSING $SRC"; return; }
  hipcc --genco -std=c++17 -DCK_AMD_GPU_GFX803 -DCK_USE_AMD_BUFFER_ATOMIC_FADD=0 \
    -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM=1 -DCK_USE_AMD_BUFFER_ADDRESSING=1 \
    -DCK_PARAM_SRC_DATATYPE=1 -DCK_PARAM_DST_DATATYPE=1 -DCK_PARAM_REDUCE_COMPTYPE=1 \
    -DCK_PARAM_BLOCKSIZE=256 -DCK_PARAM_THREAD_BUFFER_LENGTH=8 \
    -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=2 -DCK_PARAM_ACCESSES_PER_THREAD_INWARP=2 \
    -DCK_PARAM_NUM_TOREDUCE_DIMS=1 -DCK_PARAM_REDUCE_OP=0 -DCK_PARAM_NAN_PROPAGATE=1 \
    -DCK_PARAM_REDUCE_INDICES=0 -DCK_PARAM_IN_DIMS=2 -DCK_PARAM_OUT_DIMS=1 \
    -DCK_PARAM_SRC2D_PADDING=$SP -DCK_PARAM_DST1D_PADDING=$DP -mcpu=gfx803 \
    $INCS "$SRC" -o "$BASE.hsaco" 2>"$BASE.err" \
    && { [ ! -s "$BASE.err" ] && echo "OK   $BASE"; } \
    || { echo "FAIL $BASE"; grep -iE 'error' "$BASE.err" | head -3; }
  rm -f "$BASE.err"
}

BUILD first_call threadwise 0 0 v_tw_00
BUILD first_call threadwise 1 0 v_tw_10
BUILD first_call threadwise 1 1 v_tw_11
BUILD first_call warpwise 1 0 v_wp_10
BUILD first_call warpwise 1 1 v_wp_11
BUILD first_call blockwise 0 0 v_bw_00
BUILD first_call blockwise 1 0 v_bw_10
BUILD first_call multiblock 1 0 v_mb_10
BUILD second_call threadwise 1 1 v2_tw_11
BUILD second_call warpwise 1 0 v2_wp_10
BUILD second_call blockwise 0 0 v2_bw_00
echo "DONE_ALL -> $OUT"
