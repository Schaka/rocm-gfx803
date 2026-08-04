#!/bin/sh
# Fix the gfx803 WorkGroupMapping miscompute, by forcing WorkGroupMapping=1 in
# every Tensile solution before rocBLAS generates kernels from them.
#
# Naming note: WGM8 is the BROKEN configuration; WGM1 is the fix. Kernel names
# carry the value they were built with, so a correct library has ..._WGM1 kernels
# and no ..._WGM8 ones.
#
# WHY
# ---
# Tensile's WorkGroupMapping ("WGM") swizzles the workgroup-id -> output-tile
# assignment to improve L2 reuse: blockId = wg1 / WGM, then the workgroups are
# re-dealt across a WGM-wide block of tiles using two magic-number divides. On
# gfx803 the generated assembly for that swizzle does not produce a bijection --
# several workgroups land on the same output tile while other tiles are never
# claimed at all -- so the GEMM silently returns wrong results with
# rocblas_status_success.
#
# Measured on an RX 470 (Polaris, gfx803) against ROCm 6.4.4, by enumerating the
# candidate solutions for one problem with rocblas_gemm_ex_get_solutions and
# running each one individually:
#
#   WGM1  assembly kernels : 39 / 39 correct   (ISA803 / KLA)
#   WGM8  assembly kernels :  0 / 14 correct   (ISA803 / KLA)
#   WGM8  source kernels   :  2 /  2 correct   (ISA000 / KLS -- Tensile's HIP-source
#                                               path, where the swizzle is emitted by
#                                               the compiler rather than by Tensile)
#
# The separation is exact, and the pair
# Cijk_..._MT64x64x16_..._WGM8 / Cijk_..._MT64x64x16_..._WGM1 differs in nothing
# but that one parameter, so it is the swizzle and not the tiling. Confirmed by
# NOPing the swizzle out of the shipped code objects: 55/55 solutions then pass.
#
# It is not a "big matrix" bug even though that is how it shows up. The broken
# kernels are wrong at every size; which size fails is decided purely by which
# solution the selector happens to pick, so M<=512 looked fine and M>=768 did not.
# Nor is it fp32-only: WGM8 kernels exist in the gfx803 libraries for double,
# complex, half, bfloat16 and int8 as well.
#
# WHAT
# ----
# WGM only permutes which workgroup claims which tile -- the grid and the set of
# tiles are identical either way -- so WGM=1 (identity) is always a valid
# assignment. The cost is some L2 locality on large GEMMs; the benefit is
# arithmetic that is correct. gfx803 has been abandoned upstream since ROCm 6.0,
# so carrying this patch indefinitely is fine.
#
# Applied to the whole Logic tree rather than just Logic/asm_full/r9nano (the
# gfx803-tuned directory) because the per-type "fallback" libraries are generated
# from other logic directories and carry WGM8 kernels too. That is safe here
# precisely because this image builds ONE architecture: rmake.py runs with
# -a gfx803, so no other GPU's kernels are generated from this tree.
set -eu

SRC="${1:-/rocblas-src}"
LOGIC="$SRC/library/src/blas3/Tensile/Logic"

[ -d "$LOGIC" ] || { echo "FATAL: no Tensile logic tree at $LOGIC" >&2; exit 1; }

before=$(grep -rh "WorkGroupMapping:" "$LOGIC" --include='*.yaml' \
         | grep -cv "WorkGroupMapping: 1$" || true)
if [ "$before" -eq 0 ]; then
    echo "FATAL: found no non-1 WorkGroupMapping in $LOGIC -- upstream layout changed," >&2
    echo "       so this patch is no longer doing anything and the gfx803 GEMM bug" >&2
    echo "       it works around would ship silently. Re-check before removing." >&2
    exit 1
fi
echo "rewriting $before non-1 WorkGroupMapping entries to 1"

find "$LOGIC" -name '*.yaml' -print0 \
    | xargs -0 sed -i -E 's/^([[:space:]]*)WorkGroupMapping: -?[0-9]+[[:space:]]*$/\1WorkGroupMapping: 1/'

after=$(grep -rh "WorkGroupMapping:" "$LOGIC" --include='*.yaml' \
        | grep -cv "WorkGroupMapping: 1$" || true)
if [ "$after" -ne 0 ]; then
    echo "FATAL: $after WorkGroupMapping entries still non-1 after rewrite" >&2
    grep -rn "WorkGroupMapping:" "$LOGIC" --include='*.yaml' \
        | grep -v "WorkGroupMapping: 1$" | head -10 >&2
    exit 1
fi
echo "all WorkGroupMapping entries are now 1"
