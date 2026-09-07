#!/bin/sh
# Build the LD_PRELOAD SGEMM shim.
#
# rocBLAS and Tensile's own SGEMM kernels are wrong on gfx803 for every shape
# tested, so this shim answers standard-algo f32 rocblas_sgemm and
# rocblas_gemm_ex with a kernel that was verified on the card. It also takes over
# fp16 gemm_ex and the small-problem rocblas_gemm_strided_batched_ex that
# MIGraphX's batched attention dots land in. The final image sets LD_PRELOAD.
set -eu

ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
SHIM=/opt/rocm/lib/libgfx803_sgemm_shim.so

hipcc -O2 -fPIC -shared "--offload-arch=$ARCH" -I/opt/rocm/include \
    /patches/rocblas/sgemm-shim/sgemm_shim.cpp \
    /patches/rocblas/sgemm-shim/gfx803_gemm_lib.hip \
    -o "$SHIM" \
    -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lrocblas -ldl

# Each marker is a string the corresponding source fix adds. An old source tree
# links and loads fine, so only the marker tells the versions apart.
if ! strings "$SHIM" | grep -q "sb-takeover-no-algo-gate"; then
    echo "FATAL: shim built without the strided-batched takeover fix (algo gate)." >&2
    exit 1
fi
if ! strings "$SHIM" | grep -q "f16-takeover"; then
    echo "FATAL: shim built without the fp16 takeover." >&2
    exit 1
fi
if ! strings "$SHIM" | grep -q "f16-map-nm"; then
    echo "FATAL: shim built with the old fp16 operand mapping. It hands the kernel" >&2
    echo "       (m, n) where the column-major contract needs (n, m), which" >&2
    echo "       transposes the answer and reads past both operands when m != n." >&2
    exit 1
fi
echo "OK: SGEMM shim built for $ARCH with all three fixes."
