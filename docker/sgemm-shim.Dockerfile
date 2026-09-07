# syntax=docker/dockerfile:1
#
# The LD_PRELOAD SGEMM shim, built from this tree's own sources.
#
# It starts from rocblas because that is all it links against, and it is its own
# target so that the final image always gets a shim built from the tree it was
# built with, whether or not rocBLAS itself was rebuilt in the same run. It has
# no rocSOLVER dependency, so it must not chain through that target and pay for a
# full rocSOLVER compile to reach rocBLAS.
#
# This target is never published. It exists only as a build context for final.
FROM rocblas

ARG ROCM_ARCH

RUN --mount=type=bind,source=scripts/build/sgemm-shim.sh,target=/scripts/build/sgemm-shim.sh \
    --mount=type=bind,source=patches/rocblas/sgemm-shim,target=/patches/rocblas/sgemm-shim \
    /scripts/build/sgemm-shim.sh
