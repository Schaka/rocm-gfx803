# syntax=docker/dockerfile:1
#
# rocSOLVER built from source for gfx803, installed over /opt/rocm.
#
# The rocSOLVER the pinned 10.0 stack ships has an empty .hip_fatbin for every
# architecture: host stubs and kernel registration tables, no device code. Every
# hipSOLVER backed torch.linalg entry point crashes at its first launch.
#
# It links rocBLAS, so it starts from that image. Only the library file is taken
# forward downstream, never this whole tree.
FROM python-base

ARG ROCM_LIBRARIES_REF
ARG ROCM_LIBRARIES_SHA
ARG ROCM_ARCH
ARG BUILD_PARALLEL_LEVEL
ARG GFX803_LINE

COPY --from=rocblas /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

# libfmt-dev: rocSOLVER does find_package(fmt REQUIRED) and has no
# install_deps.cmake to fetch it, and this base image ships no fmt.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=scripts/clone-sparse.sh,target=/scripts/clone-sparse.sh \
    /scripts/clone-sparse.sh https://github.com/ROCm/rocm-libraries.git /solver-src-root \
        "${ROCM_LIBRARIES_SHA:-${ROCM_LIBRARIES_REF}}" \
        projects/rocsolver shared

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-rocsolver \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/rocsolver.sh,target=/scripts/build/rocsolver.sh \
    --mount=type=bind,source=patches/rocsolver,target=/patches/rocsolver \
    /scripts/build/rocsolver.sh

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" rocsolver "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
