# syntax=docker/dockerfile:1
#
# Patched ROCR-Runtime and CLR, installed over the base image's own /opt/rocm.
#
# ROCm 7 and later reject Polaris at HSA agent creation, because the GpuAgent
# constructor throws for any agent whose DoorbellType is not 2 and Polaris
# reports a legacy type. Restoring enumeration and real dispatch needs the full
# legacy-doorbell code path, so this is a source rebuild rather than a flag.
# Everything downstream inherits the patched runtime through the normal /opt/rocm
# paths.
FROM python-base

ARG ROCM_SYSTEMS_REF
ARG ROCM_SYSTEMS_SHA
ARG BUILD_PARALLEL_LEVEL

# CLR's README asks for rocm-llvm-dev, an apt package from the classic apt-based
# ROCm install. This base image has no ROCm apt repo at all, and the compiler CLR
# needs is already in /opt/rocm/llvm.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config \
        libnuma-dev libdrm-dev libelf-dev xxd \
        libgl1-mesa-dev libx11-dev mesa-common-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install CppHeaderParser

RUN --mount=type=bind,source=scripts/clone-sparse.sh,target=/scripts/clone-sparse.sh \
    /scripts/clone-sparse.sh https://github.com/ROCm/rocm-systems.git /rocm-systems-src \
        "${ROCM_SYSTEMS_SHA:-${ROCM_SYSTEMS_REF}}" \
        projects/rocr-runtime projects/clr projects/hip

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-rocr-clr \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/rocr-clr.sh,target=/scripts/build/rocr-clr.sh \
    --mount=type=bind,source=patches/rocm-systems,target=/patches/rocm-systems \
    /scripts/build/rocr-clr.sh

ARG GFX803_LINE GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" rocr-clr "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
