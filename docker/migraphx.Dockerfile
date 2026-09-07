# syntax=docker/dockerfile:1
#
# MIGraphX built from source into /opt/rocm. No prebuilt package exists for
# gfx803. Two gfx803 patches fix build breaks that only appear with hipBLASLt and
# MLIR off, both of which are required here. See patches/migraphx/.
#
# pytorch, ort and final all inherit this target's whole /opt/rocm, so it has to
# carry every earlier component's fix.
FROM python-base

ARG ROCM_ARCH
ARG MIGRAPHX_REF
ARG MIGRAPHX_SHA
ARG BUILD_PARALLEL_LEVEL
ARG GFX803_LINE

COPY --from=rocblas /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

# Only the library files from miopen and rocsolver, never their whole trees:
# both of those start from an earlier point in the chain, so copying their
# /opt/rocm wholesale would revert rocBLAS back to stock.
#
# The source and destination are /opt/rocm/core-10.0/lib, not /opt/rocm/lib,
# because /opt/rocm/lib is a symlink to /etc/alternatives/rocm-lib and a COPY
# with a wildcard source does not follow a symlinked directory in the middle of
# the path. It silently matches zero files instead of failing, which `ls` and
# `readlink -f` inside a running container will not show you, because those do
# follow it. A ROCm version bump moves this path.
COPY --from=miopen /opt/rocm/core-10.0/lib/libMIOpen.so.* /opt/rocm/core-10.0/lib/
COPY --from=rocsolver /opt/rocm/core-10.0/lib/librocsolver.so.* /opt/rocm/core-10.0/lib/

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=scripts/git-pin.sh,target=/git-pin \
    /git-pin /migraphx-src https://github.com/ROCm/AMDMIGraphX.git \
        "${MIGRAPHX_REF}" "${MIGRAPHX_SHA}"

# rbuild is MIGraphX's own documented from-source build path. Its dependency
# cget crashes at import time under Python 3.13 and later, so it gets the
# uv-managed 3.12 rather than the system interpreter.
RUN uv venv /rbuild-venv --python 3.12 --seed \
    && /rbuild-venv/bin/pip install --no-cache-dir \
        https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-migraphx \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/migraphx.sh,target=/scripts/build/migraphx.sh \
    --mount=type=bind,source=patches/migraphx,target=/patches/migraphx \
    /scripts/build/migraphx.sh

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" migraphx "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
