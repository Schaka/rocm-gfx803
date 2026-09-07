# syntax=docker/dockerfile:1
#
# MIOpen built from source for gfx803, installed over /opt/rocm.
#
# MIOpen still lists gfx803 in ALL_GPU_DATABASES and keeps its Polaris
# device-name gating, so it builds for the architecture. Two gfx803 patches on
# top of that fix real miscomputes. See patches/miopen/.
#
# This starts from rocr-clr, not from rocblas, so its /opt/rocm carries the
# patched runtime but no rocBLAS. Only the library file is taken forward
# downstream, never this whole tree.
FROM python-base

ARG ROCM_LIBRARIES_REF
ARG ROCM_LIBRARIES_SHA
ARG ROCM_ARCH
ARG BUILD_PARALLEL_LEVEL
ARG GFX803_LINE

COPY --from=rocr-clr /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

# rocm-cmake is not an apt package on this base image, and MIOpen's own
# install_deps.cmake fetches it through cget anyway. "half" is libhalf-dev on
# this Ubuntu release.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libhalf-dev libboost-system-dev libboost-filesystem-dev \
        libsqlite3-dev libbz2-dev lbzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=scripts/clone-sparse.sh,target=/scripts/clone-sparse.sh \
    /scripts/clone-sparse.sh https://github.com/ROCm/rocm-libraries.git /miopen-src-root \
        "${ROCM_LIBRARIES_SHA:-${ROCM_LIBRARIES_REF}}" \
        projects/miopen shared

# Neither composable_kernel nor rocMLIR has ever supported gfx8.
#
# The PATH override puts the uv-managed 3.12 first because install_deps.cmake
# shells out to whatever python3 it finds, and cget crashes at import time under
# 3.13 and later: it references urllib.request.FancyURLopener, which those
# releases removed.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-miopen \
    cd /miopen-src-root/projects/miopen \
    && sed -i '/composable_kernel/d; /rocMLIR/d' requirements.txt \
    && PATH="$(dirname "$(uv python find 3.12)"):$PATH" \
        cmake -P install_deps.cmake --minimum --prefix /miopen-deps

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-miopen \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/miopen.sh,target=/scripts/build/miopen.sh \
    --mount=type=bind,source=patches/miopen,target=/patches/miopen \
    /scripts/build/miopen.sh

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" miopen "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
