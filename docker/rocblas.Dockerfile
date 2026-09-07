# syntax=docker/dockerfile:1
#
# rocBLAS built from source for gfx803, installed over /opt/rocm.
#
# gfx803 was dropped from the default TARGET_LIST at ROCm 6.0, but the Tensile
# logic for it is still in the tree, so `rmake.py -a gfx803` builds it back.
# Three gfx803 patches on top of that fix real miscompiles. See patches/rocblas/.
FROM python-base

ARG ROCM_LIBRARIES_REF
ARG ROCM_LIBRARIES_SHA
ARG ROCM_ARCH
ARG BUILD_PARALLEL_LEVEL
ARG GFX803_LINE

COPY --from=rocr-clr /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

# libmsgpack-dev is a transitional dummy package on Ubuntu 26.04. The real C++
# bindings Tensile's CMake looks for are in libmsgpack-cxx-dev.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config gfortran ccache \
        libmsgpack-cxx-dev wget \
    && rm -rf /var/lib/apt/lists/*

# Debian ships the package's CMake config as msgpack-cxx-*.cmake, but Tensile
# looks for a msgpackc-cxx package.
RUN mkdir -p /usr/local/lib/cmake/msgpackc-cxx \
    && for f in /usr/lib/x86_64-linux-gnu/cmake/msgpack-cxx/msgpack-cxx-*.cmake; do \
        ln -sf "$f" "/usr/local/lib/cmake/msgpackc-cxx/$(basename "$f" | sed 's/^msgpack-cxx/msgpackc-cxx/')"; \
    done

RUN pip3 install pyyaml joblib

# cmake and shared are pulled in beside projects/rocblas because rocBLAS's
# CMakeLists reaches up into both.
RUN --mount=type=bind,source=scripts/clone-sparse.sh,target=/scripts/clone-sparse.sh \
    /scripts/clone-sparse.sh https://github.com/ROCm/rocm-libraries.git /rocblas-src-root \
        "${ROCM_LIBRARIES_SHA:-${ROCM_LIBRARIES_REF}}" \
        cmake shared projects/rocblas

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-rocblas \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/rocblas.sh,target=/scripts/build/rocblas.sh \
    --mount=type=bind,source=patches/rocblas,target=/patches/rocblas \
    /scripts/build/rocblas.sh

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" rocblas "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
