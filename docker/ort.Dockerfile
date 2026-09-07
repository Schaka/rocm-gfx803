# syntax=docker/dockerfile:1
#
# ONNX Runtime built with the MIGraphX execution provider only.
#
# ORT's own ROCm execution provider was deleted upstream after v1.22.2, and
# --use_rocm is gone from its build flags as of 1.28, so MIGraphX is the only
# provider available on any current version. Accepted cost: gfx803 has no
# Composable Kernel or MLIR fusion behind that provider.
#
# Named "builder": nothing chains off this stage the way torchvision/torchaudio
# chain off pytorch, but final only ever takes the wheel from
# /onnxruntime/dist (see the "wheels" stage below), never this stage's own
# /opt/rocm -- the comment on that COPY above says so directly. Everything else
# here (the full copied ROCm tree, apt build tools, the /onnxruntime checkout and
# its build directory) exists only to produce that one wheel.
FROM python-base AS builder

ARG ROCM_ARCH
ARG ORT_VERSION
ARG ORT_SHA
ARG GFX803_LINE

COPY --from=migraphx /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

# ROCm 10.0 ships flatbuffers v25 in /opt/rocm, and the MIGraphX provider sets
# CMAKE_PREFIX_PATH=/opt/rocm. ORT's FetchContent declaration treats its own
# v23.5.26 pin as a minimum, so it find_package()s the v25 config instead of
# downloading v23, and the v25 headers then fail ORT's generated-schema
# static_assert. Removing the ROCm flatbuffers makes find_package fail so that
# FetchContent downloads the right version. Nothing here uses the ROCm one, and
# this target's /opt/rocm never reaches the final image.
RUN rm -rf /opt/rocm/include/flatbuffers \
        /opt/rocm/lib/cmake/flatbuffers \
        /opt/rocm/lib/libflatbuffers.a \
        /opt/rocm/lib/pkgconfig/flatbuffers.pc

# libdrm-dev: the ROCm provider's find_package(rocm_smi) needs libdrm's
# pkg-config file, which only the -dev package ships.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libprotobuf-dev protobuf-compiler libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools \
    && /build-venv/bin/pip install --no-cache-dir numpy packaging cmake
ENV PATH=/build-venv/bin:$PATH

RUN --mount=type=bind,source=scripts/git-pin.sh,target=/git-pin \
    /git-pin /onnxruntime https://github.com/microsoft/onnxruntime.git "${ORT_VERSION}" "${ORT_SHA}" \
    && git -C /onnxruntime submodule sync --recursive \
    && git -C /onnxruntime submodule update --init --recursive --jobs 4

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-ort \
    --mount=type=bind,source=scripts/build/ort.sh,target=/scripts/build/ort.sh \
    /scripts/build/ort.sh

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" ort "${GFX803_SOURCE_REV}" "${GFX803_PINS}"

# See pytorch.Dockerfile's own "wheels" stage for why this split exists.
FROM scratch AS wheels
COPY --from=builder /onnxruntime/dist /onnxruntime/dist
