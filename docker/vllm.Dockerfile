# syntax=docker/dockerfile:1
#
# The gfx803 vLLM fork, built from the vendored source at vllm/ (see
# AGENTS.md, "vLLM lives in vllm/ as a hard fork"). No upstream checkout, no
# ref pin: the build context already is the pinned source.
#
# Needs no patched rocBLAS, MIOpen or rocSOLVER: vllm/CMakeLists.txt links only
# libamdhip64 at build time (the only find_package(HIP)-driven link target in
# that file), the same load-bearing fact that lets docker/triton.Dockerfile
# skip migraphx's tree too. python-base's own stock ROCm install already
# carries hipcc and the HIP headers that build needs. It does need PyTorch,
# because vllm/setup.py imports torch and compiles its extension against it,
# so it takes the trimmed pytorch-wheels image rather than the full one.
#
# Named "builder": final only ever takes the wheel and the three compiled
# gfx803 kernels from it (see the "wheels" stage below).
FROM python-base AS builder

ARG ROCM_ARCH
ARG BUILD_PARALLEL_LEVEL

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools
ENV PATH=/build-venv/bin:$PATH

COPY --from=pytorch /wheels/*.whl /tmp/torch/
RUN pip install --no-cache-dir /tmp/torch/*.whl && rm -rf /tmp/torch

# cmake, ninja, setuptools-rust, jinja2 and regex: vllm/setup.py and its cmake
# extension build need these. numpy, pyyaml and requests: vllm/setup.py reads
# them at import time before the extension build even starts, through
# vllm/envs.py and vllm/version.py. This mirrors the box's own build, which
# runs `pip3 install --no-build-isolation --no-deps -e .` against a venv that
# already carries the same set from an earlier, unrelated install -- see
# vllm/BUILD.md.
RUN pip install --no-cache-dir \
        "cmake>=3.26.1,<4" ninja "setuptools>=77.0.3,<81.0.0" setuptools-scm \
        setuptools-rust "jinja2>=3.1.6" regex numpy pyyaml requests packaging

COPY vllm /vllm-src

# --no-build-isolation: the build-venv above already carries every build
# dependency, and isolation would otherwise fetch its own torch from PyPI
# instead of using the gfx803 one just installed. --no-deps: this is a build
# step, not the final image's dependency resolution -- final-vllm-wheels.sh
# installs vllm's own runtime requirements once, against the full final venv,
# not this throwaway build-venv.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-vllm \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/vllm.sh,target=/scripts/build/vllm.sh \
    ROCM_ARCH="$ROCM_ARCH" BUILD_PARALLEL_LEVEL="$BUILD_PARALLEL_LEVEL" /scripts/build/vllm.sh

# The final image only ever takes /wheels/*.whl and /kernels/*.so from this
# stage. Everything else above -- the build-venv, the /vllm-src checkout, the
# compiled build tree -- exists only to produce those files. See
# pytorch.Dockerfile's own "wheels" stage for why this split exists.
FROM scratch AS wheels
COPY --from=builder /wheels /wheels
COPY --from=builder /kernels /kernels
