# syntax=docker/dockerfile:1
#
# PyTorch built from source for gfx803. No gfx803 wheel has ever been published
# on any ROCm line, so there is only the from-source path.
#
# It starts from migraphx because torch needs the gfx803 rocBLAS, MIOpen and
# rocSOLVER, and that target's /opt/rocm is the one that carries all of them.
#
# Named "builder": torchvision and torchaudio build on top of this stage's full
# tree (the compiled /opt/rocm, the venv, the /pytorch checkout), so it stays the
# image docker-bake.hcl's "pytorch" target publishes. The "wheels" stage below is
# a second, separate target for a second, much smaller published image -- see the
# comment there for why that split exists.
FROM python-base AS builder

ARG ROCM_ARCH
ARG PYTORCH_REF
ARG PYTORCH_SHA
ARG BUILD_PARALLEL_LEVEL
ARG TENSOR_TOPK_OPT_LEVEL
ARG GFX803_LINE

COPY --from=migraphx /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libopenblas-dev libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

# The wheel has to match the 3.12 venv the final image uses. --seed so that pip
# is directly callable.
RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools \
    && /build-venv/bin/pip install --no-cache-dir numpy pyyaml typing_extensions requests six build
ENV PATH=/build-venv/bin:$PATH

RUN --mount=type=bind,source=scripts/git-pin.sh,target=/git-pin \
    /git-pin /pytorch https://github.com/ROCm/pytorch.git "${PYTORCH_REF}" "${PYTORCH_SHA}" \
    && git -C /pytorch submodule sync --recursive \
    && git -C /pytorch submodule update --init --recursive --depth 1 --jobs 4

# C10_WARP_SIZE compiles to 32 for gfx803 because only __GFX9__ gets 64, but
# Polaris is wave64 only. Every eager kernel that warp-reduces through it drops
# half the wave and returns garbage. It is baked in at compile time, so it has to
# be fixed before the build rather than at runtime.
RUN --mount=type=bind,source=patches/pytorch,target=/patches/pytorch \
    bash /patches/pytorch/apply-gfx803-c10-warp-size-wave64.sh /pytorch

WORKDIR /pytorch
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 tools/amd_build/build_amd.py

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-pytorch \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/pytorch.sh,target=/scripts/build/pytorch.sh \
    /scripts/build/pytorch.sh

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" pytorch "${GFX803_SOURCE_REV}" "${GFX803_PINS}"

# The final image only ever takes /wheels/*.whl from this component (see
# final.Dockerfile). Everything else in the stage above -- the full ROCm SDK,
# apt build tools, the /pytorch checkout with submodules, the compiled build
# tree -- exists only so this wheel could get built, and every one of those
# bytes still had to be pulled and unpacked by any consumer of the "pytorch"
# tag, final included, even though it reads a single small directory. A build
# that pulls this trimmed "wheels" image instead pays for the wheel and
# nothing else. docker-bake.hcl publishes this stage under its own,
# separate tag; the "pytorch" tag above still publishes the full "builder"
# stage, because torchvision and torchaudio build against that one.
FROM scratch AS wheels
COPY --from=builder /wheels /wheels
