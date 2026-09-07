# syntax=docker/dockerfile:1
#
# Triton built from source for gfx803, with a self-built LLVM/MLIR (triton's
# setup.py otherwise downloads a prebuilt one that carries none of these fixes).
#
# Triton has never targeted pre-RDNA/pre-CDNA GCN. Three patches make it work on
# gfx803: an ISA-family entry (Triton's AMD backend hard-rejects any arch whose
# family is Unknown), a DPP lowering fix (the fallback path Triton picks for
# non-CDNA/non-RDNA targets uses an instruction gfx803 does not have), and an
# unrelated compiler miscompile that hangs the GPU on a specific loop shape. See
# patches/triton/README.md for the hardware verification each one carries.
#
# Needs no other component's /opt/rocm: it links no ROCm library at build time,
# only ROCm C headers it vendors itself, and it resolves libamdhip64 and
# libhsa-runtime64 by dlopen at runtime, against whatever the final image
# provides.
FROM python-base

ARG TRITON_REF
ARG BUILD_PARALLEL_LEVEL

# clang and lld build LLVM itself, not just link against it -- LLVM/MLIR's own
# Release build is the expensive part of this stage, and clang is
# meaningfully faster at it than gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build clang lld gcc g++ ccache python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir \
        setuptools wheel "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" build
ENV PATH=/build-venv/bin:$PATH
ENV CC=clang
ENV CXX=clang++

COPY scripts/git-pin.sh /git-pin

# No gfx803-line stamp here: unlike every other component, this stage inherits
# no /opt/rocm from an earlier stage (it links no ROCm library at build time),
# and nothing downstream inherits this stage's own /opt/rocm either -- final
# takes only /wheels/*.whl from it. There is no tree whose lineage needs
# recording.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-triton-llvm \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/triton.sh,target=/scripts/build/triton.sh \
    --mount=type=bind,source=patches/triton,target=/patches/triton \
    /scripts/build/triton.sh
