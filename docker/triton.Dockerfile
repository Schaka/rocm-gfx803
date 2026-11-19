# syntax=docker/dockerfile:1
#
# Triton built from source for gfx803, with a self-built LLVM/MLIR and a
# self-built AMD code-generation library on its own LLVM (triton's setup.py
# otherwise downloads prebuilt ones).
#
# Triton has never targeted pre-RDNA/pre-CDNA GCN. Three patches make it work on
# gfx803: an ISA-family entry (Triton's AMD backend hard-rejects any arch whose
# family is Unknown), an unrelated compiler miscompile that hangs the GPU on a
# specific loop shape, and a gate on the packed v_dot intrinsics (without it an
# fp16 dot with f32 accumulation is a fatal LLVM abort, because gfx803 has no
# v_dot to select -- and that is the shape of the ROCm attention backend's
# prefill kernel, so no vLLM engine starts). See patches/triton/README.md for the
# hardware verification each one carries.
#
# Needs no other component's /opt/rocm: it links no ROCm library at build time,
# only ROCm C headers it vendors itself, and it resolves libamdhip64 and
# libhsa-runtime64 by dlopen at runtime, against whatever the final image
# provides.
#
# Named "builder": final only ever takes /wheels/*.whl from it (see the "wheels"
# stage below). Everything else here -- apt build tools and, above all, the two
# LLVM trees triton.sh compiles from source -- exists only to
# produce that one wheel and is easily this component's largest published
# content otherwise.
FROM python-base AS builder

ARG TRITON_REF
ARG BUILD_PARALLEL_LEVEL

# gcc builds both LLVM trees and the AMD codegen library (triton.sh says why),
# and clang builds triton itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build clang lld gcc g++ ccache \
    && rm -rf /var/lib/apt/lists/*

# triton.sh builds with --no-isolation, so this venv is all the build gets: the
# pins mirror the build-system requirements in triton's pyproject.toml, and
# triton's setup.py imports nanobind before it does anything else.
RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir \
        setuptools wheel "cmake>=3.20,<4.0" "ninja>=1.11.1" "nanobind==2.10.2" build
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
    --mount=type=cache,target=/root/.ccache-amd-codegen,id=gfx803-rocm10-triton-amd-codegen \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/triton.sh,target=/scripts/build/triton.sh \
    --mount=type=bind,source=patches/triton,target=/patches/triton \
    /scripts/build/triton.sh

# See pytorch.Dockerfile's own "wheels" stage for why this split exists.
FROM scratch AS wheels
COPY --from=builder /wheels /wheels
