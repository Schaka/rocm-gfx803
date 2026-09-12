#!/bin/sh
# Build the gfx803 vLLM wheel and the three hand-written gfx803 HIP kernels.
#
# Produces /wheels/*.whl (the vllm package and its C++/HIP extension) and
# /kernels/<relative path under site-packages/vllm>/*.so (the three kernels),
# kept in the same directory shape their Python ctypes loaders expect, so
# final.Dockerfile can copy each one straight onto the installed vllm package
# with no per-file name mapping to keep in sync. See vllm/BUILD.md for what
# each kernel does and the exact hipcc command this mirrors.
set -eu

ROCM_ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
. /scripts/lib/build-jobs.sh

cd /vllm-src

jobs="$(resolve_build_jobs)"
echo "vLLM build: arch $ROCM_ARCH, $jobs parallel jobs"

# VLLM_VERSION_OVERRIDE: this vendored tree carries no .git of its own (see
# AGENTS.md, "vLLM lives in vllm/ as a hard fork"), so setuptools-scm has no
# tag or commit history to derive a version from and fails outright. The
# override matches what the box-native editable install already reports (see
# NOTES.md's vLLM 0.28.0 migration item), so both build paths name the same
# version until that migration happens.
env "MAX_JOBS=$jobs" "PYTORCH_ROCM_ARCH=$ROCM_ARCH" "VLLM_VERSION_OVERRIDE=0.20.1+gfx803" \
    pip wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels .

hipcc="/opt/rocm/bin/hipcc"
kernels="/vllm-src/vllm/gfx803_kernels"

mkdir -p /kernels/model_executor/layers /kernels/v1/attention/ops

"$hipcc" --offload-arch="$ROCM_ARCH" -O3 -shared -fPIC \
    -o /kernels/model_executor/layers/libgfx803gemm.so \
    "$kernels/gfx803_gemm_lib.hip"

"$hipcc" --offload-arch="$ROCM_ARCH" -O3 -shared -fPIC \
    -o /kernels/model_executor/layers/libgfx803gemv_m.so \
    "$kernels/gfx803_gemv_m.hip"

"$hipcc" --offload-arch="$ROCM_ARCH" -O3 -shared -fPIC \
    -o /kernels/v1/attention/ops/libgfx803attn.so \
    "$kernels/gfx803_attn_split.hip"
