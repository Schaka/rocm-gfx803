#!/bin/sh
# Build a torchvision or torchaudio wheel against the torch already installed in
# this image.
#
# No prebuilt wheel has ever been published for gfx803, for either package, so
# there is only the from-source path and no tier to choose between.
#
# usage: torch-companion.sh <source-dir>
set -eu

SRC="$1"
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"

cd "$SRC"
env USE_ROCM=1 USE_CUDA=0 "PYTORCH_ROCM_ARCH=$ARCH" FORCE_CUDA=0 \
    python3 setup.py bdist_wheel

mkdir -p /wheels
cp dist/*.whl /wheels/
ls -l /wheels
