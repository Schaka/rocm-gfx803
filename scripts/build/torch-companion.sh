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
# FORCE_CUDA=1: torchvision compiles its HIP ops (nms, roi_align, deform_conv2d
# and the rest) only when torch sees a GPU at build time or FORCE_CUDA=1 is set,
# and the build container has no GPU. torchaudio ignores it and reads USE_ROCM.
env USE_ROCM=1 USE_CUDA=0 "PYTORCH_ROCM_ARCH=$ARCH" FORCE_CUDA=1 \
    python3 setup.py bdist_wheel

mkdir -p /wheels
cp dist/*.whl /wheels/
ls -l /wheels

# A torchvision wheel without HIP ops imports and passes every CPU check. It
# fails only once a GPU tensor reaches one of its ops, so catch it here.
for whl in /wheels/torchvision-*.whl; do
    [ -e "$whl" ] || continue
    check_dir="$(mktemp -d)"
    python3 -m zipfile -e "$whl" "$check_dir"
    if ! readelf -d "$check_dir"/torchvision/_C*.so | grep -Eq 'libamdhip64|libc10_hip'; then
        echo "FATAL: $whl has a torchvision/_C.so without HIP linkage, so it carries no GPU ops." >&2
        exit 1
    fi
    rm -rf "$check_dir"
    echo "OK: $whl carries HIP ops."
done
