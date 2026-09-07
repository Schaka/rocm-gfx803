#!/bin/sh
# Apply the gfx803 patches to rocm-systems, then build ROCR-Runtime and CLR over
# the base image's own /opt/rocm.
#
# Every patch file carries its own WHY header. Only the ordering rules are
# repeated here, because the patch files cannot state them.
set -eu

SRC=/rocm-systems-src
. /scripts/lib/build-jobs.sh

# This order is load-bearing in three places. The three va-reuse-defer patches
# build on each other. d2h-null-dsthost fixes a destination address that
# d2h-staged-copy introduces. aql-ring-queue-full-workaround multiplies the legacy
# doorbell mask that hsa-agent-rejects-legacy-doorbell restores, so it has to see
# that mask already in place.
for p in \
    hsa-agent-rejects-legacy-doorbell \
    opencl-gfx8-hardcoded-rejection \
    sdma-doorbell-missing-sfence \
    va-reuse-defer \
    va-reuse-defer-mapping \
    va-reuse-defer-noremap \
    pinned-release-system-scope \
    graph-replay-batch-chunk-deadlock \
    d2h-staged-copy \
    d2h-null-dsthost \
    aql-ring-queue-full-workaround \
    gfx803-tc-invalidate-acquire-mem
do
    sh "/patches/rocm-systems/${p}.sh" "$SRC"
done

jobs="$(resolve_build_jobs)"

# ROCR-Runtime first, because CLR's HIP build links against it and needs the
# patched runtime already installed in /opt/rocm.
echo "ROCR-Runtime build: $jobs parallel jobs"
rocr_build="$SRC/projects/rocr-runtime/build"
cmake -S "$SRC/projects/rocr-runtime" -B "$rocr_build" \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_BUILD_TYPE=Release
make -C "$rocr_build" -j"$jobs"
make -C "$rocr_build" install
rm -rf "$rocr_build"

# HIP only. The whole stack above this (rocBLAS, MIOpen, MIGraphX, PyTorch, ORT)
# is HIP based. The OpenCL patch is still applied, so turning CLR_BUILD_OCL back
# on stays a one-line change.
echo "CLR (HIP) build: $jobs parallel jobs"
clr_build="$SRC/projects/clr/build"
cmake -S "$SRC/projects/clr" -B "$clr_build" \
    -DHIP_COMMON_DIR="$SRC/projects/hip" \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCLR_BUILD_HIP=ON \
    -DCLR_BUILD_OCL=OFF \
    -DHIP_PLATFORM=amd
make -C "$clr_build" -j"$jobs"
make -C "$clr_build" install
rm -rf "$clr_build"

# Collapse each HIP runtime library family to exactly one real file. The suffix
# that `make install` writes comes from the build id: a commit hash under CI, the
# literal -0000000 on a machine that supplies none, which is also what the base
# image's own stock files are called. So the name cannot identify our file later
# on. It can identify it here, where the freshest mtime is proof of what this
# install just wrote.
cd /opt/rocm/lib
for n in libamdhip64 libhiprtc libhiprtc-builtins; do
    ours="$(find . -maxdepth 1 -name "${n}.so.7.*" -type f -printf '%T@ %f\n' | sort -rn | head -1 | cut -d' ' -f2)"
    if [ -z "$ours" ]; then
        echo "FATAL: no real ${n}.so.7.* file after install." >&2
        exit 1
    fi
    mv "$ours" "/tmp/${n}.keep"
    rm -f "${n}.so" "${n}.so.7" ${n}.so.7.*
    mv "/tmp/${n}.keep" "./${ours}"
    ln -sf "${ours}" "${n}.so.7"
    ln -sf "${n}.so.7" "${n}.so"
    echo "single ${n} build: ${ours}"
done

# Both markers are strings the patches add. Their absence means the build linked
# the stock libraries back over ours.
if strings /opt/rocm/lib/libhsa-runtime64.so* 2>/dev/null | grep -q "deprecated doorbell type"; then
    echo "FATAL: installed libhsa-runtime64.so still throws on DoorbellType != 2." >&2
    exit 1
fi
if ! strings "$(readlink -f /opt/rocm/lib/libamdhip64.so)" 2>/dev/null | grep -q "Image extension queries failed"; then
    echo "FATAL: active libamdhip64.so does not carry the gfx8 opencl patch marker." >&2
    exit 1
fi
echo "OK: patched ROCR-Runtime and CLR installed."
/opt/rocm/bin/hipconfig --version
