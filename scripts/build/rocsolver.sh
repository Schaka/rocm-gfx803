#!/bin/sh
# Apply the gfx803 wavesize patch, build rocSOLVER for one architecture, and
# install the resulting library over /opt/rocm.
#
# The stack's own rocSOLVER ships an empty .hip_fatbin for every architecture, so
# every hipSOLVER backed torch.linalg entry point crashes at its first launch.
# The patch file carries its own WHY header.
set -eu

SRC=/solver-src-root/projects/rocsolver
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
. /scripts/lib/build-jobs.sh

sh /patches/rocsolver/rocsolver-wavesize-gfx8.sh "$SRC"

jobs="$(resolve_build_jobs)"
echo "rocSOLVER build: arch $ARCH, $jobs parallel jobs"

# CMAKE_CXX_COMPILER=hipcc is required, not stylistic. rocSOLVER's sources are
# .cpp files carrying __global__ kernels, and they take their offload flags from
# hip::device rather than by switching the language. Under CMake's default
# compiler every one of them dies on "unrecognised command-line option
# --offload-arch=gfx803" from g++. USE_HIPCXX=ON is not an alternative:
# check_language(HIP) fails on this base image.
cmake -S "$SRC" -B /rs-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_PATH=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DAMDGPU_TARGETS="$ARCH" \
    -DBUILD_WITH_SPARSE=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF \
    -DBUILD_CLIENTS_SAMPLES=OFF \
    -DBUILD_OFFLOAD_COMPRESS=OFF -DBUILD_COMPRESSED_DBG=OFF
cmake --build /rs-build -j"$jobs"
cp -a /rs-build/library/src/librocsolver.so* /tmp/
rm -rf /rs-build

echo "Copying the rocSOLVER $ARCH build into /opt/rocm..."
resolved="$(readlink -f /opt/rocm/lib/librocsolver.so)"
stock_size="$(stat -c%s "$resolved" 2>/dev/null || echo 0)"
built="$(find /tmp -maxdepth 1 -iname 'librocsolver.so.*' ! -iname '*.so' -type f | head -1)"
cp -a "$built" "$resolved"
rm -f /tmp/librocsolver.so*

new_size="$(stat -c%s "$resolved")"
echo "librocsolver resolved path: $resolved (stock $stock_size bytes, new $new_size bytes)"
if [ "$new_size" = "$stock_size" ]; then
    echo "FATAL: librocsolver is unchanged from stock at $new_size bytes, so our build did not land." >&2
    exit 1
fi

echo "Verifying librocsolver.so embeds real $ARCH device code..."
objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocsolver_fatbin.bin
fatbin_size="$(stat -c%s /tmp/rocsolver_fatbin.bin)"
rm -f /tmp/rocsolver_fatbin.bin
if [ "$fatbin_size" -lt 100000 ]; then
    echo "FATAL: librocsolver.so's .hip_fatbin is only $fatbin_size bytes, too small to hold real $ARCH device code (expect MBs)." >&2
    exit 1
fi
echo "OK: librocsolver.so is a $ARCH build with a ${fatbin_size}-byte .hip_fatbin."
