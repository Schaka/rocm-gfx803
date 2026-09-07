#!/bin/sh
# Apply the gfx803 patches, build MIOpen for one architecture, and install the
# resulting library over /opt/rocm.
#
# Every patch file carries its own WHY header.
set -eu

SRC=/miopen-src-root/projects/miopen
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
. /scripts/lib/build-jobs.sh

sh /patches/miopen/winograd-fused-conv-miscompute.sh "$SRC"
sh /patches/miopen/reduce-prod-wrong-identity.sh "$SRC"

jobs="$(resolve_build_jobs)"
echo "MIOpen build: arch $ARCH, $jobs parallel jobs"
cd "$SRC"

# -Wno-error=#warnings: amdclang++ resolves this base image's system libstdc++
# (GCC 15) for standard headers. GCC 15's <ciso646> self-deprecates through a
# #warning pragma, which clang turns into a hard error under MIOpen's -Werror.
# Downgrade that one diagnostic class rather than disabling -Werror broadly.
mkdir -p build
cd build
CXX=/opt/rocm/bin/amdclang++ cmake .. \
    -DCMAKE_PREFIX_PATH=/miopen-deps \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="$ARCH" \
    -DMIOPEN_BACKEND=HIP \
    -DMIOPEN_USE_COMPOSABLEKERNEL=Off \
    -DMIOPEN_USE_MLIR=Off \
    -DMIOPEN_USE_HIPBLASLT=Off \
    -DMIOPEN_BUILD_DRIVER=Off \
    -DBUILD_TESTING=Off \
    "-DCMAKE_CXX_FLAGS=-Wno-error=#warnings"
make -j"$jobs"
cp -a lib/libMIOpen.so* /tmp/
cd ..
rm -rf build

echo "Copying the MIOpen $ARCH build into /opt/rocm..."
resolved="$(readlink -f /opt/rocm/lib/libMIOpen.so)"
stock_size="$(stat -c%s "$resolved" 2>/dev/null || echo 0)"
# -type f: the glob also matches libMIOpen.so.1, a symlink whose name does not
# end in .so, so the name filter alone lets it through. find's directory order
# can then hand `head -1` a 16-byte symlink instead of the real library.
built="$(find /tmp -maxdepth 1 -iname 'libMIOpen.so.*' ! -iname '*.so' -type f | head -1)"
cp -a "$built" "$resolved"
rm -f /tmp/libMIOpen.so*

new_size="$(stat -c%s "$resolved")"
echo "libMIOpen resolved path: $resolved (stock $stock_size bytes, new $new_size bytes)"
if [ "$new_size" -lt 10000000 ]; then
    echo "FATAL: libMIOpen is only $new_size bytes, too small to be a real build (expect 100+MB)." >&2
    exit 1
fi
if [ "$new_size" = "$stock_size" ]; then
    echo "FATAL: libMIOpen is unchanged from stock at $new_size bytes, so our build did not land." >&2
    exit 1
fi
echo "OK: the MIOpen $ARCH build is in place."
