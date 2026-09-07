#!/bin/sh
# Apply the gfx803 Tensile patches, build rocBLAS for one architecture, and
# install it over /opt/rocm.
#
# Every patch file carries its own WHY header.
set -eu

ROOT=/rocblas-src-root
SRC="$ROOT/projects/rocblas"
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
. /scripts/lib/build-jobs.sh

sh /patches/rocblas/wgm-miscompute-source.sh "$ROOT"
sh /patches/rocblas/small-gemm-assembly-miscompute.sh "$SRC"
sh /patches/rocblas/tensile-gfx803-fp16-nond16.sh "$ROOT"

# The logic file is what makes Tensile generate the fp16 kernels the patch above
# adds codegen for.
cp /patches/rocblas/r9nano_Cijk_Ailk_Bljk_HB.yaml \
    "$SRC/library/src/blas3/Tensile/Logic/asm_full/r9nano/"

jobs="$(resolve_build_jobs)"
echo "rocBLAS build: arch $ARCH, $jobs parallel jobs"
cd "$SRC"
python3 ./rmake.py -i -a "$ARCH" -j "$jobs" --no_hipblaslt

# Per-directory readlink plus cp, not a plain `cp -a src/. dst/`. This base
# image's /opt/rocm/{bin,lib,include,share} are symlinks into /etc/alternatives
# rather than real directories, and cp refuses to merge a real directory over a
# destination that lstat()s as a symlink. Resolving each destination first
# sidesteps that.
echo "Copying rocBLAS $ARCH install output into /opt/rocm..."
install_dir="$SRC/build/release/rocblas-install"
for d in include lib share; do
    [ -e "$install_dir/$d" ] || continue
    real_dest="$(readlink -f "/opt/rocm/$d" 2>/dev/null || echo "/opt/rocm/$d")"
    mkdir -p "$real_dest"
    cp -a "$install_dir/$d/." "$real_dest/"
done
find "$install_dir" -mindepth 1 -maxdepth 1 ! -name include ! -name lib ! -name share \
    -exec cp -a {} /opt/rocm/ \;

# ! -type l so a sibling symlink cannot be picked instead of the real file.
built_real="$(find "$install_dir/lib" -maxdepth 1 -name 'librocblas.so.*' ! -type l | head -1)"
built_size="$(stat -c%s "$built_real" 2>/dev/null || echo 0)"
rm -rf "$SRC/build"

echo "Verifying the $ARCH Tensile library is present in /opt/rocm..."
if ! find -L /opt/rocm -iname "*TensileLibrary*${ARCH}*" | grep -q .; then
    echo "FATAL: /opt/rocm has no $ARCH Tensile library after the copy." >&2
    exit 1
fi

# A correct build that never gets loaded, because librocblas.so still resolves to
# the stock gfx900+ file, is the failure the 6.4.4 line lost days to. Compare
# sizes, not paths: the copy above made the same content exist under two names.
echo "Verifying librocblas.so resolves to the $ARCH build, not the stock one..."
resolved="$(readlink -f /opt/rocm/lib/librocblas.so)"
if [ "$built_size" = "0" ] || [ "$(stat -c%s "$resolved")" != "$built_size" ]; then
    echo "FATAL: /opt/rocm/lib/librocblas.so resolves to $resolved, which is not the build we just made." >&2
    echo "       The stock base image rocBLAS is what would load at runtime." >&2
    exit 1
fi

# The Tensile check only covers the Tensile .dat files, not rocBLAS's own HIP
# kernels. A link that succeeded with an empty .hip_fatbin happened on the 6.4.4
# line: it exits 0, gets pushed, and fails much later with "Illegal seek for GPU
# arch: gfx803".
echo "Verifying librocblas.so embeds real $ARCH device code..."
objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocblas_fatbin.bin
fatbin_size="$(stat -c%s /tmp/rocblas_fatbin.bin)"
rm -f /tmp/rocblas_fatbin.bin
if [ "$fatbin_size" -lt 1000000 ]; then
    echo "FATAL: librocblas.so's .hip_fatbin is only $fatbin_size bytes, too small to hold real $ARCH device code (expect several MB)." >&2
    exit 1
fi
echo "OK: librocblas.so is a $ARCH build with a ${fatbin_size}-byte .hip_fatbin."
