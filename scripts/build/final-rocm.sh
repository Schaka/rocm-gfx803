#!/bin/sh
# Put the last gfx803 pieces into /opt/rocm and prove that all of them landed.
#
# Every component reaches this image as a prebuilt image, so a stale or wrongly
# wired one is the failure mode this guards against. Each check below names the
# input that is wrong when it fires.
set -eu

# A component image from another ROCm line assembles, imports, and misbehaves
# only on real hardware, so fail at the first layer that inherits a whole tree.
if [ ! -d /opt/rocm/core-10.0 ]; then
    echo "FATAL: the inherited /opt/rocm has no core-10.0 (found: $(ls -d /opt/rocm/core-* 2>/dev/null | tr '\n' ' '))." >&2
    echo "       A component image from a different ROCm line is wired into this build." >&2
    exit 1
fi

# The three HIP runtime libraries must be our own rocr-clr build, but the base
# image and every published chain image can still carry a stock twin under the
# same SONAME. The base ships "-0000000", and our build ships "-<commit>" under
# CI and "-0000000" locally, so a filename pattern cannot tell them apart and one
# that assumes it can deletes the wrong file. Install from the rocr-clr image's
# own lib directory instead, which holds exactly one real file per family.
for n in libamdhip64 libhiprtc libhiprtc-builtins; do
    ours="$(find /opt/rocm-clr-lib -maxdepth 1 -name "${n}.so.7.*" -type f | head -1)"
    if [ -z "$ours" ]; then
        echo "FATAL: $n is missing from the rocr-clr image." >&2
        exit 1
    fi
    rm -f "/opt/rocm/lib/${n}.so" "/opt/rocm/lib/${n}.so.7" "/opt/rocm/lib/${n}".so.7.*
    cp -a "$ours" "/opt/rocm/lib/$(basename "$ours")"
    ln -sf "$(basename "$ours")" "/opt/rocm/lib/${n}.so.7"
    ln -s "${n}.so.7" "/opt/rocm/lib/${n}.so"
done
rm -rf /opt/rocm-clr-lib
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
ldconfig

if ! strings "$(readlink -f /opt/rocm/lib/libamdhip64.so)" 2>/dev/null | grep -q "Image extension queries failed"; then
    echo "FATAL: the active libamdhip64.so does not carry the gfx8 opencl patch marker." >&2
    echo "       The rocr-clr image wired into this build does not carry the gfx803 fix." >&2
    exit 1
fi

# MIOpen has no .hip_fatbin to size-check: its kernels are compiled on demand and
# served through its own kernel database, and Composable Kernel, which would add
# precompiled binaries, is off. Compare against the untouched base image instead.
# A before-and-after comparison around the copy does not work, because the
# migraphx image already carries the same fixed file forward.
resolved="$(readlink -f /opt/rocm/lib/libMIOpen.so)"
stock_ref="$(find /tmp/miopen-stock-ref -maxdepth 1 -name 'libMIOpen.so.*' -type f | sort -V | tail -1)"
if [ -z "$resolved" ] || [ ! -f "$resolved" ] || [ -z "$stock_ref" ]; then
    echo "FATAL: could not resolve libMIOpen.so ('$resolved') or the stock reference ('$stock_ref')." >&2
    exit 1
fi
new_size="$(stat -c%s "$resolved")"
stock_size="$(stat -c%s "$stock_ref")"
rm -rf /tmp/miopen-stock-ref
if [ "$new_size" = "$stock_size" ]; then
    echo "FATAL: $resolved is the same size ($new_size bytes) as the untouched base image's MIOpen." >&2
    echo "       The miopen image wired into this build does not carry the gfx803 fix." >&2
    exit 1
fi
echo "OK: MIOpen at $resolved ($new_size bytes) differs from the stock $stock_size bytes."

resolved="$(readlink -f /opt/rocm/lib/librocsolver.so)"
if [ -z "$resolved" ] || [ ! -f "$resolved" ]; then
    echo "FATAL: /opt/rocm/lib/librocsolver.so does not resolve to a real file." >&2
    exit 1
fi
objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocsolver_fatbin.bin
fatbin_size="$(stat -c%s /tmp/rocsolver_fatbin.bin)"
rm -f /tmp/rocsolver_fatbin.bin
if [ "$fatbin_size" -lt 100000 ]; then
    echo "FATAL: $resolved's .hip_fatbin is only $fatbin_size bytes, too small to hold real gfx803 device code." >&2
    echo "       The rocsolver image wired into this build does not carry the gfx803 fix." >&2
    exit 1
fi
echo "OK: rocSOLVER at $resolved has a ${fatbin_size}-byte .hip_fatbin."

if ! strings /opt/rocm/lib/libgfx803_sgemm_shim.so | grep -q "f16-map-nm"; then
    echo "FATAL: the installed SGEMM shim does not carry the fp16 operand-mapping fix." >&2
    exit 1
fi
echo "OK: the SGEMM shim carries the fp16 mapping fix."
