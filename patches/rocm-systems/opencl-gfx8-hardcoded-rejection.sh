#!/bin/sh
# Apply opencl-gfx8-hardcoded-rejection.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunks actually landed.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/opencl-gfx8-hardcoded-rejection.patch"
DEVICE_FILE="$SRC/projects/clr/rocclr/device/device.hpp"
ROCDEVICE_FILE="$SRC/projects/clr/rocclr/device/rocm/rocdevice.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

if grep -q "versionMajor_ == 8" "$DEVICE_FILE"; then
    echo "FATAL: versionMajor_ == 8 OpenCL gate still present in $DEVICE_FILE after git apply reported success" >&2
    exit 1
fi
if ! grep -q "image_queries_ok" "$ROCDEVICE_FILE"; then
    echo "FATAL: image_queries_ok fallback not found in $ROCDEVICE_FILE after git apply reported success" >&2
    exit 1
fi
echo "OpenCL gfx8-rejection patch applied and verified in $DEVICE_FILE and $ROCDEVICE_FILE"
