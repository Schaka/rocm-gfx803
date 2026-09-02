#!/bin/sh
# Apply va-reuse-defer-noremap.patch (see that file for the full WHY/WHAT).
# Drop the va-reuse-defer park branch's _fmm_map_to_gpu re-map: it leaves a
# kernel GPUVM mapping behind that the fmm allocator then re-hands out, so
# every code-object load on gfx803 fails with the kernel rejecting the VA
# (EINVAL -> HSA_STATUS_ERROR_OUT_OF_RESOURCES -> vLLM 10.0 SIGSEGV).
# Must run AFTER va-reuse-defer.sh.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/va-reuse-defer-noremap.patch"
FILE="$SRC/projects/rocr-runtime/libhsakmt/src/fmm.c"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE missing" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

grep -q 'AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &args);' "$FILE" || {
    echo "FATAL: unmap-before-free marker not found in $FILE after git apply reported success" >&2
    exit 1
}
grep -q '_fmm_map_to_gpu(ctx, aperture, object->start,' "$FILE" && {
    echo "FATAL: park-branch re-map still present in $FILE" >&2
    exit 1
}
echo "VA-reuse-defer noremap patch applied and verified"
