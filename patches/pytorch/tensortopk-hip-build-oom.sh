#!/bin/sh
# See tensortopk-hip-build-oom.patch for the full WHY/WHAT. Uses `patch`,
# not `git apply` -- see ../rocblas/small-gemm-assembly-miscompute.sh for why.
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/tensortopk-hip-build-oom.patch"
CMAKELISTS="$SRC/caffe2/CMakeLists.txt"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$CMAKELISTS" ] || { echo "FATAL: $CMAKELISTS does not exist -- upstream moved this file" >&2; exit 1; }

if grep -q 'TENSORTOPK_HIP_MEMORY_WORKAROUND_PATCH' "$CMAKELISTS"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --verbose < "$PATCH"

if ! grep -q 'TENSORTOPK_HIP_MEMORY_WORKAROUND_PATCH' "$CMAKELISTS"; then
    echo "FATAL: marker not found in $CMAKELISTS after patch reported success" >&2
    exit 1
fi
echo "TensorTopK.hip build-memory workaround applied and verified in $CMAKELISTS"
