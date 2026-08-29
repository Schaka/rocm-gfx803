#!/bin/sh
# See mlir-stub-missing-symbols.patch for the full WHY/WHAT. Uses `patch`,
# not `git apply` -- see ../rocblas/small-gemm-assembly-miscompute.sh for why.
set -eu
SRC="${1:-/migraphx-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/mlir-stub-missing-symbols.patch"
CPP="$SRC/src/targets/gpu/mlir.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$CPP" ] || { echo "FATAL: $CPP does not exist -- upstream moved this file" >&2; exit 1; }

if grep -q 'IS_MODULE_FUSIBLE_MLIR_STUB_PATCH' "$CPP"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$SRC" --verbose < "$PATCH"

if ! grep -q 'IS_MODULE_FUSIBLE_MLIR_STUB_PATCH' "$CPP"; then
    echo "FATAL: marker not found in $CPP after patch reported success" >&2
    exit 1
fi
echo "MLIR-disabled stub symbols (is_module_fusible, dump_mlir_to_file, dump_mlir_to_mxr, adjust_param_shapes) added and verified in $CPP"
