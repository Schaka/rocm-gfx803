#!/bin/sh
# Apply mlir-stub-missing-symbols.patch. That file gives the reason and the
# change. This driver uses `patch` and not `git apply`, for the reason stated in
# ../rocblas/small-gemm-assembly-miscompute.sh.
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
