#!/bin/sh
# Build correctness-suite sweeps against a target image's MIOpen
# headers/libs. Run this *inside* a container that has hipcc + MIOpen
# installed (the images this repo builds all qualify), or point ROCM_PATH at
# a host ROCm install.
#
# Usage:
#   sh build.sh [output_dir] [rocm_path]
#   ONLY="reduce_sweep conv_solver_sweep" sh build.sh [output_dir] [rocm_path]
#
# ONLY (space-separated .cpp basenames, optional): build just these instead
# of all sweeps -- useful while iterating on a fix for one specific bug.
#
# Example (run from inside the target container, repo mounted at /work):
#   sh /work/gfx803/tools/correctness-suite/build.sh /tmp/suite-bin /opt/rocm-6.4.4
#   ONLY=reduce_sweep sh /work/gfx803/tools/correctness-suite/build.sh /tmp/suite-bin /opt/rocm-6.4.4
set -eu

SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
OUT="${1:-$SELF_DIR/bin}"
ROCM_PATH="${2:-/opt/rocm}"
ONLY="${ONLY:-}"

mkdir -p "$OUT"

# reduce_sweep.cpp, reduce_extreme_sweep.cpp, layernorm_sweep.cpp,
# groupnorm_sweep.cpp, glu_sweep.cpp, cat_sweep.cpp, rope_sweep.cpp and
# kthvalue_sweep.cpp use MIOpen's beta API surface, compiled behind #ifdef
# MIOPEN_BETA_API -- always on in official builds (see MIOpen's
# CMakeLists.txt), but the macro still needs to be defined for our own
# translation units to see those declarations.
BETA_API_FILES="reduce_sweep reduce_extreme_sweep layernorm_sweep groupnorm_sweep glu_sweep cat_sweep rope_sweep kthvalue_sweep"

built=0
for src in "$SELF_DIR"/*.cpp; do
    name="$(basename "$src" .cpp)"
    if [ -n "$ONLY" ]; then
        case " $ONLY " in
            *" $name "*) ;;
            *) continue ;;
        esac
    fi
    extra=""
    for b in $BETA_API_FILES; do
        [ "$name" = "$b" ] && extra="-DMIOPEN_BETA_API"
    done
    echo "building $name..."
    hipcc -O2 $extra -I"$ROCM_PATH/include" "$src" -o "$OUT/$name" \
        -lMIOpen -lm -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib"
    built=$((built + 1))
done

echo "built $built binaries in $OUT"
