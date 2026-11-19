#!/bin/sh
# Build the two LLVMs triton needs and its AMD code-generation library, apply
# the gfx803 patches, and build the triton wheel.
#
# Triton pins exact LLVM commits rather than branches, because its lowering
# passes are written against one specific LLVM/MLIR API surface. Both pins are
# read from the checkout instead of hand-copied here, so a TRITON_REF bump
# always builds what its own source tree expects:
#   - cmake/llvm-info.json: the LLVM/MLIR that libtriton links.
#   - cmake/amd-llvm-info.json: a separate LLVM that only the AMD
#     code-generation library (third_party/amd/backend/codegen) links. The AMD
#     backend hands its finished LLVM IR to that library through ctypes to turn
#     it into gfx ISA, so gfx803's instruction selection runs in this second
#     LLVM, not in the first.
#
# Not cmake/llvm-build-info.json or cmake/amd-llvm-build-info.json: those name
# what triton's own .github/workflows/llvm-build.yml pre-builds, which can be
# the candidate for the next LLVM bump rather than the commit this source tree
# is written against. A build against such a newer candidate has failed with a
# real compile error in triton's MLIR dialect headers, not with a stale patch.
set -eu

TRITON_REF="${TRITON_REF:?TRITON_REF is required}"
. /scripts/lib/build-jobs.sh

# json_field <file> <key> [default]: no default makes a missing key fatal.
json_field() {
    python3 -I -c '
import json, sys
data = json.load(open(sys.argv[1]))
print(data[sys.argv[2]] if len(sys.argv) < 4 else data.get(sys.argv[2], sys.argv[3]))
' "$@"
}

/git-pin /triton-src https://github.com/triton-lang/triton.git "$TRITON_REF" "$TRITON_REF"
git -C /triton-src submodule sync --recursive
git -C /triton-src submodule update --init --recursive --depth 1

# bash, not sh: these drivers are written against bash (${BASH_SOURCE[0]},
# [[ ]]), unlike every other patch driver in this repo.
#
# gfx803-isa-family.patch goes first: it is what makes the AMD backend accept
# gfx803 at all, and gfx803-vdot-gate.patch keeps one of its lines as context.
bash /patches/triton/apply-gfx803-isa-family.sh /triton-src
bash /patches/triton/apply-fold-true-cmpi-while-nested-in-for-hang.sh /triton-src
bash /patches/triton/apply-gfx803-vdot-gate.sh /triton-src

# Neither info file has to name a repository; triton's own CI fetches both pins
# from triton-lang/llvm-project.
llvm_hash="$(json_field /triton-src/cmake/llvm-info.json llvm_hash)"
llvm_repo="$(json_field /triton-src/cmake/llvm-info.json repository triton-lang/llvm-project)"
amd_llvm_hash="$(json_field /triton-src/cmake/amd-llvm-info.json llvm_hash)"
amd_llvm_repo="$(json_field /triton-src/cmake/amd-llvm-info.json repository triton-lang/llvm-project)"
amd_llvm_build="$(json_field /triton-src/cmake/amd-llvm-info.json build_number)"
echo "triton pins LLVM $llvm_repo @ $llvm_hash"
echo "triton pins its AMD code-generation LLVM $amd_llvm_repo @ $amd_llvm_hash (build $amd_llvm_build)"
/git-pin /llvm-project "https://github.com/${llvm_repo}.git" "$llvm_hash" "$llvm_hash"

jobs="$(resolve_build_jobs)"
echo "LLVM/MLIR build for triton: $jobs parallel jobs"

# Project set and target list mirror the "llvm" package of triton's own
# .github/workflows/llvm-build.yml -- that is what turns into the prebuilt LLVM
# triton's setup.py would otherwise download, so it is the known working
# recipe. The distribution list is upstream's minus not and split-file, which
# nothing in this wheel build uses. NVPTX is required even though this stack
# has no NVIDIA card: triton's CMakeLists links LLVMNVPTXCodeGen and the MLIR
# NVVM dialect into the core triton library unconditionally, not only into the
# (also always built, and here simply unused) nvidia backend.
#
# Two flags are NOT in that upstream recipe and are added here because our own
# hardware verification found them load-bearing, not because upstream's own
# recipe is wrong for upstream's own (downloaded, not self-built) LLVM binaries:
#   - LLVM_ENABLE_RTTI=ON: triton's own Python extension compiles without
#     -fno-rtti, so it needs typeinfo for LLVM/MLIR's class hierarchy, which a
#     default (RTTI-off) LLVM build does not export.
#   - LLVM_ABI_BREAKING_CHECKS=FORCE_OFF: LLVM_ENABLE_ASSERTIONS=ON otherwise
#     auto-enables ABI-breaking checks, emitting a build with
#     EnableABIBreakingChecks where triton's headers expect the Disable symbol.
# See patches/triton/README.md.
#
# GCC, not clang, builds this LLVM despite the ENV CC/CXX above and despite
# clang being the faster choice for LLVM's own Release build: Ubuntu clang
# 21.1.8 segfaults deterministically compiling
# mlir/lib/IR/BuiltinDialectBytecode.cpp (SideEffects aside, this is a
# generic clang bug, not a gfx803 patch touching LLVM/MLIR source, since no
# patch in this repo does). Confirmed not a stack-overflow: the crash
# reproduces identically with the process stack limit raised to 1GB, to 4GB,
# and to truly unlimited. Confirmed not a PCH problem either:
# CMAKE_DISABLE_PRECOMPILE_HEADERS=ON still crashes the same way. GCC compiles
# the same source without issue. ENV CC/CXX stay clang for triton's own wheel
# build below, which never touches this file.
cmake -S /llvm-project/llvm -B /llvm-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_DISTRIBUTION_COMPONENTS="llvm-headers;llvm-libraries;cmake-exports;mlir-headers;mlir-libraries;mlir-cmake-exports;lld-headers;lld-libraries;lld-cmake-exports;clang;clang-resource-headers;FileCheck;llc;opt;llvm-config;mlir-tblgen;mlir-translate" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX=/opt/triton-llvm
# check-mlir (LLVM/MLIR's own test suite, part of upstream's recipe) is skipped:
# these gfx803 patches touch only triton's third_party/amd sources, never
# LLVM/MLIR itself, so LLVM's own correctness is not this repo's thing to prove.
cmake --build /llvm-build --target install-distribution -j"$jobs"
rm -rf /llvm-build /llvm-project

if [ ! -x /opt/triton-llvm/bin/FileCheck ]; then
    echo "FATAL: /opt/triton-llvm has no FileCheck -- triton's own CMake copies this into the wheel and fails without it." >&2
    exit 1
fi

/git-pin /amd-llvm-project "https://github.com/${amd_llvm_repo}.git" "$amd_llvm_hash" "$amd_llvm_hash"
echo "LLVM build for triton's AMD code generation: $jobs parallel jobs"

# The recipe of the "amd-codegen" package in triton's llvm-build.yml: no extra
# projects, the AMDGPU target alone, and headers, static libraries and CMake
# exports as the whole distribution. The codegen library's own CMake refuses any
# LLVM library that is not static: it links LLVM into itself and hides it
# (--exclude-libs,ALL), so that this LLVM and the one in libtriton can live in
# one process without their symbols colliding.
#
# Neither extra flag of the build above applies here. The codegen library
# compiles against this LLVM's headers alone and picks -fno-rtti and
# -fno-exceptions to match how this LLVM was built.
#
# GCC builds this LLVM and the library on top of it, so that the archives and
# the code linking them come from the one compiler that has already built LLVM
# in this image.
#
# A ccache directory of its own: ccache caps each directory's size, and this
# build must not evict the main LLVM's objects from theirs.
amd_ccache=/root/.ccache-amd-codegen
env CCACHE_DIR="$amd_ccache" cmake -S /amd-llvm-project/llvm -B /amd-llvm-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_ENABLE_PROJECTS="" \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_UTILS=OFF \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_INSTALL_UTILS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_PIC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_DISTRIBUTION_COMPONENTS="llvm-headers;llvm-libraries;cmake-exports" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX=/opt/triton-amd-llvm
env CCACHE_DIR="$amd_ccache" cmake --build /amd-llvm-build --target install-distribution -j"$jobs"
rm -rf /amd-llvm-build /amd-llvm-project

# The revision string is the one triton's own build gives this library. The AMD
# backend folds it into every kernel cache key, so the format must match.
amd_llvm_revision="${amd_llvm_hash}-${amd_llvm_build}"
env CCACHE_DIR="$amd_ccache" cmake -S /triton-src/third_party/amd/backend/codegen -B /amd-codegen-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_DIR=/opt/triton-amd-llvm/lib/cmake/llvm \
    -DCMAKE_INSTALL_PREFIX=/opt/triton-amd-codegen \
    -DTRITON_AMD_LLVM_REVISION="$amd_llvm_revision"
env CCACHE_DIR="$amd_ccache" cmake --build /amd-codegen-build -j"$jobs"
cmake --install /amd-codegen-build
rm -rf /amd-codegen-build /opt/triton-amd-llvm

amd_codegen=/opt/triton-amd-codegen/lib/libtriton_amd_codegen.so
if [ ! -f "$amd_codegen" ]; then
    echo "FATAL: the AMD code-generation build produced no $amd_codegen." >&2
    exit 1
fi

echo "triton build: $jobs parallel jobs"
cd /triton-src

# Setup otherwise downloads a prebuilt libtriton_amd_codegen.so, built against
# an LLVM this repo never compiled. With TRITON_AMD_CODEGEN_PATH set, triton's
# build helper neither downloads nor copies one, so the wheel packages exactly
# the file placed here. The helper would have copied its own library to this
# same path, which MANIFEST.in's `graft third_party` puts into
# triton.backends.amd.
cp "$amd_codegen" third_party/amd/backend/lib/libtriton_amd_codegen.so
export TRITON_AMD_CODEGEN_PATH="$amd_codegen"

# An empty dependency cache, so that the check after the build can tell
# whether setup fetched or built anything of its own for the AMD backend: it
# keeps all of that under <cache>/amd.
export TRITON_HOME=/tmp/triton-home
rm -rf "$TRITON_HOME"
mkdir -p "$TRITON_HOME"

export LLVM_SYSPATH=/opt/triton-llvm
export LLVM_INCLUDE_DIRS=/opt/triton-llvm/include
export LLVM_LIBRARY_DIR=/opt/triton-llvm/lib
export MAX_JOBS="$jobs"
python3 -m build --wheel --no-isolation
mkdir -p /wheels
cp dist/triton*.whl /wheels/
rm -rf /opt/triton-llvm

if [ -e "$TRITON_HOME/.triton/amd" ]; then
    echo "FATAL: setup fetched or built its own AMD code-generation pieces under $TRITON_HOME/.triton/amd instead of using $amd_codegen." >&2
    ls -R "$TRITON_HOME/.triton/amd" >&2
    exit 1
fi
rm -rf "$TRITON_HOME"

echo "Verifying the wheel carries the gfx803 patches and the self-built AMD code generation..."
built_whl="$(ls /wheels/triton*.whl)"
check_dir=/tmp/triton-wheel-check
rm -rf "$check_dir"
mkdir -p "$check_dir"
python3 -m zipfile -e "$built_whl" "$check_dir"

libtriton="$(find "$check_dir" -path '*/triton/_C/libtriton*.so' | head -1)"
if [ -z "$libtriton" ]; then
    echo "FATAL: $built_whl has no triton/_C/libtriton*.so extension module." >&2
    exit 1
fi
wheel_codegen="$(find "$check_dir" -path '*/triton/backends/amd/lib/libtriton_amd_codegen.so' | head -1)"
if [ -z "$wheel_codegen" ]; then
    echo "FATAL: $built_whl has no triton/backends/amd/lib/libtriton_amd_codegen.so." >&2
    exit 1
fi

# All three patches compile into libtriton; none reaches the codegen library,
# whose only sources are amdgpu_codegen.cc and lib/Tools/LLVMOptions.cpp. Each
# marker is a name that exists only in the patched sources. It reaches the .so
# through the debug info of triton's default build type (-O2 -g) as a string of
# its own, so the match is on a whole line: a longer symbol that merely
# contains it, such as the isGCN3Encoding of the LLVM AMDGPU code linked into
# libtriton, must not count.
strings "$libtriton" > "$check_dir/libtriton.strings"
for marker in \
    "GCN3:gfx803-isa-family.patch" \
    "isOwnWhileConditionOperand:fold-true-cmpi-while-nested-in-for-hang.patch" \
    "supportsVDot:gfx803-vdot-gate.patch"; do
    name="${marker%%:*}"
    patch="${marker#*:}"
    if ! grep -qx "$name" "$check_dir/libtriton.strings"; then
        echo "FATAL: $libtriton carries no $name marker -- $patch did not make it into the built extension." >&2
        exit 1
    fi
done

# A byte-for-byte match is what proves that the wheel ships the library built
# above and not one that setup put in its place. The revision string shows
# which LLVM pin that library was built from.
if [ "$(sha256sum < "$wheel_codegen")" != "$(sha256sum < "$amd_codegen")" ]; then
    echo "FATAL: the wheel's libtriton_amd_codegen.so is not the one built from $amd_llvm_revision." >&2
    exit 1
fi
if ! strings "$wheel_codegen" | grep -qx "$amd_llvm_revision"; then
    echo "FATAL: the wheel's libtriton_amd_codegen.so does not carry revision $amd_llvm_revision." >&2
    exit 1
fi
rm -rf "$check_dir" /opt/triton-amd-codegen
echo "OK: triton wheel built with the gfx803 patches, a FileCheck-carrying LLVM and the self-built AMD code generation ($amd_llvm_revision)."
