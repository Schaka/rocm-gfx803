#!/bin/sh
# Build LLVM/MLIR for triton, apply the gfx803 patches, and build the triton wheel.
#
# Triton hard-pins an exact upstream LLVM commit (cmake/llvm-info.json in the
# triton checkout) rather than a branch, because its AMD/NVPTX lowering passes
# are written against one specific LLVM/MLIR API surface. Read that pin from
# the checkout instead of hand-copying it here, so a TRITON_REF bump always
# builds the LLVM its own source tree actually expects.
#
# Not cmake/llvm-build-info.json: that file is a second, separate LLVM pin
# used only by triton's own .github/workflows/llvm-build.yml to pre-build the
# *next* candidate LLVM ahead of a future triton release. It can be (and was,
# at TRITON_REF 675c598) a commit newer than what this exact triton source
# tree's dialect headers were written against. Building against it produced a
# real compile error, not a stale patch: mlir/Interfaces/SideEffectInterfaces.h
# lacked `Resource::getName() const` and `Resource::getParent()`, which
# triton/include/triton/Dialect/Triton/IR/Dialect.h already assumes. Confirmed
# by reading both pinned LLVM commits directly: llvm-info.json's hash has both
# methods, llvm-build-info.json's does not.
set -eu

TRITON_REF="${TRITON_REF:?TRITON_REF is required}"
. /scripts/lib/build-jobs.sh

/git-pin /triton-src https://github.com/triton-lang/triton.git "$TRITON_REF" "$TRITON_REF"
git -C /triton-src submodule sync --recursive
git -C /triton-src submodule update --init --recursive --depth 1

# bash, not sh: these three drivers are written against bash (${BASH_SOURCE[0]},
# [[ ]]), unlike every other patch driver in this repo.
bash /patches/triton/apply-gfx803-isa-family.sh /triton-src
bash /patches/triton/apply-gfx803-dpp-broadcast-warpreduce.sh /triton-src
bash /patches/triton/apply-fold-true-cmpi-while-nested-in-for-hang.sh /triton-src

llvm_hash="$(python3 -c "import json; print(json.load(open('/triton-src/cmake/llvm-info.json'))['llvm_hash'])")"
llvm_repo="$(python3 -c "import json; print(json.load(open('/triton-src/cmake/llvm-info.json')).get('repository', 'triton-lang/llvm-project'))")"
echo "triton pins LLVM $llvm_repo @ $llvm_hash"
/git-pin /llvm-project "https://github.com/${llvm_repo}.git" "$llvm_hash" "$llvm_hash"

jobs="$(resolve_build_jobs)"
echo "LLVM/MLIR build for triton: $jobs parallel jobs"

# Project set, target list, and distribution component list mirror triton's own
# .github/workflows/llvm-build.yml exactly -- that is what turns into the
# prebuilt LLVM triton's setup.py would otherwise download, so it is the known
# working recipe. NVPTX is required even though this stack has no NVIDIA card:
# triton's CMakeLists links LLVMNVPTXCodeGen and the MLIR NVVM dialect into the
# core triton library unconditionally, not only into the (also always built,
# and here simply unused) nvidia backend.
#
# Two flags are NOT in that upstream recipe and are added here because our own
# hardware verification found them load-bearing, not because upstream's own
# recipe is wrong for upstream's own (downloaded, not self-built) LLVM binaries:
#   - LLVM_ENABLE_RTTI=ON: triton's own pybind module compiles without
#     -fno-rtti, so it needs typeinfo for LLVM/MLIR's class hierarchy, which a
#     default (RTTI-off) LLVM build does not export.
#   - LLVM_ABI_BREAKING_CHECKS=FORCE_OFF: LLVM_ENABLE_ASSERTIONS=ON otherwise
#     auto-enables ABI-breaking checks, emitting a build with
#     EnableABIBreakingChecks where triton's headers expect the Disable symbol.
# Found by three rebuilds; see patches/triton/README.md.
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

echo "triton build: $jobs parallel jobs"
cd /triton-src
export LLVM_SYSPATH=/opt/triton-llvm
export LLVM_INCLUDE_DIRS=/opt/triton-llvm/include
export LLVM_LIBRARY_DIR=/opt/triton-llvm/lib
export MAX_JOBS="$jobs"
python3 -m build --wheel --no-isolation
mkdir -p /wheels
cp dist/triton*.whl /wheels/
rm -rf /opt/triton-llvm

echo "Verifying the wheel carries the gfx803 GCN3 ISA family..."
built_whl="$(ls /wheels/triton*.whl)"
if ! python3 -m zipfile -l "$built_whl" | grep -q 'libtriton'; then
    echo "FATAL: $built_whl has no libtriton extension module." >&2
    exit 1
fi
mkdir -p /tmp/triton-wheel-check
python3 -m zipfile -e "$built_whl" /tmp/triton-wheel-check
libtriton="$(find /tmp/triton-wheel-check -name 'libtriton*.so' | head -1)"
if [ -z "$libtriton" ]; then
    echo "FATAL: could not find libtriton*.so inside $built_whl." >&2
    exit 1
fi
# GCN3 is this patch set's own family name (see gfx803-isa-family.patch) and
# does not exist anywhere in unpatched triton, so its presence in the compiled
# extension is proof the patched TargetUtils.cpp, not a stale unpatched
# object, ended up in the linked .so.
if ! strings "$libtriton" | grep -q "GCN3"; then
    echo "FATAL: $libtriton carries no GCN3 marker -- the gfx803 ISA-family patch did not make it into the built extension." >&2
    exit 1
fi
rm -rf /tmp/triton-wheel-check
echo "OK: triton wheel built with the gfx803 patches and a FileCheck-carrying LLVM."
