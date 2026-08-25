# syntax=docker/dockerfile:1
# Polaris / gfx803 on ROCm 7.14 -- this repo's main line. See README.md for
# current status (what's verified vs. what's untested) and MIGRATION_NOTES.md
# for the full source-level evaluation this Dockerfile implements.
#
# Separate from rocm6.4.4/Dockerfile (the older, archived-but-still-buildable
# ROCm 6.4.4 line) on purpose -- ROCm 7's ROCR-Runtime rejects Polaris at HSA
# agent creation (legacy/pre-1.0 doorbell type), which 6.4.4 never had to work
# around. Getting past that wall needs a source-level ROCR-Runtime + CLR
# rebuild that rocm6.4.4/Dockerfile has no equivalent of.
#
# What has to be rebuilt from source, and why (see MIGRATION_NOTES.md for the
# full per-layer investigation):
#
#   ROCR-Runtime + CLR -- NEW for this line. ROCm 7's GpuAgent constructor
#                throws for any HSA agent whose DoorbellType != 2; Polaris
#                reports 0 or 1 ("legacy" doorbell). Restoring enumeration
#                AND real dispatch (not just enumeration) needs the full
#                legacy-doorbell code path an AMD engineer's from-source
#                TheRock build restored and verified on real Polaris
#                hardware -- see patches/rocm-systems/.
#   rocBLAS   -- same story as 6.4.4: gfx803 dropped from the default
#                TARGET_LIST at ROCm 6.0 but the Tensile logic
#                (Logic/asm_full/r9nano/*.yaml) is still in the tree, so
#                `rmake.py -a gfx803` builds it back. rocBLAS moved into the
#                ROCm/rocm-libraries monorepo since 6.4.4; this tracks its
#                release/therock-7.14 branch (see ROCM_LIBRARIES_REF below),
#                not a per-component release tag -- rocm-libraries stopped
#                cutting rocm-rel-X.Y tags/branches after 7.2.
#   MIGraphX  -- prebuilt for gfx900+ only, same as the main image. Still a
#                standalone repo (not folded into rocm-libraries); tracks its
#                release/rocm-rel-7.14 branch, same convention the main repo's
#                release.yml uses for MIGRAPHX_REF.
#   PyTorch   -- no gfx803 wheel has ever been published.
#   ORT       -- v1.28.0, MIGraphX EP only, matching the main (gfx900+)
#                build exactly -- NOT the 6.4.4 line's v1.22.2 ROCm-EP
#                pin. Tried keeping ROCm EP alive here too (the 6.4.4
#                line's reasoning: MIGraphX EP alone has no CK/MLIR to
#                fuse with on gfx803, so the ROCm EP fallback matters
#                more here than on gfx900+) -- v1.22.2's ROCm EP source
#                doesn't compile against ROCm 7.14's HIP headers at all
#                (a HIP API change unrelated to gfx803), and --use_rocm
#                is gone from ORT's own build flags as of 1.28 regardless.
#                Accepted cost: the CK/MLIR-fusion gap the 6.4.4 line's
#                ROCm EP fallback existed to soften is unsoftened here.
#                See MIGRATION_NOTES.md.
#   MIOpen    -- same story as 6.4.4: still lists gfx803 in
#                ALL_GPU_DATABASES and keeps its Ellesmere/Baffin/Polaris
#                device-name gating. Moved into rocm-libraries too.
#
# What's DIFFERENT from 6.4.4's known-good state, not yet re-verified on real
# hardware (see README.md "Status" for the current list):
#
#   - MIOpen's ConvOclDirectFwd/ConvOclDirectFwdFused solver (the one
#     rocm6.4.4/patches/miopen/conv-direct-fwd-grouped-oob.sh targets) was
#     REMOVED from MIOpen upstream between 6.4.4 and 7.14 -- replaced by a
#     new ConvHipDirectFwd (HIP-source, not OpenCL-source). Whether the
#     grouped-conv out-of-bounds read this patch fixed still exists in the
#     new solver is UNKNOWN; the fix cannot be safely ported without
#     re-running the original repro against the new solver on real gfx803
#     hardware first. NOT applied here. If the same symptom (OOB on grouped
#     convs) resurfaces during 7.14 validation, move the investigation over
#     to conv_hip_dir2Dfwd.cpp rather than assuming the old patch still
#     applies.
#   - Every other ported patch (rocBLAS's WGM and small-GEMM fixes, MIOpen's
#     Winograd-fused and Reduce-Prod fixes) applies cleanly to the pinned
#     7.14 source and has been read against the surrounding code to confirm
#     the fix still lands in the right place -- but NONE of them have been
#     re-run against real gfx803 hardware yet. Re-diffing restores
#     compilability, it does not substitute for re-running
#     KERNEL_BUGS.md's methodology against 7.14 binaries.
#
# What is switched off, because no version of it has ever supported gfx8
# (unchanged from 6.4.4 -- see rocm6.4.4/Dockerfile for the same reasoning):
#
#   rocMLIR, Composable Kernel, hipBLASLt.
#
# Build context is this directory, and everything this build needs lives under
# it -- patches/, including its own copy of the sgemm-shim, and tools/. That
# duplication against rocm6.4.4/ is deliberate, not an oversight: both lines
# are still actively changing, and this one started as a copy of 6.4.4 that
# can be worked on in isolation without any edit here being able to reach the
# hardware-verified 6.4.4 build. Fold the two back together once gfx803 on
# 7.14 is fully confirmed working, not before -- see README.md's
# "Convergence" section.
#
# Build: docker build -f Dockerfile -t <tag> .
ARG BASE_IMAGE=rocm/dev-ubuntu-26.04:7.14.0-full

ARG ROCR_CLR_IMAGE=rocr-clr-builder
ARG ROCBLAS_IMAGE=rocblas-builder
ARG MIOPEN_IMAGE=miopen-builder
ARG MIGRAPHX_IMAGE=migraphx-builder
ARG PYTORCH_IMAGE=pytorch-builder
ARG TORCHVISION_IMAGE=torchvision-builder
ARG TORCHAUDIO_IMAGE=torchaudio-builder
ARG ORT_IMAGE=ort-builder

# Branch pins, not commit SHAs -- same policy the main (gfx900+) repo's
# release.yml uses for MIGRAPHX_REF (release/rocm-rel-<major.minor>, "MIGraphX's
# own stable branch for that ROCm line, not develop"): track the named
# release's branch tip at build time, not a frozen point-in-time SHA. This
# build is manual-dispatch only (no schedule), so "branch tip at build time"
# means "whatever AMD has landed on release/therock-7.14 the day someone runs
# this," not a moving nightly target -- re-running the same workflow_dispatch
# twice can legitimately pick up new commits if AMD pushed a cherry-pick to
# the branch since the last run, which is expected and desired, not drift to
# chase down.
#
# rocm-libraries (rocBLAS, MIOpen) and rocm-systems (ROCR-Runtime, CLR)
# stopped cutting per-component rocm-rel-X.Y tags/branches after 7.2 (confirmed
# via `git ls-remote --heads`); ROCm/TheRock's release/therock-7.14 branch is
# the actual release line these two repos track for 7.14, confirmed by
# `git ls-remote --heads` against both landing on the exact same commits this
# Dockerfile pinned by SHA before this branch-based rewrite.
ARG ROCM_SYSTEMS_REF=release/therock-7.14
ARG ROCM_LIBRARIES_REF=release/therock-7.14

# MIGraphX stayed a standalone repo (not folded into rocm-libraries) and still
# cuts its own release/rocm-rel-<major.minor> branch -- same ref the main
# repo's release.yml derives MIGRAPHX_REF from.
ARG MIGRAPHX_REF=release/rocm-rel-7.14

# ROCm's PyTorch fork. release/2.13 matches what this repo's main
# (gfx900+) release track pins for ROCm 7.14 (see .github/workflows/
# release.yml's pytorch_version default, 2.13.0) -- kept in sync rather than
# picked independently, since "same defaults as the main part of this repo"
# was an explicit goal for this variant.
ARG PYTORCH_REF=release/2.13
# Taken directly from this repo's own source-build fallback logic
# (scripts/torch-package-build-decide.sh's determine_torchvision_repo_branch
# / determine_torchaudio_repo_branch), not extrapolated -- that script is
# the authoritative "what branch does a from-source companion build use
# for pytorch 2.13" answer already encoded in this repo, since gfx803 (no
# prebuilt wheel ever published, any ROCm line) always takes the SOURCE
# path those functions decide, never the PIP/prebuilt one.
ARG TORCHVISION_REF=release/0.28
ARG TORCHAUDIO_REF=release/2.11.0.2

# v1.28.0, matching the main (gfx900+) build's default -- NOT the 6.4.4
# line's v1.22.2 pin. That pin existed to keep ORT's ROCm EP (deleted
# upstream after v1.22.2, PR #25181) as a fallback for gfx803's lack of
# Composable Kernel/MLIR fusion in the MIGraphX EP. Tried keeping it here
# too, but v1.22.2's ROCm EP source doesn't compile against ROCm 7.14's
# HIP headers at all (`warpSize` changed from constexpr-usable to a
# non-constexpr accessor object -- a HIP API change, not gfx803-specific)
# -- confirmed via direct build failure. Patching multi-year-old,
# upstream-deleted EP code to work with a HIP version it was never built
# against isn't worth it for what's very likely not the only such
# incompatibility. --use_rocm/--rocm_home are also gone from build.py's
# own flag set as of 1.28 regardless (see docker/ort.Dockerfile's
# scripts/build/ort.sh). Accepted cost: MIGraphX-only on gfx803, same
# CK/MLIR-fusion gap the 6.4.4 line's ROCm EP fallback existed to soften,
# now unsoftened here. See MIGRATION_NOTES.md.
ARG ORT_VERSION=v1.28.0

ARG BUILD_PARALLEL_LEVEL=auto

# Ubuntu 26.04's native python3 is 3.14 (confirmed: `rocm/dev-ubuntu-26.04:
# 7.14.0-full` ships it), same situation the main Dockerfile's python-base
# documents -- numpy/onnx dependency resolution needs 3.12. uv-managed 3.12
# here, same approach as docker/python-base.Dockerfile, instead of rocm6.4.4/Dockerfile
# Dockerfile's "system python IS 3.12" assumption, which does not hold on
# this base image.
FROM ${BASE_IMAGE} AS python-base
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore
# This base image's cmake is 4.x, which hard-errors on any
# cmake_minimum_required() below 3.5 instead of just warning (older cmake
# behavior) -- and several transitively-fetched dependencies across this
# build (MIOpen's install_deps.cmake -> cget -> bzip2, seen first; likely
# not the last) bundle exactly that. Set globally as an env var, not a
# per-invocation -DCMAKE_POLICY_VERSION_MINIMUM=3.5 flag, since most of
# the places this bites (cget, cmake -P scripts) don't offer a way to pass
# extra flags through to the nested cmake calls they make themselves.
# CMake respects this env var identically to the command-line flag (3.31+).
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-dev python3-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12

# Layer 1 (EVALUATION_ROCM_7.md): restore Polaris HSA-agent creation and real
# queue dispatch (not just enumeration -- see patches/rocm-systems/
# hsa-agent-rejects-legacy-doorbell.patch for why removing the enumeration
# throw alone is not sufficient), plus CLR's own separate gfx8 rejection.
# Builds against the base image's OWN existing ROCm install (CMAKE_PREFIX_PATH
# =/opt/rocm), then installs back over the same /opt/rocm -- every later
# stage that chains from this one gets the patched runtime automatically via
# the normal /opt/rocm paths every other stage already hardcodes.
FROM python-base AS rocr-clr-builder

ARG ROCM_SYSTEMS_REF
ARG BUILD_PARALLEL_LEVEL

# CLR's README prerequisite (rocm-llvm-dev) is an apt package name from
# the classic apt-based ROCm install -- doesn't exist/apply on this
# TheRock-built base image, which has no system ROCm apt repo at all; the
# compiler CLR needs is already baked into /opt/rocm/llvm.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config \
        libnuma-dev libdrm-dev libelf-dev xxd \
        libgl1-mesa-dev libx11-dev mesa-common-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install CppHeaderParser

# Blobless partial clone + sparse-checkout: rocm-systems is a large monorepo
# (rccl, rocprofiler, hip-tests, etc.) and this build only needs three of its
# projects. --filter=blob:none defers file content until checkout touches it,
# so only the sparse set's blobs are ever fetched.
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/ROCm/rocm-systems.git /rocm-systems-src \
    && cd /rocm-systems-src \
    && git sparse-checkout init --cone \
    && git sparse-checkout set projects/rocr-runtime projects/clr projects/hip \
    && git checkout "${ROCM_SYSTEMS_REF}"

COPY patches/rocm-systems/ /rocm-systems-patches/
RUN sh /rocm-systems-patches/hsa-agent-rejects-legacy-doorbell.sh /rocm-systems-src
RUN sh /rocm-systems-patches/opencl-gfx8-hardcoded-rejection.sh /rocm-systems-src

# VA-reuse defer, re-diffed from the shipped 6.4.4 fix
# (rocm6.4.4/patches/rocr/va-reuse-defer.patch). Confirmed still needed on
# 7.14 and confirmed effective via the actual 52-shape reduce-harness
# sweep (rocm6.4.4/tools/reduce-harness/): unpatched ~10/52 fail every run
# (30/30 runs), patched 0/52 across 100/100 runs. See
# patches/rocm-systems/va-reuse-defer.patch's header for the full WHY and
# MIGRATION_NOTES.md ("va-reuse-defer: real signal from the actual
# 52-shape harness") for the validation data.
RUN sh /rocm-systems-patches/va-reuse-defer.sh /rocm-systems-src

# HIP graph-replay batch dispatch deadlock: gfx803's HW queue caps out at
# 64 packets (see graph-replay-batch-chunk-deadlock.patch's WHY -- root
# cause of the small queue itself still under investigation), far below
# DEBUG_HIP_GRAPH_BATCH_SIZE's default chunk size of 256, which made every
# HIP-graph replay of more than a few dozen packets deadlock permanently.
# Verified fix on real hardware: vLLM's graph-replay repro went from a
# deterministic hang to completing every run.
RUN sh /rocm-systems-patches/graph-replay-batch-chunk-deadlock.sh /rocm-systems-src

# ROCR-Runtime first: CLR's HIP build links against it, so the patched
# runtime has to be installed into /opt/rocm before CLR configures.
WORKDIR /rocm-systems-src/projects/rocr-runtime
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-rocr-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "ROCR-Runtime build: $jobs parallel jobs"; \
    mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
    && make -j"$jobs" \
    && make install \
    && cd .. && rm -rf build

# CLR: HIP only (CLR_BUILD_OCL=OFF) -- this build's whole stack (rocBLAS,
# MIOpen, MIGraphX, PyTorch, ORT) is HIP-based, not OpenCL. The CLR OpenCL
# patch is still applied above (harmless, and validated) in case a later
# revision of this Dockerfile turns CLR_BUILD_OCL on, but building the OCL
# runtime itself is skipped here as dead weight for this stack.
WORKDIR /rocm-systems-src/projects/clr
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-clr-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "CLR (HIP) build: $jobs parallel jobs"; \
    mkdir -p build && cd build \
    && cmake .. \
        -DHIP_COMMON_DIR=/rocm-systems-src/projects/hip \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLR_BUILD_HIP=ON \
        -DCLR_BUILD_OCL=OFF \
        -DHIP_PLATFORM=amd \
    && make -j"$jobs" \
    && make install \
    && cd .. && rm -rf build \
    # Verify the patched runtime actually landed: the doorbell throw's error
    # string should no longer be present in the installed HSA runtime lib,
    # and rocminfo/hipcc should still run cleanly (no GPU on this build box,
    # so this only proves the tools didn't break, not that a real card
    # enumerates -- see README.md for what still needs real
    # hardware).
    && echo "Verifying legacy-doorbell throw string is gone from the installed HSA runtime..." \
    && if strings /opt/rocm/lib/libhsa-runtime64.so* 2>/dev/null | grep -q "deprecated doorbell type"; then \
        echo "FATAL: installed libhsa-runtime64.so still contains the DoorbellType!=2 throw string." >&2; \
        exit 1; \
    fi \
    && echo "Verifying the active libamdhip64 is this build, not the stock one..." \
    && if ! strings "$(readlink -f /opt/rocm/lib/libamdhip64.so)" 2>/dev/null | grep -q "Image extension queries failed"; then \
        echo "FATAL: active libamdhip64.so does not carry the gfx8 opencl patch marker." >&2; \
        exit 1; \
    fi \
    && echo "OK: patched HSA runtime installed, no deprecated-doorbell throw string present, patched libamdhip64 active." \
    && /opt/rocm/bin/hipconfig --version

FROM ${ROCR_CLR_IMAGE} AS rocr-clr-export

FROM python-base AS rocblas-builder

ARG ROCM_LIBRARIES_REF
ARG ROCM_ARCH=gfx803
ARG BUILD_PARALLEL_LEVEL

COPY --from=rocr-clr-export /opt/rocm /opt/rocm

# libmsgpack-dev is a transitional dummy package on Ubuntu 26.04 (unlike
# 24.04, where rocm6.4.4/Dockerfile's identical line still works) -- the real
# C++ bindings Tensile's CMake looks for (msgpackc-cxx) are in
# libmsgpack-cxx-dev.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config gfortran ccache \
        libmsgpack-cxx-dev wget \
    && rm -rf /var/lib/apt/lists/*
# Debian ships the package's CMake config as msgpack-cxx-*.cmake, but Tensile
# looks for a msgpackc-cxx package -- same shim, same reason, as the main ROCm 7
# build's scripts/build/rocblas.sh.
RUN mkdir -p /usr/local/lib/cmake/msgpackc-cxx \
    && for f in /usr/lib/x86_64-linux-gnu/cmake/msgpack-cxx/msgpack-cxx-*.cmake; do \
        ln -sf "$f" "/usr/local/lib/cmake/msgpackc-cxx/$(basename "$f" | sed 's/^msgpack-cxx/msgpackc-cxx/')"; \
    done
RUN pip3 install pyyaml joblib

# Same blobless partial clone + sparse-checkout reasoning as rocr-clr-builder
# above -- rocm-libraries is a much larger monorepo (rocBLAS, MIOpen,
# hipBLAS, rocFFT, etc. all in one repo since the pre-7.0 restructure).
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/ROCm/rocm-libraries.git /rocblas-src-root \
    && cd /rocblas-src-root \
    && git sparse-checkout init --cone \
    # cmake + shared: the main ROCm 7 build's scripts/build/rocblas.sh checks
    # out `cmake shared/tensile projects/rocblas`; `shared` here is a superset
    # of shared/tensile, and the repo-root `cmake` directory is added to match
    # it -- rocBLAS's CMakeLists reaches up into it.
    && git sparse-checkout set cmake shared projects/rocblas \
    && git checkout "${ROCM_LIBRARIES_REF}"
ENV ROCBLAS_SRC=/rocblas-src-root/projects/rocblas

# WGM8 miscompute -- source-level fix (Tensile/SolutionStructs.py, gated on
# ISA==gfx803 && KernelLanguage==Assembly) rather than the sed-based
# wgm-miscompute.sh this replaces.
#
# A real rocblas-builder build of this Dockerfile initially looked like a
# regression vs. the sed: 518 "_WGM8" kernel-name occurrences remained in
# TensileLibrary_Type_*_fallback_gfx803.{hsaco,dat}. Investigated rather
# than assumed broken -- every one of those 518 kernel names carries
# ISA000_KLS (Tensile's HIP-*source* kernel-language marker), not
# ISA803_KLA (assembly) -- i.e. exactly the kernel class
# wgm-miscompute.sh's OWN header already measured as correct regardless of
# WGM value (2/2 correct, "the swizzle is emitted by the compiler rather
# than by Tensile"). Confirmed on real gfx803 hardware (RX 470,
# 192.168.1.214): rocblas_dgemm (TensileLibrary_Type_DD_*_fallback) and
# rocblas_zgemm (..._ZZ_*_fallback) both correct to ~1e-16 relative error
# across multiple shapes. Also confirmed zero "_WGM8" kernels remain
# outside the fallback libraries -- i.e. the actual assembly/KLA kernels
# this patch targets are clean. wgm-miscompute.sh kept in
# patches/rocblas/ for reference/rollback, not invoked here anymore.
COPY patches/rocblas/ /rocblas-patches/
RUN sh /rocblas-patches/wgm-miscompute-source.sh /rocblas-src-root

# Small-GEMM assembly miscompute -- re-diffed for 7.14 (indentation shift
# plus a real template-signature change, TiA/TiB collapsed to Ti; see the
# patch header for both). NOT yet re-verified on real hardware against 7.14.
RUN sh /rocblas-patches/small-gemm-assembly-miscompute.sh "${ROCBLAS_SRC}"

# fp16 (non-HPA) GEMM kernels for gfx803. Tensile's fp16 codegen used only
# d16/packed-fp16 instructions (GFX9+ only), so gfx803 had no fp16 kernels
# and every fp16 GEMM fell through to the slow _fallback_ kernel. The patch
# adds an unpacked codegen path (HasD16/halfNoD16) plus fixes a gfx803-only
# WGM-division register clobber that made the first working kernels silently
# miscompute. The HB logic file makes the Tensile library generation emit
# fp16 kernels for gfx803 (verified: generates, assembles, validates and
# benchmarks ~1.3-1.4x over the fallback on an RX 470 -- see the patch
# header and TENSILE_GFX803_FP16_TODO.md). HPA (fp32-accumulate) fp16 stays
# impossible on this hardware (no v_pk_fma_f16); this is non-HPA only.
RUN sh /rocblas-patches/tensile-gfx803-fp16-nond16.sh /rocblas-src-root \
    && cp /rocblas-patches/r9nano_Cijk_Ailk_Bljk_HB.yaml \
        /rocblas-src-root/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/r9nano/

WORKDIR ${ROCBLAS_SRC}
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-rocblas-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "rocBLAS build: arch ${ROCM_ARCH}, $jobs parallel jobs"; \
    python3 ./rmake.py -i -a "${ROCM_ARCH}" -j "$jobs" \
        --no_hipblaslt \
    && echo "Copying rocBLAS ${ROCM_ARCH} install output into /opt/rocm..." \
    # Copy-out and the three verification gates below are deliberately the
    # same mechanism, in the same order, with the same messages as the main
    # (gfx900+) ROCm 7 build's scripts/build/rocblas.sh -- that script is
    # already an arch-parameterised from-source rocBLAS builder for legacy
    # GCN arches, and gfx803 is just one more of them. Kept as a copy here
    # only because this whole directory is a self-contained fork for now
    # (see the header); written to converge so folding it back into that
    # script later is a diff, not a rewrite.
    #
    # Per-directory `readlink -f` + `cp -a`, not a plain `cp -a src/. dst/`:
    # this base image's /opt/rocm/{bin,lib,include,share} are themselves
    # symlinks (via /etc/alternatives/, to /opt/rocm/core-7.14/*) rather than
    # real directories, and cp refuses to merge a real directory over a
    # destination that lstat()s as a symlink ("cannot overwrite non-directory
    # X with directory Y") -- confirmed by direct build failure, which landed
    # zero gfx803 Tensile files in /opt/rocm. Resolving each destination to
    # its real directory first sidesteps that, and drops the rsync dependency
    # an earlier revision of this file used for the same purpose.
    && src="${ROCBLAS_SRC}/build/release/rocblas-install" \
    && for d in include lib share; do \
        [ -e "$src/$d" ] || continue; \
        real_dest="$(readlink -f "/opt/rocm/$d" 2>/dev/null || echo "/opt/rocm/$d")"; \
        mkdir -p "$real_dest"; \
        cp -a "$src/$d/." "$real_dest/"; \
    done \
    && find "$src" -mindepth 1 -maxdepth 1 ! -name include ! -name lib ! -name share \
        -exec cp -a {} /opt/rocm/ \; \
    # `! -type l` so the glob's sibling symlinks can't be picked instead of
    # the real ~100MB file -- the same trap miopen-builder below documents.
    && built_real="$(find "$src/lib" -maxdepth 1 -name 'librocblas.so.*' ! -type l | head -1)" \
    && built_size="$(stat -c%s "$built_real" 2>/dev/null || echo 0)" \
    && rm -rf "${ROCBLAS_SRC}/build" \
    && echo "Verifying ${ROCM_ARCH} Tensile library is present in /opt/rocm..." \
    && if ! find -L /opt/rocm -iname "*TensileLibrary*${ROCM_ARCH}*" | grep -q .; then \
        echo "FATAL: /opt/rocm has no ${ROCM_ARCH} Tensile library after the copy." >&2; \
        exit 1; \
    fi \
    && echo "OK: ${ROCM_ARCH} Tensile library confirmed present in /opt/rocm." \
    # Gate 2: a correct build that never gets loaded, because librocblas.so
    # still resolves to the stock gfx900+ file, is the exact failure the
    # 6.4.4 line lost days to. Size comparison rather than path, since the
    # copy above made the same content exist under two names.
    && echo "Verifying librocblas.so resolves to the ${ROCM_ARCH} build, not the stock one..." \
    && resolved="$(readlink -f /opt/rocm/lib/librocblas.so)" \
    && if [ "$built_size" = "0" ] || [ "$(stat -c%s "$resolved")" != "$built_size" ]; then \
        echo "FATAL: /opt/rocm/lib/librocblas.so resolves to ${resolved}, which is not our freshly-built rocBLAS (size mismatch)." >&2; \
        echo "The stock base-image rocBLAS is still what actually gets loaded at runtime." >&2; \
        exit 1; \
    fi \
    # Gate 3: the Tensile check only covers the Tensile .dat files, not
    # rocBLAS's own non-Tensile HIP kernels -- a link that succeeded with an
    # empty .hip_fatbin actually happened on the 6.4.4 line. Without a gate
    # it exits 0, gets pushed to the shared tag, and fails only much later at
    # runtime with "Illegal seek for GPU arch: gfx803".
    && echo "Verifying librocblas.so embeds real ${ROCM_ARCH} device code..." \
    && objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocblas_fatbin_check.bin \
    && fatbin_size="$(stat -c%s /tmp/rocblas_fatbin_check.bin)" \
    && rm -f /tmp/rocblas_fatbin_check.bin \
    && if [ "$fatbin_size" -lt 1000000 ]; then \
        echo "FATAL: librocblas.so's .hip_fatbin is only ${fatbin_size} bytes -- too small to contain real ${ROCM_ARCH} device code (expect several MB)." >&2; \
        exit 1; \
    fi \
    && echo "OK: librocblas.so resolves to a ${ROCM_ARCH} build with a ${fatbin_size}-byte .hip_fatbin."

# rocBLAS/Tensile's own SGEMM kernels are unreliable on gfx803 for every shape
# tested -- see patches/rocblas/sgemm-shim/gfx803_sgemm.h for the full
# investigation. LD_PRELOAD shim that routes the standard-algo f32
# rocblas_sgemm/rocblas_gemm_ex path to a correctness-verified replacement
# kernel; the ENV enabling it lives in the final stage. This line also
# intercepts rocblas_gemm_strided_batched_ex (small problems only) for the
# batched attention dots MIGraphX lowers there -- rocm6.4.4 does NOT have
# that interceptor yet. Kept as its own copy so this line can change without
# touching the 6.4.4 build (see the header); the strided-batched takeover
# fix is verified by the sb-takeover-no-algo-gate marker check below.
RUN mkdir -p /opt/rocm-sgemm-shim
COPY patches/rocblas/sgemm-shim/ /opt/rocm-sgemm-shim/
RUN hipcc -O2 -fPIC -shared --offload-arch=gfx803 -I/opt/rocm/include \
        /opt/rocm-sgemm-shim/sgemm_shim.cpp \
        -o /opt/rocm/lib/libgfx803_sgemm_shim.so \
        -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lrocblas -ldl \
    && if ! strings /opt/rocm/lib/libgfx803_sgemm_shim.so 2>/dev/null | grep -q "sb-takeover-no-algo-gate"; then \
        echo "FATAL: shim built without the strided-batched takeover fix (algo gate)." >&2; exit 1; \
    fi \
    && rm -rf /opt/rocm-sgemm-shim

FROM ${ROCBLAS_IMAGE} AS rocblas-export

FROM python-base AS miopen-builder

ARG ROCM_LIBRARIES_REF
ARG ROCM_ARCH=gfx803
ARG BUILD_PARALLEL_LEVEL

COPY --from=rocr-clr-export /opt/rocm /opt/rocm

# rocm-cmake is not an apt package here either (no system ROCm apt repo on
# this TheRock-built base -- same reasoning as rocr-clr-builder's
# rocm-llvm-dev fix). MIOpen's own install_deps.cmake/requirements.txt
# fetches rocm-cmake itself via cget, so it isn't needed as a system
# package the way it might be on a classic apt-based ROCm install. "half"
# was renamed to libhalf-dev on this Ubuntu release (confirmed via
# apt-cache search; "half" itself doesn't exist).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libhalf-dev libboost-system-dev libboost-filesystem-dev \
        libsqlite3-dev libbz2-dev lbzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none --no-checkout \
        https://github.com/ROCm/rocm-libraries.git /miopen-src-root \
    && cd /miopen-src-root \
    && git sparse-checkout init --cone \
    && git sparse-checkout set projects/miopen shared \
    && git checkout "${ROCM_LIBRARIES_REF}"
ENV MIOPEN_SRC=/miopen-src-root/projects/miopen

# NOT applied here: conv-direct-fwd-grouped-oob (6.4.4's grouped-conv OOB
# fix). Its target solver, ConvOclDirectFwd/ConvOclDirectFwdFused, was
# REMOVED from MIOpen upstream between 6.4.4 and 7.14 -- replaced by a new
# ConvHipDirectFwd (src/solver/conv/conv_hip_dir2Dfwd.cpp). Whether the same
# out-of-bounds weights-buffer read exists in the new solver is unknown;
# porting the fix without re-running the original repro against the new
# solver on real hardware would be guessing. See this directory's README.md
# and MIGRATION_NOTES.md for the current status of this gap.
COPY patches/miopen/ /miopen-patches/
RUN sh /miopen-patches/winograd-fused-conv-miscompute.sh "${MIOPEN_SRC}"
RUN sh /miopen-patches/reduce-prod-wrong-identity.sh "${MIOPEN_SRC}"

RUN grep -v "composable_kernel\|rocMLIR" "${MIOPEN_SRC}/requirements.txt" \
        > "${MIOPEN_SRC}/requirements-gfx803.txt" \
    && cp "${MIOPEN_SRC}/requirements-gfx803.txt" "${MIOPEN_SRC}/requirements.txt"

WORKDIR ${MIOPEN_SRC}
# Same cget/Python-3.14 incompatibility as migraphx-builder's rbuild step
# (cget subclasses urllib.request.FancyURLopener, removed in 3.13+):
# install_deps.cmake's own cget_exec() shells out to whatever "python3" it
# finds on PATH, with no override of its own -- put the uv-managed 3.12
# first so that resolves to a python cget actually works under.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-miopen-ccache \
    PATH="$(dirname "$(uv python find 3.12)"):$PATH" \
    cmake -P install_deps.cmake --minimum --prefix /miopen-deps

# amdclang++ resolves this base image's system libstdc++ (GCC 15) for
# standard headers, not a bundled libc++ -- GCC 15's <ciso646> self-
# deprecates via a #warning pragma ("not a standard header since C++20"),
# which clang's -W#warnings then turns into a hard error under MIOpen's
# own -Werror. Downgraded just that diagnostic class below rather than
# disabling -Werror broadly; confirmed via direct build failure, not a
# hypothetical.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-miopen-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "MIOpen build: arch ${ROCM_ARCH}, $jobs parallel jobs"; \
    mkdir -p build && cd build \
    && CXX=/opt/rocm/bin/amdclang++ cmake .. \
        -DCMAKE_PREFIX_PATH=/miopen-deps \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_TARGETS="${ROCM_ARCH}" \
        -DMIOPEN_BACKEND=HIP \
        -DMIOPEN_USE_COMPOSABLEKERNEL=Off \
        -DMIOPEN_USE_MLIR=Off \
        -DMIOPEN_USE_HIPBLASLT=Off \
        -DMIOPEN_BUILD_DRIVER=Off \
        -DBUILD_TESTING=Off \
        "-DCMAKE_CXX_FLAGS=-Wno-error=#warnings" \
    && make -j"$jobs" \
    && cp -a lib/libMIOpen.so* /tmp/ \
    && cd .. && rm -rf build

RUN echo "Copying MIOpen gfx803 build into /opt/rocm..." \
    && resolved="$(readlink -f /opt/rocm/lib/libMIOpen.so)" \
    && stock_size="$(stat -c%s "$resolved" 2>/dev/null || echo 0)" \
    # -type f: the glob also matches libMIOpen.so.1 (a symlink to
    # libMIOpen.so.1.0, ~16-byte target string, and neither name matches
    # "*.so" so the exclusion doesn't filter it out) -- without -type f,
    # `find`'s unordered directory listing can hand `head -1` the symlink
    # instead of the real ~100+MB file, landing a 16-byte "library" that
    # the old bare not-equal-to-stock-size check happily accepted.
    # Confirmed via direct build failure, not a hypothetical.
    && built="$(find /tmp -maxdepth 1 -iname 'libMIOpen.so.*' ! -iname '*.so' -type f | head -1)" \
    && cp -a "$built" "$resolved" \
    && rm -f /tmp/libMIOpen.so* \
    && new_size="$(stat -c%s "$resolved")" \
    && echo "libMIOpen resolved path: $resolved (stock ${stock_size} bytes -> new ${new_size} bytes)" \
    && if [ "$new_size" -lt 10000000 ]; then \
        echo "FATAL: libMIOpen is only ${new_size} bytes -- too small to be a real build (expect 100+MB)." >&2; \
        exit 1; \
    fi \
    && if [ "$new_size" = "$stock_size" ]; then \
        echo "FATAL: libMIOpen is still ${new_size} bytes, unchanged from stock -- our build did not land." >&2; \
        exit 1; \
    fi \
    && echo "OK: MIOpen gfx803 build is in place."

FROM ${MIOPEN_IMAGE} AS miopen-export

FROM python-base AS migraphx-builder

ARG ROCM_ARCH=gfx803
ARG MIGRAPHX_REF
ARG BUILD_PARALLEL_LEVEL

COPY --from=rocblas-export /opt/rocm /opt/rocm

# MIGraphX is configured with -DMIGRAPHX_USE_MIOPEN=On below and genuinely
# links against libMIOpen.so at build time -- not just a runtime dlopen --
# so it has to build against the gfx803-patched one from the start, same
# reasoning as rocblas-builder/miopen-builder building against rocr-clr-export
# instead of the base image's stock runtime. Only the .so, not the whole
# /opt/rocm: miopen-export chains from rocr-clr-export directly (not from
# rocblas-export), so copying its /opt/rocm wholesale here would silently
# revert the gfx803 rocBLAS build the line above just installed. Same glob as
# the final stage's own copy of this file, for the same base-image-SOVERSION
# reason.
COPY --from=miopen-export /opt/rocm/lib/libMIOpen.so.* /opt/rocm/lib/

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${MIGRAPHX_REF}" \
        https://github.com/ROCm/AMDMIGraphX.git /migraphx-src

# src/targets/gpu/lowering.cpp calls gpu::gfx_default_rocblas()
# unconditionally, but its only declaration/definition are gated behind
# #if MIGRAPHX_USE_HIPBLASLT -- a real upstream gap on rocm-7.14 that
# breaks the build for any target compiled with hipBLASLt off (required
# here: hipBLASLt has never had gfx8 kernels). Confirmed via direct build
# failure, not assumed; the call is dead code once hipBLASLt is disabled
# (hipblaslt_supported() itself always returns false then, so the `or
# gfx_default_rocblas()` short-circuits away) -- the fix only needs the
# symbol to exist, not to do anything reachable.
COPY patches/migraphx/ /migraphx-patches/
RUN sh /migraphx-patches/gfx-default-rocblas-hipblaslt-off-build-failure.sh /migraphx-src

# Same class of bug as the hipBLASLt patch above, different function: the
# MLIR-disabled build (-DMIGRAPHX_ENABLE_MLIR=Off, required -- no rocMLIR
# on gfx803) leaves is_module_fusible/dump_mlir_to_file/dump_mlir_to_mxr
# declared but undefined, while jit/mlir.cpp (always compiled) calls all
# three unconditionally. Unlike the hipBLASLt case this doesn't fail the
# build itself -- it links fine and only surfaces as `ImportError:
# undefined symbol` the first time anything dlopens libmigraphx_gpu.so
# (caught here via `import migraphx`, not a compile-time signal).
RUN sh /migraphx-patches/mlir-stub-missing-symbols.sh /migraphx-src

# The 6.4.4 line's parse-resize-fixes.patch (two ONNX-parser backports)
# is NOT carried over: fully obsolete against this MIGRAPHX_REF -- both
# fixes checked directly against the current source: fix 1
# (keep_aspect_ratio_policy="stretch") is already
# upstream verbatim, and fix 2's target function (calc_neighbor_points) no
# longer exists -- the whole linear-mode resize lowering it patched was
# replaced by a JIT-backed resize op upstream, exactly as the original
# patch's own header predicted might eventually happen. Nothing to port.

# --python 3.12: the base image's system python3 is 3.14 (Ubuntu 26.04),
# under which cget (rbuild's dependency manager) crashes at import time --
# it references six.moves.urllib.request.FancyURLopener, a class urllib
# dropped in 3.13+. Confirmed via direct build failure, not assumed. The
# uv-managed 3.12 from python-base sidesteps this entirely.
RUN uv venv /rbuild-venv --python 3.12 --seed \
    && /rbuild-venv/bin/pip install --no-cache-dir \
        https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

RUN sed -i '/composable_kernel/d; /rocMLIR/d' /migraphx-src/requirements.txt \
    && ! grep -q 'composable_kernel\|rocMLIR' /migraphx-src/requirements.txt

# Same MLIR-stub reasoning as rocm6.4.4/Dockerfile: rocMLIR is stripped from
# requirements.txt, but src/targets/gpu/mlir.cpp still #includes
# <mlir-c/Dialect/RockEnums.h> unconditionally -- vendor the one header
# instead of building all of rocMLIR/LLVM for it.
RUN mkdir -p /mlir-stub/mlir-c/Dialect \
    && cat > /mlir-stub/mlir-c/Dialect/RockEnums.h <<'EOF'
#ifndef MLIR_C_DIALECT_ROCK_ENUMS_H
#define MLIR_C_DIALECT_ROCK_ENUMS_H

#ifdef __cplusplus
extern "C" {
#endif

enum RocmlirTuningParamSetKind {
  RocmlirTuningParamSetKindQuick = 0,
  RocmlirTuningParamSetKindFull = 1,
  RocmlirTuningParamSetKindExhaustive = 2
};
typedef enum RocmlirTuningParamSetKind RocmlirTuningParamSetKind;

enum RocmlirSplitKSelectionLikelihood { never = 0, maybe = 1, always = 2 };
typedef enum RocmlirSplitKSelectionLikelihood RocmlirSplitKSelectionLikelihood;

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCK_ENUMS_H
EOF

# NOT re-verified against 7.14's actual src/targets/gpu/{mlir,jit/mlir}.cpp
# whether the same MIGRAPHX_MLIR-stub-function gap rocm6.4.4/Dockerfile's
# migraphx-builder stage documents (missing is_module_fusible/dump_mlir_to_*
# definitions in the #else branch) still exists on this MIGRAPHX_REF. If the
# build below fails at the python-import step with an undefined-symbol error
# for one of those three, apply the same sed fix rocm6.4.4/Dockerfile uses.

WORKDIR /migraphx-src
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-migraphx-ccache \
    ulimit -s unlimited && \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "MIGraphX build: using $jobs parallel jobs"; \
    # MIGraphX's own cmake/PythonModules.cmake ignores -DPython3_EXECUTABLE
    # for its python-module target: find_python(version) does a bare
    # `find_program(python<version>-config)` PATH search over a hardcoded
    # 3.6-3.14 list and silently skips any version whose -config script
    # isn't found. /rbuild-venv (a venv) never carries a python3.12-config
    # script itself -- that lives alongside the real interpreter uv
    # installed -- so without putting that directory on PATH,
    # find_python(3.12) fails silently while find_python(3.14) succeeds
    # against this base image's own python3.14-config, and the only
    # migraphx.so built ends up cpython-314-tagged: unimportable from the
    # 3.12 venv the final image actually uses. PYTHON_DISABLE_VERSIONS=3.14
    # makes this deterministic rather than relying on 3.12 winning a
    # discovery-order race. Confirmed as the actual mechanism by reading
    # this repo's own docker/migraphx.Dockerfile + scripts/build/migraphx.sh,
    # which hit and fixed the identical problem for the main (gfx900+) build. \
    py312_bin="$(dirname "$(uv python find 3.12)")"; \
    CMAKE_BUILD_PARALLEL_LEVEL=$jobs \
    PATH="${py312_bin}:/rbuild-venv/bin:$PATH" /rbuild-venv/bin/rbuild build -d /migraphx-deps -B build -G Ninja \
        --cxx=/opt/rocm/llvm/bin/clang++ --cc=/opt/rocm/llvm/bin/clang \
        "-DGPU_TARGETS=${ROCM_ARCH}" \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DMIGRAPHX_ENABLE_PYTHON=On \
        -DPython3_EXECUTABLE=/rbuild-venv/bin/python3 \
        -DPYTHON_DISABLE_VERSIONS=3.14 \
        -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off \
        -DMIGRAPHX_ENABLE_MLIR=Off \
        -DMIGRAPHX_USE_HIPBLASLT=Off \
        -DMIGRAPHX_USE_ROCBLAS=On \
        -DMIGRAPHX_USE_MIOPEN=On \
        -DBUILD_TESTING=Off \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_FLAGS=-I/mlir-stub \
        -DCMAKE_CXX_FLAGS=-I/mlir-stub \
        -T install \
    && rm -rf /migraphx-deps /mlir-stub

RUN echo "${MIGRAPHX_REF} $(git -C /migraphx-src rev-parse HEAD)" > /opt/rocm/migraphx-version.txt

# Same gate as the main (gfx900+) build's scripts/build/migraphx-verify-python.sh,
# and for the same reason: PYTHON_DISABLE_VERSIONS=3.14 above is what keeps
# find_python() from silently building the module against this base image's
# system 3.14, and a module with the wrong ABI tag is simply unimportable from
# the 3.12 venv the final image uses. Without this the failure surfaces two
# jobs later as an ImportError in the final stage, on a runner that has already
# spent hours -- catch it in the job that caused it.
RUN if ! find /opt/rocm -iname "migraphx.cpython-312-*.so" | grep -q .; then \
        echo "FATAL: no migraphx.cpython-312-*.so under /opt/rocm -- python module built for the wrong interpreter." >&2; \
        find /opt/rocm -iname "migraphx.cpython-*.so" >&2; \
        exit 1; \
    fi \
    && echo "OK: migraphx python module built for cpython-312."

FROM ${MIGRAPHX_IMAGE} AS migraphx-export

FROM python-base AS pytorch-builder

ARG ROCM_ARCH=gfx803
ARG PYTORCH_REF
ARG BUILD_PARALLEL_LEVEL
ARG TENSOR_TOPK_OPT_LEVEL=-O3

# migraphx-export, not rocblas-export: pytorch needs the gfx803-patched
# rocBLAS AND MIOpen, which only migraphx-export's /opt/rocm carries (same
# reasoning as ort-builder below).
COPY --from=migraphx-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libopenblas-dev libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

# --python 3.12: base image's system python3 is 3.14 (Ubuntu 26.04); the
# wheel this stage builds needs to match the 3.12 venv every other stage
# (and the final image) uses. --seed so pip is directly callable, matching
# docker/pytorch.Dockerfile's identical reasoning.
RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools \
    && /build-venv/bin/pip install --no-cache-dir numpy pyyaml typing_extensions requests six
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch "${PYTORCH_REF}" --depth 1 --shallow-submodules \
        https://github.com/ROCm/pytorch.git /pytorch

WORKDIR /pytorch
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 tools/amd_build/build_amd.py

# TensorTopK.hip's default -O3 AMDGPU backend compile has been measured
# taking 40GB+ combined RSS+swap and multiple hours regardless of
# available cores/RAM (both a 4-vCPU/16GB CI runner and a 24-core/30GB
# workstation hit the same wall). A lower TENSOR_TOPK_OPT_LEVEL for just
# this one file cuts both without touching every other kernel's codegen --
# confirmed at -O1, which cleared it in under a minute. Left at -O3 by
# default since this stays a real perf/build-time tradeoff for CI to opt
# into explicitly, not a source-level fact about the file. PyTorch's build
# system has no per-file flag override, so this wraps the compiler binary
# it invokes by absolute path instead -- a build-environment change, not a
# patch on PyTorch's own source.
RUN real=/opt/rocm/lib/llvm/bin/clang++.real \
    && mv /opt/rocm/lib/llvm/bin/clang++ "$real" \
    && printf '%s\n' \
        '#!/bin/sh' \
        "case \"\$*\" in" \
        "  *TensorTopK.hip*) exec $real \"\$@\" ${TENSOR_TOPK_OPT_LEVEL} ;;" \
        "  *) exec $real \"\$@\" ;;" \
        'esac' \
        > /opt/rocm/lib/llvm/bin/clang++ \
    && chmod +x /opt/rocm/lib/llvm/bin/clang++

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-pytorch-ccache \
    ulimit -s unlimited && \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "PyTorch build: using $jobs parallel jobs"; \
    env USE_ROCM=1 USE_CUDA=0 ROCM_HOME=/opt/rocm \
        "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        MAX_JOBS=$jobs USE_MKLDNN=0 USE_CCACHE=1 USE_NINJA=1 \
        USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
        USE_DISTRIBUTED=1 USE_ROCM_CK_GEMM=0 \
        BUILD_TEST=0 \
        python3 setup.py bdist_wheel

RUN pip install --no-cache-dir dist/torch*.whl \
    && mkdir -p /wheels && cp dist/torch*.whl /wheels/

FROM ${PYTORCH_IMAGE} AS pytorch-export

# Own image/CI job rather than folded into pytorch-builder: each of MIGraphX,
# PyTorch, torchvision and torchaudio gets its own runner and its own
# 360-minute budget in gfx803-rocm7.yml, same reasoning the mainline repo's
# docker/torchvision.Dockerfile split gives (one monolithic build overran a
# hosted runner's time budget). No prebuilt-wheel tier here, unlike the
# mainline Dockerfile -- gfx803 has never had one published for torch or its
# companions, so this always takes the from-source path.
FROM ${PYTORCH_IMAGE} AS torchvision-builder

ARG ROCM_ARCH=gfx803
ARG TORCHVISION_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev libpng-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --branch "${TORCHVISION_REF}" --depth 1 --shallow-submodules \
        https://github.com/pytorch/vision.git /vision
WORKDIR /vision
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-vision-ccache \
    env USE_ROCM=1 USE_CUDA=0 "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        FORCE_CUDA=0 \
        python3 setup.py bdist_wheel \
    && mkdir -p /wheels && cp dist/torchvision-*.whl /wheels/

FROM ${TORCHVISION_IMAGE} AS torchvision-export

# ROCm/audio, not upstream pytorch/audio -- matches
# scripts/torch-package-build-decide.sh's determine_torchaudio_repo_branch
# exactly (torchvision stays on upstream pytorch/vision; only audio has a
# ROCm-specific fork in this repo's own build logic).
FROM ${PYTORCH_IMAGE} AS torchaudio-builder

ARG ROCM_ARCH=gfx803
ARG TORCHAUDIO_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --branch "${TORCHAUDIO_REF}" --depth 1 --shallow-submodules \
        https://github.com/ROCm/audio.git /audio
WORKDIR /audio
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-audio-ccache \
    env USE_ROCM=1 USE_CUDA=0 "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        FORCE_CUDA=0 USE_FFMPEG=1 \
        python3 setup.py bdist_wheel \
    && mkdir -p /wheels && cp dist/torchaudio*.whl /wheels/

FROM ${TORCHAUDIO_IMAGE} AS torchaudio-export

FROM python-base AS ort-builder

ARG ROCM_ARCH=gfx803
ARG ORT_VERSION

COPY --from=migraphx-export /opt/rocm /opt/rocm

# libdrm-dev: onnxruntime_providers_rocm.cmake's find_package(rocm_smi)
# pkg_check_modules(libdrm) needs libdrm's .pc file, which only the -dev
# package ships (libdrm2, installed later in the final stage, is
# runtime-only and has no pkg-config data). Confirmed via direct build
# failure, not assumed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libprotobuf-dev protobuf-compiler libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

# Same 3.14-vs-3.12 reasoning as pytorch-builder above.
RUN uv venv /build-venv --python 3.12 --seed \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools \
    && /build-venv/bin/pip install --no-cache-dir numpy packaging cmake
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch "${ORT_VERSION}" --depth 1 \
        https://github.com/microsoft/onnxruntime.git /onnxruntime

# rocm6.4.4/Dockerfile's two ORT patches (mha-basic-mode-no-viable-op,
# topk-radix-tiebreak-nondeterministic) are NOT carried over here: both
# patch ROCm-EP-only source (contrib_ops/rocm/bert/... and
# core/providers/cuda/math/topk_impl.cuh, the latter shared/hipified into
# the ROCm EP build only when --use_rocm is passed). Since this build
# doesn't pass --use_rocm (see the ORT_VERSION comment above), neither
# file is compiled at all -- applying the patches would be dead weight,
# not a correctness fix. patches/onnxruntime/ is empty in this directory
# for that reason.

RUN eigen_commit=$(grep '^eigen;' /onnxruntime/cmake/deps.txt | cut -d';' -f2 | grep -oP '(?<=archive/)[0-9a-f]{40}') \
    && mkdir /eigen-src && cd /eigen-src \
    && git init -q \
    && git remote add origin https://gitlab.com/libeigen/eigen.git \
    && git fetch --depth 1 origin "$eigen_commit" \
    && git checkout -q FETCH_HEAD

WORKDIR /onnxruntime
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm7-ort-ccache \
    python3 tools/ci_build/build.py \
        --config Release \
        --build_dir /onnxruntime/build \
        --parallel \
        --build_wheel \
        --skip_tests \
        --allow_running_as_root \
        --compile_no_warning_as_error \
        --use_migraphx --migraphx_home /opt/rocm \
        --cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}" \
        --cmake_extra_defines "CMAKE_C_COMPILER_LAUNCHER=ccache" \
        --cmake_extra_defines "CMAKE_CXX_COMPILER_LAUNCHER=ccache" \
        --cmake_extra_defines "FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen-src" \
        --cmake_extra_defines "CMAKE_POLICY_VERSION_MINIMUM=3.5" \
        --cmake_extra_defines "onnxruntime_USE_COMPOSABLE_KERNEL=OFF"

# Tagging the wheel's version happens here, post-build, not by editing
# VERSION_NUMBER before the C++ build above: onnxruntime_c_api.cc has a
# compile-time static_assert comparing ORT_VERSION against a hardcoded
# literal, which a PEP 440 local segment in VERSION_NUMBER trips. wheel
# unpack/pack is the only safe way to add one -- PyPI's own onnxruntime can
# otherwise report the exact same version this build does, which would make
# an exact-version pin in the final image's constraints file (below)
# unenforceable (PyPI could satisfy it too); a suffix no PyPI release will
# ever carry closes that gap.
RUN built_whl="$(ls /onnxruntime/build/Release/dist/*.whl)" \
    && unpack_dir="/onnxruntime/build/Release/dist/unpacked" \
    && python3 -m wheel unpack "$built_whl" -d "$unpack_dir" \
    && old_dir="$(find "$unpack_dir" -mindepth 1 -maxdepth 1 -type d)" \
    && old_name="$(basename "$old_dir")" \
    && new_name="${old_name}+gfx803" \
    && mv "$old_dir" "$unpack_dir/$new_name" \
    && old_dist_info="$(find "$unpack_dir/$new_name" -maxdepth 1 -name '*.dist-info')" \
    && new_dist_info="$unpack_dir/$new_name/${new_name}.dist-info" \
    && mv "$old_dist_info" "$new_dist_info" \
    && sed -i "s/^Version: .*/Version: ${old_name#*-}+gfx803/" "$new_dist_info/METADATA" \
    && mkdir -p /onnxruntime/dist \
    && python3 -m wheel pack "$unpack_dir/$new_name" -d /onnxruntime/dist

FROM ${ORT_IMAGE} AS ort-export

FROM python-base

COPY --from=migraphx-export /opt/rocm /opt/rocm
# Only the MIOpen library files, not the whole /opt/rocm tree: miopen-builder
# chains from rocr-clr-export, not from rocblas-export/migraphx-export, so its
# own /opt/rocm never received the gfx803 rocBLAS or the from-source MIGraphX --
# copying it wholesale here would silently revert both back to stock. Same
# reasoning as rocm6.4.4/Dockerfile's final stage; the glob is looser only because
# this base image's MIOpen SOVERSION filename isn't fixed the way 6.4.4's
# libMIOpen.so.1.0.<suffix> was.
#
# migraphx-builder already copies this same file in before building itself
# (see that stage), so what's sitting in /opt/rocm after the migraphx-export
# copy above should already match -- this explicit copy is kept as the
# independently-controllable, authoritative source (final takes its own
# MIOPEN_IMAGE/with-miopen-image input), not as the only place this file
# reaches the final image.
COPY --from=miopen-export /opt/rocm/lib/libMIOpen.so.* /opt/rocm/lib/

# The wholesale /opt/rocm copies above come from pre-built component images;
# a stale cached export layer can silently re-point the libamdhip64.so.7
# symlink back at the base image's stock unpatched build (seen in practice:
# build succeeded, hipGetDeviceCount returned 0 with the patched lib sitting
# right next to the active one). Check the RESOLVED library, not the symlink
# target name, so this fails loudly if the active lib doesn't carry the gfx8
# opencl patch marker regardless of which layer re-pointed it.
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig \
    && if ! strings "$(readlink -f /opt/rocm/lib/libamdhip64.so)" 2>/dev/null | grep -q "Image extension queries failed"; then \
        echo "FATAL: active libamdhip64.so does not carry the gfx8 opencl patch marker." >&2; \
        exit 1; \
    fi

# libdrm2/libdrm-amdgpu1: every core ROCm lib built in this image
# (libhsa-runtime64, libamdhip64, librocblas, libMIOpen, migraphx's python
# module) links against the versioned libdrm.so.2/libdrm_amdgpu.so.1
# SONAMEs. This base image bundles its own *unversioned* copies under
# /opt/rocm/core-7.14/lib/rocm_sysdeps/lib (build-time-only, dev-symlink
# style -- confirmed present there without the .2/.1 suffix the dynamic
# linker actually looks for) -- confirmed via `ldd` reporting "not found"
# for these two on every one of the libs above, in a container that never
# had libdrm-dev/libdrm2 installed in its own OS layer (each build stage
# starts fresh FROM python-base; only /opt/rocm/* is copied across
# stages, not the OS package set the earlier stages needed at build time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev libopenblas0 ffmpeg libsndfile1 locales \
        libdrm2 libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Same Polaris runtime environment as rocm6.4.4/Dockerfile -- unchanged
# reasoning, see that file's comment block for HSA_OVERRIDE_GFX_VERSION /
# ROC_ENABLE_PRE_VEGA / TORCH_BLAS_PREFER_HIPBLASLT.
ENV HSA_OVERRIDE_GFX_VERSION=8.0.3
ENV ROC_ENABLE_PRE_VEGA=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=0

ENV LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim.so

# uv's own download cache would otherwise be committed into this image's
# layers; nothing here benefits from it surviving the build. Same as
# docker/final.Dockerfile.
ENV UV_NO_CACHE=1

ENV PYTHONPATH=/opt/rocm/lib
ENV VIRTUAL_ENV=/opt/venv
# --seed: matches docker/final.Dockerfile -- a uv venv ships no pip binary
# by default, but "$VIRTUAL_ENV/bin/pip" is invoked directly below.
RUN uv venv $VIRTUAL_ENV --python 3.12 --seed

COPY --from=ort-export /onnxruntime/dist/*.whl /tmp/ort/
COPY --from=pytorch-export /wheels/*.whl /tmp/torch/
COPY --from=torchvision-export /wheels/*.whl /tmp/torch/
COPY --from=torchaudio-export /wheels/*.whl /tmp/torch/
RUN "$VIRTUAL_ENV/bin/pip" install --no-cache-dir numpy /tmp/ort/*.whl /tmp/torch/*.whl \
    && rm -rf /tmp/ort /tmp/torch \
    && "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torchvision; print('torchvision', torchvision.__version__)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torchaudio; print('torchaudio', torchaudio.__version__)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import migraphx; print('migraphx python module OK')"

# This image is a drop-in base for downstream Dockerfiles -- any of them can
# `pip install`/`uv pip install` something that pulls in torch, torchvision
# or torchaudio as a transitive dependency, and pip will silently swap this
# image's ROCm/gfx803 build for a generic PyPI wheel with neither -- no
# error, no warning. Pinning the exact already-installed version turns that
# from a silent swap into a loud resolution failure: pip/uv can't satisfy
# the pin from any index, only by reusing what's already installed. Both
# env vars are set because pip reads PIP_CONSTRAINT and uv reads
# UV_CONSTRAINT -- neither honors the other's.
#
# importlib.metadata, not `pip freeze | grep`: torch/torchvision/torchaudio
# and the ORT wheel here were all installed from a local file path
# (`pip install ./dist/*.whl`), which pip freeze renders as
# `name @ file:///...` rather than `name==version` -- the `==`-anchored
# grep this used to be never matches that format regardless of install
# order. importlib.metadata.version() reads the installed distribution's
# actual version regardless of how it got there.
#
# The distribution here is onnxruntime-migraphx, not onnxruntime (its
# import name is onnxruntime, but that's a different thing from its pip
# package name) -- pinning it protects against a transitive
# onnxruntime-migraphx pull, not a transitive plain onnxruntime one. A
# downstream Dockerfile that pulls in generic `onnxruntime` (e.g.
# faster-whisper via ctranslate2) is not blocked by this constraint file;
# closing that gap needs the wheel itself renamed or a real pip
# override, not a constraints-file entry under a name nothing is
# installed as.
RUN "$VIRTUAL_ENV/bin/python3" -c "import importlib.metadata as m; [print(f'{p}=={m.version(p)}') for p in ('torch', 'torchvision', 'torchaudio', 'onnxruntime-migraphx')]" > /opt/pip-constraints.txt \
    && cat /opt/pip-constraints.txt
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt
ENV UV_CONSTRAINT=/opt/pip-constraints.txt

ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
