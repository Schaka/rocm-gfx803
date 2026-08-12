# syntax=docker/dockerfile:1
# EXPERIMENTAL. See README.md in this directory before reading further --
# this line has NOT been run on real gfx803 hardware, is not a replacement
# for the root Dockerfile (rocm7) or rocm6.4.4/Dockerfile, and its patches
# are vendored copies, re-verified to apply here but not yet re-verified for
# hardware correctness under this specific build.
#
# Purpose: build gfx803 through TheRock's OWN build orchestration
# (fetch_sources.py --patch-tag + cmake -DTHEROCK_AMDGPU_FAMILIES + ninja)
# instead of this repo's hand-rolled per-component clone/patch/cmake
# Dockerfile stages (see ../Dockerfile). Answers a different question than
# the root Dockerfile does: not "does gfx803 work with these patches" (the
# root line already answers that, on real hardware) but "do these same
# patches survive being built the way AMD's own tooling would build them."
# See ../ROCM_UPSTREAM_ANALYSIS.md for why this question matters for an
# eventual upstream ask.
#
# therock-builder replaces the root Dockerfile's rocr-clr-builder +
# rocblas-builder + miopen-builder with a single stage: TheRock's cmake
# super-project builds ROCR-Runtime/CLR, rocBLAS and MIOpen together for
# whichever -DTHEROCK_AMDGPU_FAMILIES target is requested, so there is no
# equivalent of building them as three separate, individually-cacheable
# images the way this repo's own orchestration does. Every stage after
# therock-builder (migraphx/pytorch/torchvision/torchaudio/ort/final) is
# unchanged in substance from the root Dockerfile -- TheRock doesn't build
# any of those either, upstream or here, so this repo's existing
# clone-and-patch approach for them is the correct approach here too, just
# repointed at therock-builder's /opt/rocm instead of the root Dockerfile's
# rocr-clr+rocblas+miopen stack.
#
# Build: docker build -f therock-experimental/Dockerfile -t <tag> therock-experimental/
ARG THEROCK_BASE_IMAGE=ubuntu:24.04
ARG PYTHON_BASE_IMAGE=ubuntu:24.04

ARG THEROCK_IMAGE=therock-builder
ARG MIGRAPHX_IMAGE=migraphx-builder
ARG PYTORCH_IMAGE=pytorch-builder
ARG TORCHVISION_IMAGE=torchvision-builder
ARG TORCHAUDIO_IMAGE=torchaudio-builder
ARG ORT_IMAGE=ort-builder

# TheRock has its own release/therock-7.14 branch (confirmed via
# `git ls-remote --heads`) -- same branch-pin convention as every other ref
# in this repo, not main/develop. It does not carry a gfx803 AMDGPU target
# (our own cmake patch, see therock-cmake-patch/, adds that), but its
# rocm-systems/rocm-libraries submodule pins are what matter for patch
# compatibility, and those DO already line up with this repo's own
# independently-resolved pins: TheRock's release/therock-7.14
# (418cd5f63abb7a604bad5874cd7b2e29334e640f, the tip at the time this was
# written) pins rocm-libraries @ cd9574023093742434e8c992d13b89ab9a6c1cf8
# and rocm-systems @ 2b22ab0195cc1461cd9abf3b969e9dd7c10af350 -- the exact
# commits ../patches/ was already verified against, confirmed by directly
# reading the pinned tree rather than assumed. Pinning THEROCK_REF here to
# main's tip instead (an earlier version of this file did exactly that)
# was a real mistake: it resolves those two submodules to different
# commits entirely, and two of the vendored patches needed a real re-port
# as a direct result -- see therock-experimental/README.md's
# "Re-porting across pins" section for what broke and how it was found.
# Pinned to an exact commit ON that branch, not the branch itself, for the
# same reproducibility reason this repo pins release branches to whatever
# commit a given run actually used rather than letting later reruns
# silently drift.
ARG THEROCK_REF=418cd5f63abb7a604bad5874cd7b2e29334e640f

# Same defaults as the root Dockerfile, kept in sync deliberately.
ARG MIGRAPHX_REF=release/rocm-rel-7.14
ARG PYTORCH_REF=release/2.13
ARG TORCHVISION_REF=release/0.28
ARG TORCHAUDIO_REF=release/2.11.0.2
ARG ORT_VERSION=v1.28.0

ARG BUILD_PARALLEL_LEVEL=auto

# TheRock's own README documents this exact dependency set for Ubuntu
# 24.04 (the newest release TheRock's build docs currently cover), plus its
# own patched-patchelf and Rust-1.95 install steps -- both required by
# fetch_sources.py/the cmake configure step regardless of which GPU target
# is being built, not gfx803-specific.
FROM ${THEROCK_BASE_IMAGE} AS therock-builder
ENV DEBIAN_FRONTEND=noninteractive
ARG THEROCK_REF
ARG BUILD_PARALLEL_LEVEL

# libsqlite3-dev: not in TheRock's own documented Ubuntu 24.04 prereq list
# (README.md's `apt install` lines) -- added after rocprofiler-sdk failed
# with `fatal error: 'sqlite3.h' file not found` despite its own cmake
# configure step successfully resolving TheRock's bundled therock-sqlite3
# sysdep package (built+staged early, long before rocprofiler-sdk
# configures). The resolved package just never reaches the specific
# translation units that #include it directly -- a rocprofiler-sdk-internal
# CMake target/include-path gap, not TheRock's or gfx803's. Tried disabling
# rocprofiler-sdk instead (-DTHEROCK_ENABLE_ROCPROFV3=OFF, then also
# =PROFILER/=ROCPROFILER_COMPUTE to close a requirement-propagation
# cascade); TheRock's own cmake still logged "Implicitly enabled features:
# ROCPROFV3 ..." and built it anyway regardless of all three flags -- a
# system header satisfies the plain #include unconditionally, sidestepping
# that whole feature-flag mechanism instead of excavating it further.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
        gfortran ninja-build cmake gcc g++ pkg-config xxd automake libtool \
        python3-venv python3-dev libegl1-mesa-dev texinfo bison flex \
        make patch libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/ROCm/TheRock.git /therock \
    && cd /therock && git checkout "${THEROCK_REF}"

# See therock-experimental/README.md for why this is a patched build,
# rather than a stock patchelf.
RUN cd /therock \
    && env INSTALL_PREFIX=/usr/local ./dockerfiles/install_pinned_patchelf.sh

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- --default-toolchain 1.95.0 -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /therock
RUN python3 -m venv /therock-venv \
    && /therock-venv/bin/pip install --upgrade pip \
    && /therock-venv/bin/pip install -r requirements.txt

# THEROCK_BUNDLE_SYSDEPS=ON (this build's default) needs meson on PATH --
# cmake shells out to it directly. No system python3-pip on this base
# image (only python3-venv), so installed into /therock-venv and put on
# PATH rather than reaching for a system pip3 that isn't there. Confirmed
# via direct build failure ("Building with THEROCK_BUNDLE_SYSDEPS=ON on
# Linux requires... pip install meson"), not assumed.
#
# Pinned to TheRock's requirements.txt floor (meson>=1.7.0), not left to
# float to whatever's latest on PyPI: an unpinned install pulled 1.12.0,
# which re-applies the sysdeps stages' cmake-injected LDFLAGS (containing
# --version-script=.../version.lds) a second time on `meson setup
# --reconfigure`, so libdrm's sanity-check link fails with "ld: duplicate
# version tag" -- confirmed by dumping libdrm's meson-log.txt, not
# guessed. Floor version doesn't have that reconfigure-LDFLAGS-duplication
# behavior.
RUN /therock-venv/bin/pip install 'meson==1.7.0'
ENV PATH="/therock-venv/bin:${PATH}"

# Adds the gfx803 AMDGPU target -- this line does not exist on any TheRock
# branch upstream (see THEROCK_REF's own comment above). Applied directly
# to the TheRock checkout, NOT via fetch_sources.py's --patch-tag mechanism:
# that mechanism only patches git submodules (rocm-systems, rocm-libraries,
# ...), never TheRock's own top-level repo, and this cmake file lives in
# TheRock's own tree.
COPY therock-cmake-patch/ /therock-cmake-patch/
RUN git apply --verbose /therock-cmake-patch/0001-cmake-add-gfx803-target.patch \
    && grep -q 'gfx803-dgpu' cmake/therock_amdgpu_targets.cmake \
    || { echo "FATAL: gfx803 target line missing after cmake patch apply" >&2; exit 1; }

# TheRock's own patch-tag mechanism: fetch_sources.py inits every submodule
# this target needs, then `git am`s patches/gfx803/<project>/*.patch into
# each one via the project name keying patches/<tag>/<project>/. These are
# vendored, re-verified `git am --whitespace=nowarn` clean against the
# exact pinned rocm-systems/rocm-libraries commits at the time this was
# written -- see README.md for what's carried here vs. the root Dockerfile
# (wgm-miscompute is NOT carried; it's a sed-script rewrite in the root
# line, not a diff, and has no git-am-able form yet).
COPY therock-patches/gfx803/ patches/gfx803/
RUN /therock-venv/bin/python3 build_tools/fetch_sources.py --patch-tag gfx803

# Full stack (rocBLAS, MIOpen, RCCL, etc. -- everything THEROCK_ENABLE_ALL's
# default pulls in for this target), not just the HIP/OpenCL runtime layer
# lucbruni-amd's own validation covered -- this pipeline's whole point is
# testing whether the correctness patches (rocm-libraries/) survive
# TheRock's own build, not just the enablement ones (rocm-systems/).
# THEROCK_ENABLE_ROCJITSU=OFF: rocjitsu (emulation/rocjitsu, a GPU ISA
# emulator, not part of this pipeline's scope of core libs + MIGraphX +
# PyTorch + ORT) is gated by this dedicated cmake option, not by
# EXCLUDE_TARGET_PROJECTS in the gfx803 cmake patch above -- that list
# only filters which GPU targets a project builds FOR, not whether the
# project itself is included, confirmed by reading
# emulation/CMakeLists.txt's `if(THEROCK_ENABLE_ROCJITSU)` guard after an
# EXCLUDE_TARGET_PROJECTS entry for it had no effect. Needed here because
# rocjitsu's own CMakeLists doesn't declare a build dependency on
# therock-libdrm, so under ninja's parallelism it can start (and fail on
# the missing libdrm/drm.h header) before libdrm finishes building --
# confirmed via a real build log (rocjitsu started at 97s, therock-libdrm
# didn't finish until ~135s). Not gfx803-specific (the missing dependency
# edge exists for every target), just newly exercised by this from-scratch
# full-stack build.
# /therock/build is cache-mounted (not a plain layer) starting here, and by
# every later RUN that touches it: a from-scratch build here has repeatedly
# taken 1-2+ hours and repeatedly died mid-build (OOM, upstream build-graph
# bugs) -- without this, EVERY retry re-cmake-configures and re-ninjas from
# absolute zero, since a cache mount's content lives in BuildKit's cache
# store, not in the image layer a failed RUN discards. All three RUN steps
# that read or write /therock/build (this one, the ninja build, and the
# later `cp .../dist/rocm` promotion step) share the same mount id so they
# see one persistent directory across docker build invocations, giving
# ninja's own incremental/.ninja_log state real continuity -- a bare retry
# without this mount looks identical on the outside but silently starts
# over from an empty build dir every time.
#
# THEROCK_BUILD_TESTING=OFF: hip-tests alone (part of the default `all`
# target) took 8874 seconds (~2.5h) in a from-scratch build here -- not
# needed for this pipeline's actual goal (a working /opt/rocm to build
# MIGraphX/PyTorch/ORT against, not HIP's own test suite).
#
# rocprofiler-sdk's `fatal error: 'sqlite3.h' file not found` (see the
# libsqlite3-dev comment in the apt-get step above for the full trace) is
# fixed at the header level, not by disabling the subproject: tried
# -DTHEROCK_ENABLE_ROCPROFV3=OFF, then also =PROFILER/=ROCPROFILER_COMPUTE
# to close a requirement-propagation cascade (rocprofiler-compute's
# SUBPROJECT_DEPS on rocprofiler-sdk force-re-enables it via
# cmake/therock_features.cmake), but TheRock's own cmake still logged
# "Implicitly enabled features: ROCPROFV3 ..." and built it regardless of
# all three flags -- some other default-enabled component depends on it
# too. Not worth excavating further when a system header sidesteps the
# whole mechanism.
RUN --mount=type=cache,target=/therock/build,id=gfx803-therock-experimental-builddir \
    cmake -B build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DTHEROCK_AMDGPU_FAMILIES=gfx803-dgpu \
        -DTHEROCK_ENABLE_ROCJITSU=OFF \
        -DTHEROCK_BUILD_TESTING=OFF

# -j10 on this outer ninja does NOT cap subproject compiles: every TheRock
# subproject build step runs as `cmake --build <dir>` with no --parallel
# flag (cmake/therock_subproject.cmake's add_custom_command for
# ${target_name}+build), which on the Ninja generator falls through to
# ninja's own nproc-wide default inside that nested invocation, invisible
# to the outer job pool. Confirmed by dumping journalctl -k at the second
# OOM: 25 concurrent cc1plus processes (up to 212MB anon-rss each) from a
# single amd-llvm subbuild alone, well past what -j10 should have allowed.
# CMAKE_BUILD_PARALLEL_LEVEL is what `cmake --build` actually reads when
# --parallel isn't passed explicitly, so it's the only lever that reaches
# the nested builds; outer -j1 serializes at the top level too so at most
# one subproject (and its one-job-wide nested build) runs at a time.
#
# -j1 confirmed the OOM was actually gone (host memory held flat over a
# 44-minute run) but was too slow to be usable even locally -- amd-llvm
# alone projected to multiple hours at ~1.6s/object single-threaded.
#
# Raising the OUTER ninja -j alongside CMAKE_BUILD_PARALLEL_LEVEL is wrong:
# outer -jN also lets up to N subprojects build concurrently, and each one
# starts its OWN nested ninja at CMAKE_BUILD_PARALLEL_LEVEL -- multiplying,
# not capping. Confirmed via ps at the next OOM: three independent
# `ninja -j 4` processes running simultaneously (amd-llvm + host-blas'
# COOPERLAKE/SKYLAKEX openblas kernel builds), i.e. up to 4x4=16 concurrent
# heavy compiles from a Dockerfile line that only says "-j4" twice.
#
# Real 6-wide (outer -j1 + CMAKE_BUILD_PARALLEL_LEVEL=6) never showed more
# than 6 concurrent cc1plus at once, but still OOM'd repeatedly: this host's
# 8GB zram swap crept from empty to full over 15-35 minutes of sustained
# amd-llvm compilation regardless of instantaneous job count, then any burst
# triggered the kill. Not a job-count multiplication bug this time -- just
# more sustained memory pressure than 30GB RAM + 8GB zram supports at 6-wide.
#
# outer ninja stays -j1 always (see the multiplication warning above --
# raising it here would multiply against CMAKE_BUILD_PARALLEL_LEVEL again).
# CMAKE_BUILD_PARALLEL_LEVEL itself is sized the same MemAvailable/4GB way
# every other stage in this Dockerfile does (scripts/lib/build-jobs.sh in
# the sibling rocm-migraphx-ort-builder repo is the source of this formula
# -- ~4GB RSS per -O3 clang job doing IPO/codegen on LLVM's own sources was
# the actual measured figure there, and amd-llvm here is the same kind of
# job), rather than the hardcoded 2 this line used while debugging the OOM
# chain locally -- a fixed 2 is both unusable in CI (which has more RAM
# than this box had free during that debugging) and unsafe on a smaller
# runner than this box. BUILD_PARALLEL_LEVEL=auto is the default; pass an
# explicit --build-arg BUILD_PARALLEL_LEVEL=N to override on a host whose
# RAM is known and dedicated.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-therock-experimental-ccache \
    --mount=type=cache,target=/therock/build,id=gfx803-therock-experimental-builddir \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    CMAKE_BUILD_PARALLEL_LEVEL=$jobs ninja -C build -j1

# TheRock's own output layout (build/dist/rocm) matches the /opt/rocm shape
# every downstream stage in this Dockerfile (and the root Dockerfile it was
# copied from) already assumes -- CMAKE_PREFIX_PATH=/opt/rocm, llvm at
# /opt/rocm/llvm, etc. -- so re-rooting it here is what lets those stages be
# reused with no further changes beyond which stage they COPY --from.
#
# Needs the same /therock/build cache mount as the ninja step above: that
# directory's real content (including dist/rocm, ninja's actual output)
# only exists in the cache store, not in the image layer this RUN would
# otherwise see -- without the mount here, this step would see whatever
# empty/pre-ninja state the plain image layer had instead.
RUN --mount=type=cache,target=/therock/build,id=gfx803-therock-experimental-builddir \
    mkdir -p /opt/rocm && cp -a /therock/build/dist/rocm/. /opt/rocm/ \
    && test -x /opt/rocm/bin/hipcc \
    || { echo "FATAL: /opt/rocm/bin/hipcc missing after TheRock build -- dist layout assumption broken" >&2; exit 1; }

FROM ${THEROCK_IMAGE} AS therock-export

# Everything from here down is the root Dockerfile's migraphx-builder /
# pytorch-builder / torchvision-builder / torchaudio-builder / ort-builder /
# final stages, unchanged in substance -- only the base stage they COPY
# /opt/rocm from is different (therock-export instead of
# rocblas-export/miopen-export/rocr-clr-export). See ../Dockerfile for the
# full WHY behind each of these stages; not re-derived here to avoid the
# two copies drifting apart in explanation while they're still identical in
# mechanism.
FROM ${PYTHON_BASE_IMAGE} AS python-base
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-dev python3-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12

FROM python-base AS migraphx-builder

ARG ROCM_ARCH=gfx803
ARG MIGRAPHX_REF
ARG BUILD_PARALLEL_LEVEL

COPY --from=therock-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${MIGRAPHX_REF}" \
        https://github.com/ROCm/AMDMIGraphX.git /migraphx-src

# Same two upstream gaps the root Dockerfile's migraphx-builder documents
# in full (gfx_default_rocblas() declared-but-undefined with hipBLASLt off;
# MLIR-disabled stub-function ImportError) -- MIGraphX is not part of
# TheRock's submodule tree, so it keeps this repo's existing clone-and-patch
# approach unchanged, just built against therock-export's /opt/rocm.
COPY patches-migraphx/ /migraphx-patches/
RUN sh /migraphx-patches/gfx-default-rocblas-hipblaslt-off-build-failure.sh /migraphx-src
RUN sh /migraphx-patches/mlir-stub-missing-symbols.sh /migraphx-src

RUN uv venv /rbuild-venv --python 3.12 --seed \
    && /rbuild-venv/bin/pip install --no-cache-dir \
        https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

RUN sed -i '/composable_kernel/d; /rocMLIR/d' /migraphx-src/requirements.txt \
    && ! grep -q 'composable_kernel\|rocMLIR' /migraphx-src/requirements.txt

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

WORKDIR /migraphx-src
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-therock-experimental-migraphx-ccache \
    ulimit -s unlimited && \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
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

COPY --from=migraphx-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /pytorch-venv --python 3.12 --seed \
    && /pytorch-venv/bin/pip install --no-cache-dir pyyaml numpy typing_extensions

RUN git clone --recursive --branch "${PYTORCH_REF}" --depth 1 --shallow-submodules \
        https://github.com/ROCm/pytorch.git /pytorch-src

WORKDIR /pytorch-src
ENV PYTORCH_ROCM_ARCH=${ROCM_ARCH}
ENV USE_ROCM=1
ENV MAX_JOBS=${BUILD_PARALLEL_LEVEL}
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-therock-experimental-pytorch-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    PATH="/opt/rocm/bin:/pytorch-venv/bin:$PATH" MAX_JOBS=$jobs \
    CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ \
    BUILD_TEST=0 \
    /pytorch-venv/bin/python3 setup.py bdist_wheel \
    && /pytorch-venv/bin/pip install dist/*.whl

FROM pytorch-builder AS torchvision-builder
ARG TORCHVISION_REF
ARG BUILD_PARALLEL_LEVEL
RUN git clone --branch "${TORCHVISION_REF}" --depth 1 \
        https://github.com/pytorch/vision.git /torchvision-src
WORKDIR /torchvision-src
RUN PATH="/opt/rocm/bin:/pytorch-venv/bin:$PATH" \
    /pytorch-venv/bin/python3 setup.py bdist_wheel \
    && /pytorch-venv/bin/pip install dist/*.whl

FROM pytorch-builder AS torchaudio-builder
ARG TORCHAUDIO_REF
ARG BUILD_PARALLEL_LEVEL
RUN git clone --branch "${TORCHAUDIO_REF}" --depth 1 \
        https://github.com/pytorch/audio.git /torchaudio-src
WORKDIR /torchaudio-src
RUN PATH="/opt/rocm/bin:/pytorch-venv/bin:$PATH" \
    /pytorch-venv/bin/python3 setup.py bdist_wheel \
    && /pytorch-venv/bin/pip install dist/*.whl

FROM python-base AS ort-builder
ARG ROCM_ARCH=gfx803
ARG ORT_VERSION
ARG BUILD_PARALLEL_LEVEL

COPY --from=migraphx-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /ort-venv --python 3.12 --seed \
    && /ort-venv/bin/pip install --no-cache-dir numpy packaging

RUN git clone --recursive --branch "${ORT_VERSION}" --depth 1 --shallow-submodules \
        https://github.com/microsoft/onnxruntime.git /ort-src

WORKDIR /ort-src
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-therock-experimental-ort-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then jobs=$(nproc); fi; \
    PATH="/opt/rocm/bin:/ort-venv/bin:$PATH" /ort-venv/bin/python3 tools/ci_build/build.py \
        --build_dir build --config Release --parallel "$jobs" \
        --use_migraphx --rocm_home /opt/rocm \
        --migraphx_home /opt/rocm \
        --skip_tests --build_wheel \
    && /ort-venv/bin/pip install build/Release/dist/*.whl

FROM ${ORT_IMAGE} AS ort-export
FROM ${TORCHVISION_IMAGE} AS torchvision-export
FROM ${TORCHAUDIO_IMAGE} AS torchaudio-export

FROM python-base AS final
COPY --from=migraphx-export /opt/rocm /opt/rocm
COPY --from=ort-export /ort-venv /ort-venv
COPY --from=torchvision-export /pytorch-venv /pytorch-venv
COPY --from=torchaudio-export /torchaudio-src/dist/*.whl /tmp/

RUN /pytorch-venv/bin/pip install /tmp/*.whl && rm -f /tmp/*.whl

RUN /ort-venv/bin/python3 -c "import onnxruntime as ort; \
    assert 'MIGraphXExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()" \
    && /pytorch-venv/bin/python3 -c "import torch, torchvision, torchaudio; print(torch.__version__)" \
    && /pytorch-venv/bin/python3 -c "import migraphx" \
    && echo "OK: therock-experimental final image sanity checks passed"
