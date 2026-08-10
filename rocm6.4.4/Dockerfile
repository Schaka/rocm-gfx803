# syntax=docker/dockerfile:1
# Polaris / gfx803 variant: MIGraphX + ONNX Runtime + PyTorch on ROCm 6.4.x.
# User-facing docs (versions, packages, caveats): gfx803/README.md
#
# Separate from the main Dockerfile on purpose. gfx803 (RX 460/470/480/550/560/
# 570/580/590) cannot be an arch in that build's matrix, because it isn't a
# different GPU target on the same stack -- it's a different ROCm major:
#
#   * ROCm 7's ROCR-Runtime rejects agents whose HSA DoorbellType is 0 or 1
#     ("legacy" doorbells), which is what Polaris reports. The GPU agent is
#     dropped before enumeration, so rocminfo returns HSA_STATUS_ERROR and
#     clinfo sees zero devices -- see https://github.com/ROCm/clr/issues/269.
#     Restoring it means patching and rebuilding ROCR itself.
#   * ROCm 6.x has no such check: the stock rocm/dev-ubuntu-24.04:6.4.x-complete
#     image enumerates a Polaris card as shipped, no runtime patch required.
#
# So this file pins the last ROCm 6 release (6.4.4) rather than carrying the
# ROCR patch set on 7.x. Everything here is version-pinned and built manually
# (see .github/workflows/gfx803.yml) -- there is no nightly for it.
#
# What still has to be rebuilt from source, and why:
#
#   ROCR      -- libhsa-runtime64 (libhsakmt is statically linked into it).
#                On gfx803, immediate recycling of freed GPU virtual addresses
#                -- in both libhsakmt's fmm aperture free-list and the
#                SimpleHeap fragment sub-allocator above it -- makes MIOpen's
#                CK reduction kernels intermittently mis-read data (ORT
#                ReductionOpTest.ReduceSum_apex_matrix_large). Rebuilt with
#                patches/rocr/va-reuse-defer.sh, which defers freed-VAs on a
#                length-bounded FIFO and defaults the fragment allocator to
#                off. See KERNEL_BUGS.md and the patch header.
#
#   rocBLAS   -- dropped gfx803 from its default TARGET_LIST at ROCm 6.0, but
#                the gfx803 Tensile logic (Logic/asm_full/r9nano/*.yaml) is
#                still in the tree, so `rmake.py -a gfx803` builds it back.
#                The base image's prebuilt rocBLAS has no gfx803 code objects.
#   MIGraphX  -- prebuilt for gfx900+ only, same as on the main image.
#   PyTorch   -- no gfx803 wheel has ever been published.
#   ORT       -- built against the MIGraphX above.
#   MIOpen    -- the base image's prebuilt MIOpen keeps its
#                Ellesmere/Baffin/Polaris10/Polaris11 -> gfx803 device map and
#                its gfx803 asm/Winograd conv solvers fine (it only lacked a
#                gfx803 .kdb, the *tuning* database -- untuned, not broken).
#                But ConvOclDirectFwd (-> MIOpenConvUniBatchNormActiv via the
#                fused ConvActivationFusion path) has a genuine out-of-bounds
#                weights-buffer read for grouped/depthwise convolutions on
#                gfx803: confirmed via a minimal, deterministic, standalone
#                repro and two independent AMD_SERIALIZE_KERNEL=3-attributed
#                production crashes (see KERNEL_BUGS.md). Rebuilt with
#                patches/miopen/conv-direct-fwd-grouped-oob.sh, which makes
#                that solver's applicability check reject grouped convs
#                outright, so MIOpen routes them to a different solver
#                instead of ever reaching the broken kernel.
#
# What is switched off, because no version of it has ever supported gfx8:
#
#   rocMLIR   -- AmdArchDb.cpp knows gfx9/gfx10/gfx11/gfx12 only, on every
#                branch from rocm-5.7 through rocm-6.4. Never had gfx8.
#   CK        -- composable_kernel's floor is gfx908 in the 6.x line (gfx900
#                on develop). Never had gfx8 either.
#   hipBLASLt -- gfx90a+.
#
# Expect this to be slow: untuned MIOpen, no packed-fp16, no dot instructions.
# It is meant to make a 20-euro card usable, not competitive.
#
# Build: docker build -f gfx803/Dockerfile -t <tag> gfx803
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:6.4.4-complete

# Component-image references, same indirection as the main Dockerfile: in CI
# each component builds as its own job/image and the next stage COPYs the
# prebuilt result instead of recompiling. Defaults point at the local stage
# names so a plain one-shot `docker build -f gfx803/Dockerfile gfx803`
# still works.
ARG ROCBLAS_IMAGE=rocblas-builder
ARG MIOPEN_IMAGE=miopen-builder
ARG MIGRAPHX_IMAGE=migraphx-builder
ARG PYTORCH_IMAGE=pytorch-builder
ARG ORT_IMAGE=ort-builder
ARG ROCR_IMAGE=rocr-builder

# Everything below is pinned. Unlike the main image there is no develop-tracking
# ref and so no SOURCE_DATE cache-bust: each of these refs is immutable, so its
# clone layer's cache key only changes when the pin does, which is what we want.
#
# ROCM_VERSION picks the rocBLAS/rocm-cmake source tags and must match
# BASE_IMAGE's ROCm release -- rocBLAS built from a different release than the
# hipBLAS/hipBLASLt it links against is not a supported combination.
ARG ROCM_VERSION=6.4.4

# MIGraphX 6.4 is the newest release branch on the ROCm 6 line.
ARG MIGRAPHX_REF=release/rocm-rel-6.4

# PyTorch from ROCm's fork, not upstream: the fork carries the ROCm-side fixes
# for the release it's cut against, and it's what the gfx803 community builds
# have used. release/2.8 matches what AMD ships in their own
# rocm/onnxruntime:rocm6.4.4_* image (torch 2.8.0).
#
# Caveat worth knowing before debugging a failure here: the *hardware*-verified
# gfx803 torch is release/2.6 (that's what has actually been run on RX 570/580
# by the gfx803_rocm project). 2.8 is verified against ROCm 6.4.4, not against
# gfx803. If torch misbehaves on-device, dropping to
# --build-arg PYTORCH_REF=release/2.6 TORCHVISION_REF=v0.21.0
# TORCHAUDIO_REF=v2.6.0 is the known-good fallback.
ARG PYTORCH_REF=release/2.8
ARG TORCHVISION_REF=v0.23.0
ARG TORCHAUDIO_REF=v2.8.0

# v1.22.2 is the newest ORT release with a real ROCMExecutionProvider --
# confirmed directly against upstream history: PR #25181 ("Delete ROCM EP,
# because there is no active development and we have another AMD GPU
# EP(migraphx) to use") removed onnxruntime/core/providers/rocm/ entirely
# right before the v1.23.0 tag (v1.22.2 still has all 76 files, v1.23.0 has
# zero). MIGraphX EP alone isn't an equivalent replacement on gfx803 --
# unlike gfx9+, MIGraphX has no Composable Kernel and no MLIR to fuse with
# here (see the requirements.txt strip further down), so ROCM EP staying
# available as the fallback path matters more on this arch than most.
# AMD's own published rocm/onnxruntime tags pair ROCm 6.4 with ORT 1.21, but
# the actual upstream cutoff is 1.22.2, not 1.21 -- confirmed by diffing
# v1.21.1..v1.22.2's onnxruntime/core/providers/rocm and
# onnxruntime/contrib_ops/rocm/bert trees: the changes there are pure
# interface-compatibility churn (e.g. threading a new GraphOptimizerRegistry
# param through GetCapability), not behavior changes, so nothing about
# using 1.22.2 instead of 1.21.1 should be riskier in practice.
ARG ORT_VERSION=v1.22.2

ARG BUILD_PARALLEL_LEVEL=auto

# Shared ancestor. Ubuntu 24.04's native python3 IS 3.12, which is the version
# every wheel here has to target (downstream consumers pin numpy in a way that
# only resolves against onnx's deps under 3.12) -- so unlike the main
# Dockerfile, which has to overlay a uv-managed 3.12 onto a 3.14 base, this
# variant just uses the system interpreter throughout.
FROM ${BASE_IMAGE} AS python-base
ENV DEBIAN_FRONTEND=noninteractive
# PIP_BREAK_SYSTEM_PACKAGES: Ubuntu 24.04 marks the system interpreter
# externally-managed (PEP 668), which makes the `pip3 install` calls inside
# rocBLAS's rmake.py and PyTorch's requirements step fail outright. Every stage
# here is a throwaway build container, so overriding is the least invasive fix;
# the wheels that matter are installed into a real venv in the final stage.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ROCR-Runtime (libhsa-runtime64, with libhsakmt statically linked in)
# rebuilt from source with patches/rocr/va-reuse-defer.patch: on gfx803,
# immediate recycling of freed GPU virtual addresses (both in libhsakmt's
# fmm aperture free-list and in the hsa-runtime SimpleHeap fragment
# sub-allocator above it) makes MIOpen's CK reduction kernels intermittently
# mis-read freshly written data -- ORT ReductionOpTest.ReduceSum_
# apex_matrix_large. The patch defers freed-VAs on a length-bounded FIFO
# and defaults the fragment allocator to off. See the patch file and
# KERNEL_BUGS.md for the full investigation, and gfx803/tools/reduce-harness/
# for the repro harness. Only libhsa-runtime64 is rebuilt: the KFD-layer
# allocator is statically linked into it, so clr/HIP does not need a rebuild.
FROM python-base AS rocr-builder
ARG ROCM_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config xxd \
        libnuma-dev libdrm-dev libelf-dev \
        rocm-llvm-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --branch "rocm-${ROCM_VERSION}" --depth 1 \
        https://github.com/ROCm/ROCR-Runtime.git /rocr-src
COPY patches/rocr/ /rocr-patches/
RUN sh /rocr-patches/va-reuse-defer.sh /rocr-src
RUN cmake -S /rocr-src -B /rocr-src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/llvm" \
    && cmake --build /rocr-src/build -j"$(nproc)" \
    && cmake --install /rocr-src/build \
    && rm -rf /rocr-src/build

FROM python-base AS rocblas-builder

ARG ROCM_VERSION
ARG ROCM_ARCH=gfx803
ARG BUILD_PARALLEL_LEVEL

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config gfortran ccache \
        libmsgpack-dev wget \
    && rm -rf /var/lib/apt/lists/*
# rmake's -d installer apt-installs python3-yaml/python3-joblib, which the
# base image's apt lists don't have (no `universe` component enabled) --
# `apt-get install` 404s with "Unable to locate package". pip them instead and
# drop -d below rather than fight the base image's sources.list.
RUN pip3 install pyyaml joblib

RUN git clone --branch "rocm-${ROCM_VERSION}" --depth 1 \
        https://github.com/ROCm/rocBLAS.git /rocblas-src

# gfx803's Tensile kernels miscompute whenever WorkGroupMapping != 1 -- silently,
# with rocblas_status_success. This rewrites the solution logic before Tensile
# generates any assembly from it; the script header carries the measurements and
# the reasoning, and it fails the build if it ever stops matching upstream.
COPY patches/rocblas/ /rocblas-patches/
RUN sh /rocblas-patches/wgm-miscompute.sh /rocblas-src

# Separately: gfx803's tuned Tensile assembly GEMM kernels silently miscompute
# when *every* dimension of the problem is small (they are correct from roughly
# 48x48x48 up, and correct for thin shapes where one dimension is large). This
# makes rocBLAS's Tensile dispatch prefer a HIP source kernel for exactly those
# small fp32 problems, which is both correct and faster at that size. Fixed at
# the rocBLAS level rather than in ORT so every consumer benefits -- takes
# onnxruntime_test_all from 96 failures to 16. The script header carries the
# measurements and it fails the build if the file it patches ever moves.
RUN sh /rocblas-patches/small-gemm-assembly-miscompute.sh /rocblas-src

WORKDIR /rocblas-src
# rmake.py directly rather than install.sh: as of 6.4, install.sh's own getopt
# only understands :cdghik and forwards anything else to rmake.py, so passing
# -a through it is fragile -- and install.sh's help text explicitly says
# invoking rmake.py directly is supported. -i builds and installs the package
# into /opt/rocm over the base image's gfx900+ rocBLAS. No -d: its distro
# package install is covered above except PyYAML/joblib (pip'd instead, see
# above) -- everything else it wants (cmake, msgpack-dev, wget, python3-venv,
# python3-pip, make) is already installed here or in python-base.
#
# --no_hipblaslt: rocBLAS can dispatch through hipBLASLt, which has no gfx803
# support at all (gfx90a+). Leaving it linked would mean a runtime path that
# can only fail on this hardware.
#
# Disk: Tensile generates an enormous intermediate tree even for a single
# architecture, and a hosted runner has ~14GB of disk after the base image.
# rmake.py has its own --cleanup flag for this, but it deletes the entire
# build/release tree -- including the install output this stage needs to
# copy out to /opt/rocm -- as its last internal action, before returning
# control to the shell. That raced against (and silently defeated) the copy
# step below. --cleanup is intentionally NOT passed here; the `rm -rf
# /rocblas-src/build` after the copy reclaims the same disk without the race.
#
# Job count uses the same MemAvailable/4GB-per-job "auto" sizing as the main
# Dockerfile's compile stages.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocblas-ccache \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "rocBLAS build: arch ${ROCM_ARCH}, $jobs parallel jobs"; \
    python3 ./rmake.py -i -a "${ROCM_ARCH}" -j "$jobs" \
        --no_hipblaslt \
    # rmake's own -i install step does NOT land in /opt/rocm: its
    # CMAKE_INSTALL_PREFIX is the relative "rocblas-install", which cmake
    # resolves inside the build directory (/rocblas-src/build/release/
    # rocblas-install), not against ROCM_DIR/ROCM_PATH/CPACK_PACKAGING_
    # INSTALL_PREFIX -- those three only point the *build* at the existing
    # ROCm install for headers/deps, they don't control install output.
    # Confirmed from a real build log: rmake correctly compiles and installs
    # TensileLibrary_lazy_gfx803.dat and friends, but into that relative path.
    #
    # A second bug compounded this: rmake.py's own --cleanup flag (previously
    # passed here) deletes the entire build/release tree -- rocblas-install
    # included -- as its last internal action, before returning control to
    # the shell. So even after adding a `cp -a rocblas-install/. /opt/rocm/`
    # step, the copy ran against an already-emptied source and silently
    # copied nothing (confirmed: a second build log showed the copy "succeed"
    # with zero gfx803 files landing in /opt/rocm). --cleanup is dropped
    # entirely below; we already do our own `rm -rf /rocblas-src/build`
    # right after the copy, so nothing is lost by not asking rmake to do it
    # itself first.
    && echo "Copying rocBLAS gfx803 install output into /opt/rocm..." \
    && cp -a /rocblas-src/build/release/rocblas-install/. /opt/rocm/ \
    # rmake's own install produces librocblas.so.4.4 (rocBLAS's own SOVERSION,
    # gfx803-agnostic) -- but the base image's pre-existing librocblas.so.4
    # symlink points at librocblas.so.4.4.60404: AMD's own downstream
    # packaging appends a ROCm-release patch suffix beyond rocBLAS's own
    # versioning, in the deterministic form major*10000+minor*100+patch (e.g.
    # ROCm 6.4.4 -> 60404). Without landing our build at that exact filename,
    # the copy above just adds an unused file alongside the untouched stock
    # one, and every real load still resolves through the symlink to the
    # gfx900+ stock build with no gfx803 code objects at all. Deriving the
    # suffix from ROCM_VERSION rather than hardcoding "60404" keeps this
    # correct if the base image ever moves to a different 6.4.x patch level.
    && rocblas_suffix=$(echo "${ROCM_VERSION}" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}') \
    && echo "Landing rocBLAS gfx803 build at librocblas.so.4.4.${rocblas_suffix} so the existing librocblas.so.4 symlink resolves to it instead of the stock build" \
    && cp -a /opt/rocm/lib/librocblas.so.4.4 "/opt/rocm/lib/librocblas.so.4.4.${rocblas_suffix}" \
    && rm -rf /rocblas-src/build \
    # Verify the copy actually landed gfx803 Tensile kernels in /opt/rocm
    # before letting this stage be considered a success. Without this check,
    # a broken rebuild or a future change to rmake's install layout still
    # exits 0 and gets pushed to the shared :gfx803 tag -- and every
    # downstream consumer (migraphx, pytorch, the final image) then inherits
    # a rocBLAS with no gfx803 code objects at all, failing only much later
    # at runtime with "Illegal seek for GPU arch: gfx803" instead of here, at
    # build time.
    #
    # /opt/rocm itself is a symlink (-> /etc/alternatives/rocm ->
    # /opt/rocm-6.4.4 in this base image). GNU find's default -P mode never
    # follows a symlink given as the search root -- it treats /opt/rocm as
    # a leaf node and never recurses into it, so a plain `find /opt/rocm`
    # here always matches zero regardless of what's actually installed
    # (confirmed: an unfiltered `find /opt/rocm/lib/rocblas/library` --
    # which resolves the same symlink transparently via normal path
    # resolution, since it names the target directory explicitly instead of
    # relying on find's own recursion -- showed the gfx803 files were
    # present all along). `-L` makes find follow the symlink so it actually
    # recurses.
    && echo "Verifying gfx803 Tensile library is present in /opt/rocm..." \
    && if ! find -L /opt/rocm -iname "*TensileLibrary*gfx803*" | grep -q .; then \
        echo "FATAL: /opt/rocm has no gfx803 Tensile library after the copy." >&2; \
        echo "Expected a file matching *TensileLibrary*gfx803* under /opt/rocm -- found none." >&2; \
        echo "Contents of /opt/rocm/lib/rocblas/library (if present):" >&2; \
        find /opt/rocm/lib/rocblas/library -maxdepth 1 2>&1 >&2 || true; \
        exit 1; \
    fi \
    && echo "OK: gfx803 Tensile library confirmed present in /opt/rocm." \
    # Verify librocblas.so.4 (the symlink everything actually loads through)
    # now resolves to OUR build, not the stock one left over from the base
    # image -- this is exactly the bug that motivated the rocblas_suffix step
    # above (a correct build that just never got loaded at runtime because
    # the symlink still pointed at the untouched stock file). Comparing sizes
    # rather than paths, since `cp -a` above made them the same file's
    # content under two names, not a symlink to compare against.
    && echo "Verifying librocblas.so.4 resolves to the gfx803 build, not the stock one..." \
    && resolved="$(readlink -f /opt/rocm/lib/librocblas.so.4)" \
    && if [ "$(stat -c%s "$resolved")" != "$(stat -c%s /opt/rocm/lib/librocblas.so.4.4)" ]; then \
        echo "FATAL: librocblas.so.4 resolves to ${resolved}, which is not our freshly-built librocblas.so.4.4 (size mismatch)." >&2; \
        echo "The stock base-image rocBLAS is still what actually gets loaded at runtime." >&2; \
        exit 1; \
    fi \
    # Verify the resolved library actually embeds gfx803 device code, not an
    # empty .hip_fatbin -- catches a broken/incomplete compile even when the
    # symlink and Tensile checks above both pass (a real failure mode seen
    # during debugging: link succeeded, Tensile was fine, but the plain
    # non-Tensile kernels' device code silently never got embedded).
    && echo "Verifying librocblas.so.4 embeds real gfx803 device code..." \
    && objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocblas_fatbin_check.bin \
    && fatbin_size="$(stat -c%s /tmp/rocblas_fatbin_check.bin)" \
    && rm -f /tmp/rocblas_fatbin_check.bin \
    && if [ "$fatbin_size" -lt 1000000 ]; then \
        echo "FATAL: librocblas.so.4's .hip_fatbin is only ${fatbin_size} bytes -- too small to contain real gfx803 device code (expect several MB)." >&2; \
        exit 1; \
    fi \
    && echo "OK: librocblas.so.4 resolves to a gfx803 build with a ${fatbin_size}-byte .hip_fatbin."

# rocBLAS/Tensile's own SGEMM kernels are unreliable on gfx803 for every shape
# tested (not just the already-fixed WGM8 bug) -- see sgemm-shim/gfx803_sgemm.h
# for the full investigation. This builds an LD_PRELOAD shim that intercepts
# the standard-algo f32 rocblas_sgemm/rocblas_gemm_ex path and routes it to a
# plain, correctness-verified replacement kernel instead; the ENV enabling it
# lives in the final stage.
RUN mkdir -p /opt/rocm-sgemm-shim
COPY patches/rocblas/sgemm-shim/ /opt/rocm-sgemm-shim/
RUN hipcc -O2 -fPIC -shared --offload-arch=gfx803 -I/opt/rocm/include \
        /opt/rocm-sgemm-shim/sgemm_shim.cpp \
        -o /opt/rocm/lib/libgfx803_sgemm_shim.so \
        -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lrocblas -ldl \
    && rm -rf /opt/rocm-sgemm-shim

# Indirection alias (BuildKit rejects a variable in COPY --from, so resolve the
# ARG through a FROM with a static name first).
FROM ${ROCBLAS_IMAGE} AS rocblas-export

FROM ${ROCR_IMAGE} AS rocr-export

FROM python-base AS miopen-builder

ARG ROCM_VERSION
ARG ROCM_ARCH=gfx803
ARG BUILD_PARALLEL_LEVEL

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        rocm-cmake half libboost-system-dev libboost-filesystem-dev \
        libsqlite3-dev libbz2-dev lbzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "rocm-${ROCM_VERSION}" --depth 1 \
        https://github.com/ROCm/MIOpen.git /miopen-src

# MIOpenConvUniBatchNormActiv (generated by ConvOclDirectFwd/
# ConvOclDirectFwdFused, src/solver/conv/conv_ocl_dir2Dfwd.cpp) has an
# out-of-bounds weights-buffer read for grouped convolutions on gfx803 --
# see the script for the full investigation and how it was root-caused.
COPY patches/miopen/ /miopen-patches/
RUN sh /miopen-patches/conv-direct-fwd-grouped-oob.sh /miopen-src

# ConvBinWinogradRxSFused (src/solver/conv_bin_winoRxS_fused.cpp) miscomputes
# for its padding-generalized non-3x3 shapes (e.g. plain 1x1 pointwise convs)
# -- see the patch for the full investigation and why it's scoped to spare
# genuine 3x3 shapes, where the same kernel is verified correct.
RUN sh /miopen-patches/winograd-fused-conv-miscompute.sh /miopen-src

# ReduceCalculation's Prod (multiply) kernel seeds its accumulator with 0
# instead of 1, so miopenReduceCalculationForward(..., PROD) always returns
# all zeros -- a plain logic bug in arch-generic kernel source, not specific
# to gfx8, still present unfixed as of the newest upstream tag we could find
# (rocm-7.2.4). Kept in this repo's patch set since we build MIOpen from
# source here anyway; see the patch for the full investigation.
RUN sh /miopen-patches/reduce-prod-wrong-identity.sh /miopen-src

# miopenReduceTensor's dynamic-reduction dispatch intermittently returns wrong
# sums on gfx803 (confirmed via ReductionOpTest.ReduceSum_apex_matrix_large and a
# standalone repro). The prior kernel-cache-eviction workaround
# (reduce-kernel-cache-eviction.patch) passed every standalone repro but failed
# ORT's real test suite, and was abandoned. A NEW narrow workaround
# (reduce-program-bound-eviction.patch, which bounds how many distinct reduction
# programs stay resident instead of evicting every call) is under validation and
# opt-in here via ENABLE_REDUCE_BOUND -- NOT part of any default production
# build until it passes the real ORT suite AND a perf benchmark. See
# KERNEL_BUGS.md, "The ReduceSum kernel-cache mystery".
ARG ENABLE_REDUCE_BOUND=0
RUN if [ "$ENABLE_REDUCE_BOUND" = "1" ]; then \
        sh /miopen-patches/reduce-program-bound-eviction.sh /miopen-src; \
    else \
        echo "reduce-program-bound workaround NOT applied (opt-in, gfx803 validation)"; \
    fi

# MIOpen's requirements.txt unconditionally pulls in composable_kernel and
# rocMLIR, neither of which has ever supported gfx8 (see the top-of-file
# comment) -- composable_kernel alone is tens of thousands of instantiation
# files and would multiply this stage's build time for code that could never
# run on this card. Filtered out before install_deps.cmake ever sees them.
RUN grep -v "composable_kernel\|rocMLIR" /miopen-src/requirements.txt \
        > /miopen-src/requirements-gfx803.txt \
    && cp /miopen-src/requirements-gfx803.txt /miopen-src/requirements.txt

WORKDIR /miopen-src
# install_deps.cmake's own cget(install -f requirements.txt) call uses a
# relative path, so it must run with /miopen-src as the working directory --
# invoking it via an absolute `cmake -P /miopen-src/install_deps.cmake` from
# elsewhere does NOT imply that cwd, and cget then silently finds no
# requirements.txt and installs nothing beyond its own recipes bootstrap
# (looks like success -- exits quickly, no error -- but leaves boost/sqlite3/
# zstd/nlohmann_json/etc missing, which only surfaces later as a confusing
# "could not find nlohmann_json" CMake configure failure).
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-miopen-ccache \
    cmake -P install_deps.cmake --minimum --prefix /miopen-deps

# -DMIOPEN_USE_COMPOSABLEKERNEL/MLIR/HIPBLASLT=Off: none of the three has
# ever supported gfx8 (see top-of-file comment); Off here matches the
# filtered requirements.txt above and the main Dockerfile's own gfx803
# switches. -DMIOPEN_BUILD_DRIVER/BUILD_TESTING=Off: neither ships in the
# final image, skipping them cuts real build time.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-miopen-ccache \
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
    && make -j"$jobs" \
    # rmake-style disk reclaim: MIOpen's build tree is large and this stage's
    # only output the final image needs is the installed .so, copied below.
    && cp -a lib/libMIOpen.so* /tmp/ \
    && cd .. && rm -rf build

RUN echo "Copying MIOpen gfx803 build into /opt/rocm..." \
    && rocm_suffix=$(echo "${ROCM_VERSION}" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}') \
    && resolved="$(readlink -f /opt/rocm/lib/libMIOpen.so.1)" \
    && stock_size="$(stat -c%s "$resolved")" \
    # Land at the exact versioned filename the base image's libMIOpen.so ->
    # libMIOpen.so.1 -> libMIOpen.so.1.0.<suffix> symlink chain resolves to
    # (overwriting the stock file in place, same reasoning as rocBLAS's
    # rocblas_suffix above) so the existing symlink picks up our build
    # automatically instead of silently keeping the untouched stock one.
    && cp -a /tmp/libMIOpen.so.1.0 "$resolved" \
    && rm -f /tmp/libMIOpen.so* \
    # Verify our content actually landed: our gfx803-only, CK/MLIR-off build
    # is always meaningfully smaller than the base image's multi-arch stock
    # one (~380MB here vs ~695MB stock at the time of writing), so an
    # unchanged size means the overwrite silently didn't happen.
    && new_size="$(stat -c%s "$resolved")" \
    && echo "libMIOpen.so.1 resolved path: $resolved (stock ${stock_size} bytes -> new ${new_size} bytes)" \
    && if [ "$new_size" = "$stock_size" ]; then \
        echo "FATAL: libMIOpen.so.1 is still ${new_size} bytes, unchanged from stock -- our build did not land." >&2; \
        exit 1; \
    fi \
    && echo "OK: MIOpen gfx803 build with the grouped-conv OOB fix is in place."

FROM ${MIOPEN_IMAGE} AS miopen-export

FROM python-base AS migraphx-builder

ARG ROCM_VERSION
ARG ROCM_ARCH=gfx803
ARG MIGRAPHX_REF
ARG BUILD_PARALLEL_LEVEL

# /opt/rocm here is the base image's ROCm with the gfx803 rocBLAS built over
# it -- MIGraphX links rocBLAS, so it has to see the rebuilt one, not the
# gfx900+ stock copy.
COPY --from=rocblas-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# Fresh rocm-cmake at the matching release tag, same reasoning as the main
# Dockerfile: MIGraphX's CMakeLists.txt uses macros (rocm_add_version_resource)
# that the bundled copy doesn't reliably carry.
RUN git clone --branch "rocm-${ROCM_VERSION}" --depth 1 \
        https://github.com/ROCm/rocm-cmake.git /rocm-cmake-src \
    && cmake -S /rocm-cmake-src -B /rocm-cmake-src/build -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    && cmake --build /rocm-cmake-src/build --target install \
    && rm -rf /rocm-cmake-src

# rbuild builds every dependency listed in requirements.txt at the exact pinned
# commit MIGraphX expects. Isolated venv so its `cget` dependency doesn't fight
# the system interpreter's PEP 668 marking.
RUN python3 -m venv /rbuild-venv \
    && /rbuild-venv/bin/pip install --no-cache-dir \
        https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Fixes to MIGRAPHX_REF's pinned commit that upstream merged to develop but
# never backported to this release branch. patches/ is grouped by what gets
# patched (migraphx/, rocblas/) rather than by arch: this whole directory is
# gfx803-exclusive, so a patches/<arch>/ level underneath it only repeated the
# name of its own parent. Each patch documents its own why/what in its header;
# applied right after the clone so a patch that stops applying (e.g.
# MIGRAPHX_REF moves) fails the build loudly instead of silently shipping
# unpatched code.
COPY patches/migraphx/ /migraphx-patches/
# No SHELL ["/bin/bash", ...] override in this file (default /bin/sh), so
# this loop uses the POSIX-portable "does the glob match anything" idiom
# instead of bash's compgen: an unmatched glob falls through as its own
# literal string, and `[ -e ... ]` on that fails, skipping it.
RUN git clone --branch "${MIGRAPHX_REF}" --depth 1 \
        https://github.com/ROCm/AMDMIGraphX.git /migraphx-src; \
    for p in /migraphx-patches/*.patch; do \
        [ -e "$p" ] || continue; \
        echo "applying ${p}"; \
        git -C /migraphx-src apply --verbose "${p}"; \
    done

WORKDIR /migraphx-src
# Drop composable_kernel and rocMLIR from requirements.txt before rbuild runs.
# Turning them off via -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off/-DMIGRAPHX_ENABLE_MLIR=Off
# is necessary but NOT sufficient: rbuild builds everything requirements.txt
# lists regardless of MIGraphX's own cmake options, so both would still be
# compiled for gfx803 -- and neither supports gfx8 on any branch, so both fail.
# Removing the lines is what actually keeps them out of the build.
RUN sed -i '/composable_kernel/d; /rocMLIR/d' requirements.txt \
    && ! grep -q 'composable_kernel\|rocMLIR' requirements.txt

# src/targets/gpu/mlir.cpp includes <mlir-c/Dialect/RockEnums.h> above its
# `#ifdef MIGRAPHX_MLIR` guard, and is compiled into migraphx_gpu unconditionally
# (src/targets/gpu/CMakeLists.txt never gates it on MIGRAPHX_ENABLE_MLIR) --
# so -DMIGRAPHX_ENABLE_MLIR=Off alone still fails to find the header once
# rocMLIR itself is stripped from requirements.txt above. The header is just
# two plain C enums with no further includes (rocMLIR
# mlir/include/mlir-c/Dialect/RockEnums.h, unchanged since it was added), so
# vendor that one file instead of building all of rocMLIR/LLVM for it.
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

# src/targets/gpu/jit/mlir.cpp calls is_module_fusible(), dump_mlir_to_mxr()
# and dump_mlir_to_file() unconditionally (no MIGRAPHX_MLIR guard at all in
# that file), but mlir.cpp only *defines* them inside its own
# `#ifdef MIGRAPHX_MLIR` block -- the `#else` stub block right below it stubs
# most other MLIR entry points (compile_mlir, insert_mlir,
# get_tuning_config_mlir, the two dump_mlir(module[, shapes]) overloads) but
# misses these three, so with MLIR off the symbols are declared, called, and
# never defined: migraphx_gpu.so links fine but importing the python module
# (which pulls in jit/mlir.cpp's translation unit) fails at runtime with
# "undefined symbol ...". Add the missing stubs, consistent with the pattern
# of their neighbors -- all three are only ever reached from the MLIR
# compiler pass, otherwise unreachable with MLIR disabled, so the stub bodies
# don't matter functionally.
RUN sed -i \
    's/std::string dump_mlir(module) { return {}; }/std::string dump_mlir(module) { return {}; }\n\nbool is_module_fusible(const module\&, const context\&, const value\&) { return false; }\n\nvoid dump_mlir_to_file(module, const std::vector<shape>\&, const fs::path\&) {}\n\nvoid dump_mlir_to_mxr(module, const std::vector<instruction_ref>\&, const fs::path\&) {}/' \
    src/targets/gpu/mlir.cpp \
    && grep -q 'bool is_module_fusible(const module&, const context&, const value&) { return false; }' src/targets/gpu/mlir.cpp \
    && grep -q 'void dump_mlir_to_file(module, const std::vector<shape>&, const fs::path&) {}' src/targets/gpu/mlir.cpp \
    && grep -q 'void dump_mlir_to_mxr(module, const std::vector<instruction_ref>&, const fs::path&) {}' src/targets/gpu/mlir.cpp

# Same ccache-mount and job-capping reasoning as the main Dockerfile. Without
# rocMLIR in the graph this build no longer compiles LLVM, so it is far lighter
# than the main image's migraphx stage -- the cap is kept anyway because the
# MIGraphX TUs themselves are still multi-GB.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-migraphx-ccache \
    ulimit -s unlimited && \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "MIGraphX build: using $jobs parallel jobs"; \
    CMAKE_BUILD_PARALLEL_LEVEL=$jobs \
    PATH=/rbuild-venv/bin:$PATH /rbuild-venv/bin/rbuild build -d /migraphx-deps -B build -G Ninja \
        --cxx=/opt/rocm/llvm/bin/clang++ --cc=/opt/rocm/llvm/bin/clang \
        "-DGPU_TARGETS=${ROCM_ARCH}" \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DMIGRAPHX_ENABLE_PYTHON=On \
        -DPython3_EXECUTABLE=/usr/bin/python3 \
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

# Same version stamp as the main image, so downstream consumers can detect a
# MIGraphX change and invalidate a compiled-model cache.
RUN echo "${MIGRAPHX_REF} $(git -C /migraphx-src rev-parse HEAD)" > /opt/rocm/migraphx-version.txt

FROM ${MIGRAPHX_IMAGE} AS migraphx-export

FROM python-base AS pytorch-builder

ARG ROCM_ARCH=gfx803
ARG PYTORCH_REF
ARG TORCHVISION_REF
ARG TORCHAUDIO_REF
ARG BUILD_PARALLEL_LEVEL

# torch links rocBLAS but not MIGraphX, so this stage only needs the rocBLAS
# rebuild -- keeping it off the MIGraphX chain lets CI run the two in parallel.
COPY --from=rocblas-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libopenblas-dev libdrm-dev libomp-dev \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Build venv rather than the bare system interpreter: torchvision/torchaudio
# both import the just-built torch at their own build time, so all three need
# one consistent site-packages, and Ubuntu's python3-yaml/python3-filelock
# would otherwise shadow the versions torch's requirements.txt pins.
RUN python3 -m venv /build-venv \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools \
    && /build-venv/bin/pip install --no-cache-dir numpy pyyaml typing_extensions requests six
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch "${PYTORCH_REF}" --depth 1 --shallow-submodules \
        https://github.com/ROCm/pytorch.git /pytorch

WORKDIR /pytorch
RUN pip install --no-cache-dir -r requirements.txt

# setup.py does not hipify CUDA sources itself; build_amd.py has to run first
# or CMake fails looking for .hip sources that don't exist yet.
RUN python3 tools/amd_build/build_amd.py

# USE_FLASH_ATTENTION/USE_MEM_EFF_ATTENTION: default on for ROCm and pull in
# aotriton, which has no gfx803 support whatsoever (and configures its own
# isolated venv that breaks this stage's build anyway, same as on the main
# image).
#
# USE_DISTRIBUTED: pulls in NCCL/NVSHMEM paths that don't build under ROCm at
# these tags, and a single Polaris card has no use for it.
#
# TORCH_BLAS_PREFER_HIPBLASLT=0 is baked into the final image rather than set
# here -- it's a *runtime* dispatch preference. torch 2.8 links hipBLASLt
# unconditionally (cmake/Dependencies.cmake), so it cannot be excluded at build
# time; the env var is what keeps GEMMs off a library with no gfx803 kernels.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-pytorch-ccache \
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
        USE_DISTRIBUTED=0 \
        python3 setup.py bdist_wheel

# torchvision and torchaudio both compile extensions against the installed
# torch, so torch has to be importable before either configures -- install the
# wheel just built, then build the other two in dependency order. Their wheels
# are collected into /wheels alongside torch's so the final stage has one
# directory to copy.
RUN pip install --no-cache-dir dist/torch*.whl \
    && mkdir -p /wheels && cp dist/torch*.whl /wheels/

RUN git clone --recursive --branch "${TORCHVISION_REF}" --depth 1 --shallow-submodules \
        https://github.com/pytorch/vision.git /vision
WORKDIR /vision
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-vision-ccache \
    env USE_ROCM=1 USE_CUDA=0 "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        FORCE_CUDA=0 \
        python3 setup.py bdist_wheel \
    && pip install --no-cache-dir dist/torchvision-*.whl \
    && cp dist/torchvision-*.whl /wheels/

RUN git clone --recursive --branch "${TORCHAUDIO_REF}" --depth 1 --shallow-submodules \
        https://github.com/pytorch/audio.git /audio
WORKDIR /audio
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-audio-ccache \
    env USE_ROCM=1 USE_CUDA=0 "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        FORCE_CUDA=0 USE_FFMPEG=1 \
        python3 setup.py bdist_wheel \
    && cp dist/torchaudio*.whl /wheels/

FROM ${PYTORCH_IMAGE} AS pytorch-export

FROM python-base AS ort-builder

ARG ROCM_ARCH=gfx803
ARG ORT_VERSION

COPY --from=migraphx-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config ccache \
        libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /build-venv \
    && /build-venv/bin/pip install --no-cache-dir -U pip wheel setuptools \
    && /build-venv/bin/pip install --no-cache-dir numpy packaging cmake
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch "${ORT_VERSION}" --depth 1 \
        https://github.com/microsoft/onnxruntime.git /onnxruntime

# MultiHeadAttention's ROCm "Generic" pipeline (the only one available once
# Composable Kernel is compiled out, see onnxruntime_USE_COMPOSABLE_KERNEL=OFF
# below) refuses the plain three-tensor MHA call whenever query
# sequence_length > 1 -- an ordinary shape, not an edge case -- because it was
# never given the Q/K/V transpose its own GEMMs need. See the patch for the
# full investigation; this is an ORT bug, not rocBLAS/MIOpen.
COPY patches/onnxruntime/ /onnxruntime-patches/
RUN sh /onnxruntime-patches/mha-basic-mode-no-viable-op.sh /onnxruntime

# ROCm's generic TopK kernel picks a different winner among exactly-tied
# candidates from run to run on identical input -- a hipCUB block-primitive
# tie-break defect, not a gfx803-specific bug. Surfaced as beam search
# decoding a different (but individually plausible) token sequence each run.
# See the patch for the full investigation, including two fixes that looked
# obvious and were rejected (one crashes, one can't safely cover this shape).
RUN sh /onnxruntime-patches/topk-radix-tiebreak-nondeterministic.sh /onnxruntime

# ORT's cmake/deps.txt fetches Eigen as a GitLab commit-archive zip and pins
# its SHA1. Those archives aren't reproducible server-side (gitlab.com/
# libeigen/eigen/-/issues/2744), so the hash drifts and FetchContent's
# verification fails intermittently. Fetch the exact pinned commit via git
# instead and point FetchContent at it directly (FETCHCONTENT_SOURCE_DIR_<name>
# skips the download+hash-verify step for that dependency entirely).
RUN eigen_commit=$(grep '^eigen;' /onnxruntime/cmake/deps.txt | cut -d';' -f2 | grep -oP '(?<=archive/)[0-9a-f]{40}') \
    && mkdir /eigen-src && cd /eigen-src \
    && git init -q \
    && git remote add origin https://gitlab.com/libeigen/eigen.git \
    && git fetch --depth 1 origin "$eigen_commit" \
    && git checkout -q FETCH_HEAD

WORKDIR /onnxruntime
# Unlike the main Dockerfile's v1.27 build, 1.21 still has a real ROCm EP, so
# both --use_rocm and --use_migraphx are passed: the MIGraphX EP handles what
# MIGraphX can compile and the ROCm EP is the fallback for the rest. On gfx803
# that fallback matters more than usual, since MIGraphX here has no CK and no
# MLIR to fuse with.
RUN --mount=type=cache,target=/root/.ccache,id=gfx803-ort-ccache \
    python3 tools/ci_build/build.py \
        --config Release \
        --build_dir /onnxruntime/build \
        --parallel \
        --build_wheel \
        --skip_tests \
        --allow_running_as_root \
        --compile_no_warning_as_error \
        --use_rocm --rocm_home /opt/rocm \
        --use_migraphx --migraphx_home /opt/rocm \
        --cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}" \
        --cmake_extra_defines "CMAKE_C_COMPILER_LAUNCHER=ccache" \
        --cmake_extra_defines "CMAKE_CXX_COMPILER_LAUNCHER=ccache" \
        --cmake_extra_defines "FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen-src" \
        --cmake_extra_defines "CMAKE_POLICY_VERSION_MINIMUM=3.5" \
        --cmake_extra_defines "onnxruntime_USE_COMPOSABLE_KERNEL=OFF" \
    && mkdir -p /onnxruntime/dist && cp /onnxruntime/build/Release/dist/*.whl /onnxruntime/dist/

FROM ${ORT_IMAGE} AS ort-export

FROM python-base

COPY --from=migraphx-export /opt/rocm /opt/rocm
# Only the specific MIOpen files, not the whole /opt/rocm tree: miopen-builder
# is an independent stage (doesn't chain from rocblas-export/migraphx-export),
# so its own /opt/rocm never received the gfx803 rocBLAS/MIGraphX builds --
# copying it wholesale here would silently revert those back to stock.
COPY --from=miopen-export /opt/rocm/lib/libMIOpen.so.1.0.* /opt/rocm/lib/

# Same reasoning for the patched ROCR (libhsa-runtime64 with the gfx803
# VA-reuse defer, see the rocr-builder stage): copy only the libhsa-runtime64
# files. `make install` in rocr-builder already repointed the
# libhsa-runtime64.so.1 soname symlink at the new .so.1.15.0, and COPY
# preserves that.
COPY --from=rocr-export /opt/rocm/lib/libhsa-runtime64.so* /opt/rocm/lib/

# rocm/dev-ubuntu-* doesn't register /opt/rocm/lib with the dynamic linker, so
# libs are present on disk but unresolvable at runtime without this.
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev libopenblas0 libomp5 ffmpeg libsndfile1 locales \
    && rm -rf /var/lib/apt/lists/*

# ORT's StringNormalizer op constructs a std::locale by name (e.g.
# "en_US.UTF-8") at runtime; Ubuntu's base images ship no locale data at all,
# so that construction throws and the op fails for every caller, not just the
# test suite. locale-gen writes the actual locale archive; update-locale sets
# the default so it applies without callers having to pass a locale
# explicitly.
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Polaris runtime environment. These are the difference between "the card is
# enumerated" and "the stack actually dispatches to it":
#
#   HSA_OVERRIDE_GFX_VERSION=8.0.3 -- a real gfx803 card doesn't strictly need
#     this, but the Polaris family also reports gfx800/gfx802/gfx805 on some
#     boards, and the override collapses them onto the one ISA everything here
#     was compiled for.
#   ROC_ENABLE_PRE_VEGA=1 -- HIP/OpenCL refuse pre-Vega devices without it.
#   TORCH_BLAS_PREFER_HIPBLASLT=0 -- torch links hipBLASLt unconditionally and
#     hipBLASLt has no gfx803 kernels; without this, GEMMs dispatch into a
#     library that can only fail. Forces the rocBLAS path rebuilt above.
ENV HSA_OVERRIDE_GFX_VERSION=8.0.3
ENV ROC_ENABLE_PRE_VEGA=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=0

# rocBLAS/Tensile's SGEMM kernels are unreliable on gfx803 (see
# patches/rocblas/sgemm-shim/gfx803_sgemm.h) -- this routes the standard-algo
# f32 GEMM path through a correctness-verified replacement instead. Set
# GFX803_SGEMM_SHIM_DISABLE=1 at runtime to fall back to stock rocBLAS for
# A/B testing.
ENV LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim.so

# MIOpen's grouped-conv OOB fix is a source patch (miopen-builder, above,
# via patches/miopen/conv-direct-fwd-grouped-oob.sh) rather than an env var:
# an earlier, blunter fix (MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD=0) disabled the
# whole solver rather than just the broken grouped-conv case, which also
# risked pushing unrelated non-grouped shapes onto a *different* MIOpen
# fallback path with its own unverified correctness. See KERNEL_BUGS.md for
# both the root-cause investigation and why the source patch replaced the
# env var.

ENV PYTHONPATH=/opt/rocm/lib
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV

COPY --from=ort-export /onnxruntime/dist/*.whl /tmp/ort/
COPY --from=pytorch-export /wheels/*.whl /tmp/torch/
RUN "$VIRTUAL_ENV/bin/pip" install --no-cache-dir numpy /tmp/ort/*.whl /tmp/torch/*.whl \
    && rm -rf /tmp/ort /tmp/torch \
    && "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torchvision; print('torchvision', torchvision.__version__)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torchaudio; print('torchaudio', torchaudio.__version__)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import migraphx; print('migraphx python module OK')"

ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
