# Build ROCr-Runtime + libhsakmt at the gfx803 7.14 pin with the box's full
# current patch chain (doorbell, va-reuse-defer, graph-replay-batch-chunk-
# deadlock -- same set the main Dockerfile applies, in the same order) PLUS
# sdma-doorbell-missing-sfence on top, to test that one new patch in
# isolation against an otherwise-identical runtime to what's already live on
# the box. Ships libhsa-runtime64.so* + libhsakmt.so* for swap-in on the box.
#
# Build: docker build -f Dockerfile -t rocr-novad:latest .
# Then:  docker run --rm rocr-novad:latest tar -C /out -cf - . > libhsa-novad.tar
# Ship the tar to the gfx803 box, extract over /opt/rocm/lib/, and re-run
# the Qwen3.5-2B repro. The bug is intermittent, so a single clean run does
# not confirm the fix -- repeat several times before trusting it.
#
# NOT a general-purpose ROCm rebuild: only the two runtime libs are produced.
FROM rocm/dev-ubuntu-26.04:7.14.0-full

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config git \
        libnuma-dev libdrm-dev libelf-dev xxd \
        libgl1-mesa-dev libx11-dev mesa-common-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && (pip3 install --break-system-packages CppHeaderParser || pip3 install CppHeaderParser)

COPY rocm-systems-src /rocm-systems-src

# Each patch step is idempotent: this local rocm-systems-src is a reused dev
# checkout, not a pristine clone freshly extracted per build, so it may
# already carry some of these from earlier work -- `git apply` errors out
# (rather than no-op) on an already-applied hunk, so check the marker first.

# Doorbell patch: required for gfx803 device enumeration (the box's stock
# build applies it; leaving it out makes rocminfo report no GPU).
COPY hsa-agent-rejects-legacy-doorbell.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "doorbell_type_ == 0" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    || git apply /tmp/hsa-agent-rejects-legacy-doorbell.patch; } \
    && grep -q "doorbell_type_ == 0" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    && echo DOORBELL_APPLIED

COPY va-reuse-defer.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "GFX803_VA_REUSE_DEFER_PATCH" projects/rocr-runtime/libhsakmt/src/fmm.c \
    || git apply /tmp/va-reuse-defer.patch; } \
    && grep -q "GFX803_VA_REUSE_DEFER_PATCH" projects/rocr-runtime/libhsakmt/src/fmm.c \
    && echo VA_REUSE_DEFER_APPLIED

COPY graph-replay-batch-chunk-deadlock.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "std::min<size_t>(DEBUG_HIP_GRAPH_BATCH_SIZE, sw_queue_size)" projects/clr/rocclr/device/rocm/rocvirtual.cpp \
    || git apply /tmp/graph-replay-batch-chunk-deadlock.patch; } \
    && grep -q "std::min<size_t>(DEBUG_HIP_GRAPH_BATCH_SIZE, sw_queue_size)" projects/clr/rocclr/device/rocm/rocvirtual.cpp \
    && echo GRAPH_REPLAY_CHUNK_CLAMP_APPLIED

COPY sdma-doorbell-missing-sfence.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "_mm_sfence();" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp \
    || git apply /tmp/sdma-doorbell-missing-sfence.patch; } \
    && grep -q "_mm_sfence();" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp \
    && echo SDMA_SFENCE_APPLIED

# Defensive mitigation, not a root-cause fix. Root cause: gfx7/8's IH
# payload is unreliable and undocumented at the software level; on a
# system with a correctly-sized PCIe BAR (Resizable BAR / Above 4G
# Decoding enabled -- see README's hardware requirements), code-object
# loads route around the erratum-prone GPU-dispatch copy path entirely
# and this is rarely if ever needed; on a small-BAR system, or if it
# still shows up, opt in below. Off by default, matching pristine
# upstream behavior --
# set ROCR_GFX8_EOP_MITIGATION=1 (+ optionally
# ROCR_GFX8_EOP_MITIGATION_TIMEOUT_US, default 1000 = 1ms) at runtime to
# opt in on a system that still sees this after fixing its BAR size. Two
# separate completion-wait sites can hang forever on the same class of
# lost GPU completion notification:
#  - BlitKernel::SubmitLinearCopyCommand's AQL dispatch signal (hit
#    whenever code-object loads route through the compute-dispatch copy
#    path, e.g. on a small-BAR system) -- bounded retry: same signal, then
#    fresh signal same queue, then a brand new queue.
#  - AqlQueue::ExecutePM4's gfx8 read-index spin (hit by
#    InvalidateCodeCaches, called after EVERY code-object load regardless
#    of BAR size -- previously unreachable/unobserved because a failed
#    BlitKernel copy always aborted before reaching it) -- bounded retry
#    on a fresh queue slot, same pattern.
COPY blit-kernel-eop-interrupt-retry.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "AqlQueue::ExecutePM4.*packet did not complete" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    || git apply /tmp/blit-kernel-eop-interrupt-retry.patch; } \
    && grep -q "BlitKernel::RecreateQueue" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_kernel.cpp \
    && grep -q "Gfx8EopMitigationMaxAttempts" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    && echo BLIT_KERNEL_EOP_RETRY_APPLIED

# Third completion-wait site, same erratum: rocclr's shared WaitForSignal()
# (device/rocm/rocvirtual.hpp) backs HIP memcpy (hipMemcpyWithStream et
# al.) and other ROCclr dispatch waits -- unreachable/unobserved as long
# as a preceding BlitKernel copy aborted first, so this only showed up
# once the BAR fix routed copies around that path. Same opt-in
# ROCR_GFX8_EOP_MITIGATION master switch, own
# ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US (default 5000us) since this
# function has no size/duration context to auto-pick a safe default --
# tune it above your workload's genuinely slowest dispatch through this
# path.
COPY rocclr-eop-wait-mitigation.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "Gfx8EopMitigationHipTimeoutUs" projects/clr/rocclr/device/rocm/rocvirtual.hpp \
    || git apply /tmp/rocclr-eop-wait-mitigation.patch; } \
    && grep -q "Gfx8EopMitigationHipTimeoutUs" projects/clr/rocclr/device/rocm/rocvirtual.hpp \
    && echo ROCCLR_EOP_WAIT_MITIGATION_APPLIED

WORKDIR /rocm-systems-src/projects/rocr-runtime
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" \
    && make install

# CLR (HIP only, matches the main Dockerfile's rocr-clr-builder stage) --
# links against the just-installed patched rocr-runtime.
WORKDIR /rocm-systems-src/projects/clr
RUN mkdir -p build && cd build \
    && cmake .. \
        -DHIP_COMMON_DIR=/rocm-systems-src/projects/hip \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLR_BUILD_HIP=ON \
        -DCLR_BUILD_OCL=OFF \
        -DHIP_PLATFORM=amd \
    && make -j"$(nproc)" \
    && make install

RUN mkdir -p /out \
    && cp -L /opt/rocm/lib/libhsa-runtime64.so* /out/ \
    && cp -L /opt/rocm/lib/libhsakmt.so* /out/ 2>/dev/null || true \
    && cp -L /opt/rocm/lib/libamdhip64.so* /out/ 2>/dev/null || true \
    && ls -la /out/

CMD ["true"]