# Build ROCr-Runtime + libhsakmt at the gfx803 7.14 pin with the box's full
# current patch chain, in the same order the main Dockerfile applies it, so
# a single new patch can be tested against an otherwise-identical runtime to
# what is already live on the box. Ships libhsa-runtime64.so* +
# libhsakmt.so* for swap-in on the box.
#
# Currently on top: aql-ring-queue-full-workaround, the GFXIP 7/8
# double-mapped AQL ring. Note it needs a kernel WITHOUT
# REFERENCE-amdkfd-gfx7-8-queue-size-writeback -- that patch forces the HQD
# back to the single ring size and cancels this one out.
#
# Build: docker build -f Dockerfile -t rocr-novad:latest .
# Then:  docker run --rm rocr-novad:latest tar -C /out -cf - . > libhsa-novad.tar
# Ship the tar to the gfx803 box and extract over /opt/rocm/lib/ to pick up
# the queue-full fix on its own merits (real queue sizes instead of the
# 64-packet floor). It was originally built to test this patch against the
# silent dispatch hang and the Qwen3.5-2B graph-replay hang; both were later
# root-caused to VRAM-clock marginality instead (see RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md
# and README.md's "Host VBIOS setting" section) -- this patch does not fix
# either hang.
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

# The real fix for every one of those wait sites: restore the GFXIP 7/8
# double-mapped AQL ring. Without it the CP cannot tell a full ring from an
# empty one, parks, and never drains -- so the waits above are waiting on a
# packet that will never be fetched. See
# patches/rocm-systems/aql-ring-queue-full-workaround.patch.
#
# rocclr-eop-wait-mitigation.patch used to be applied here and is
# deliberately not any more. It adds ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US
# to ROCclr's WaitForSignal(), which gives up on a packet still live in a
# ring the CP has not drained; enabling it hard-locked the whole box twice
# (no SSH, no ping, physical power-cycle to recover). Not something to leave
# one env var away from a validation run on real hardware.
COPY aql-ring-queue-full-workaround.patch /tmp/
RUN cd /rocm-systems-src && { grep -q "queue_full_workaround_" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    || git apply /tmp/aql-ring-queue-full-workaround.patch; } \
    && grep -q "ui32.AQLQueueMemory" projects/rocr-runtime/runtime/hsa-runtime/core/driver/kfd/amd_kfd_driver.cpp \
    && grep -q "MemoryRegion::AllocateDoubleMap" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    && echo AQL_RING_QUEUE_FULL_WORKAROUND_APPLIED

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