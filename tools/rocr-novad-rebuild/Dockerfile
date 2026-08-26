# Build ROCr-Runtime + libhsakmt at the gfx803 7.14 pin with the box's full
# current patch chain (doorbell, va-reuse-defer, graph-replay-batch-chunk-
# deadlock -- same set the main Dockerfile applies, in the same order) PLUS
# sdma-doorbell-missing-sfence on top, to test that one new patch in
# isolation against an otherwise-identical runtime to what's already live on
# the box. Ships libhsa-runtime64.so* + libhsakmt.so* for swap-in on the box.
#
# Build: docker build -f Dockerfile -t rocr-novad:latest .
# Then:  docker run --rm rocr-novad:latest tar -C /out -cf - . > libhsa-novad.tar
# Ship the tar to the gfx803 box, extract over /opt/rocm/lib/ (inside the
# ct-old container, not the host -- see AGENTS.md/SESSION_HANDOFF for why
# host rocminfo is irrelevant here), and re-run the Qwen3.5-2B repro. The
# bug is intermittent, so a single clean run does not confirm the fix --
# repeat several times before trusting it.
#
# NOT a general-purpose ROCm rebuild: only the two runtime libs are produced.
FROM rocm/dev-ubuntu-26.04:7.14.0-full

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config git \
        libnuma-dev libdrm-dev libelf-dev xxd \
    && rm -rf /var/lib/apt/lists/*

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

WORKDIR /rocm-systems-src/projects/rocr-runtime
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" \
    && make install

RUN mkdir -p /out \
    && cp -L /opt/rocm/lib/libhsa-runtime64.so* /out/ \
    && cp -L /opt/rocm/lib/libhsakmt.so* /out/ 2>/dev/null || true \
    && ls -la /out/

CMD ["true"]