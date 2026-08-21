# Build ROCr-Runtime + libhsakmt at the gfx803 7.14 pin WITHOUT the
# va-reuse-defer patch, to A/B test whether that patch causes the llama.cpp
# HIP-graph launch hang on Polaris. The doorbell patch IS applied (it is
# required for Polaris device enumeration — a build without it can't see the
# GPU). Ships libhsa-runtime64.so* + libhsakmt.so* for swap-in on the box.
#
# Build: docker build -f Dockerfile -t rocr-novad:latest .
# Then:  docker run --rm rocr-novad:latest tar -C /out -cf - . > libhsa-novad.tar
# Ship the tar to the gfx803 box, extract over /opt/rocm/lib/, and re-run the
# llama.cpp HIP-graph bench. Compare vs the stock (va-reuse-defer) runtime.
#
# NOT a general-purpose ROCm rebuild: only the two runtime libs are produced.
FROM rocm/dev-ubuntu-26.04:7.14.0-full

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential pkg-config git \
        libnuma-dev libdrm-dev libelf-dev xxd \
    && rm -rf /var/lib/apt/lists/*

COPY rocm-systems-src /rocm-systems-src

# Doorbell patch: required for gfx803 device enumeration (the box's stock
# build applies it; leaving it out makes rocminfo report no GPU).
COPY hsa-agent-rejects-legacy-doorbell.patch /tmp/
RUN cd /rocm-systems-src && git apply /tmp/hsa-agent-rejects-legacy-doorbell.patch \
    && grep -q "doorbell_type_ == 0" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp \
    && echo DOORBELL_APPLIED

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