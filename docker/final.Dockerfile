# syntax=docker/dockerfile:1
#
# The published gfx803 runtime image: a ROCm 10.0 tree with every gfx803 fix in
# it, plus a Python 3.12 venv holding torch, torchvision, torchaudio and ORT.
#
# This target compiles nothing. Every component arrives as a prebuilt image, so
# the only failure mode left is a wrongly wired input, and scripts/build/
# final-rocm.sh checks each one by name.
FROM python-base

ARG GFX803_LINE

COPY --from=migraphx /opt/rocm /opt/rocm
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

# migraphx already carries the same MIOpen and rocSOLVER files. Copying them
# again from their own images is what makes those inputs independently
# selectable, and it is the copy the checks below are written against.
#
# The source and destination are /opt/rocm/core-10.0/lib, not /opt/rocm/lib,
# because /opt/rocm/lib is a symlink to /etc/alternatives/rocm-lib and a COPY
# with a wildcard source does not follow a symlinked directory in the middle of
# the path. It silently matches zero files instead of failing. A ROCm version
# bump moves this path.
COPY --from=miopen /opt/rocm/core-10.0/lib/libMIOpen.so.* /opt/rocm/core-10.0/lib/
COPY --from=rocsolver /opt/rocm/core-10.0/lib/librocsolver.so.* /opt/rocm/core-10.0/lib/

# The untouched base image's MIOpen, kept only as a size reference for the check.
COPY --from=python-base /opt/rocm/core-10.0/lib/libMIOpen.so.* /tmp/miopen-stock-ref/

COPY --from=sgemm-shim /opt/rocm/lib/libgfx803_sgemm_shim.so /opt/rocm/lib/
COPY --from=rocr-clr /opt/rocm/lib /opt/rocm-clr-lib/

RUN --mount=type=bind,source=scripts/build/final-rocm.sh,target=/scripts/build/final-rocm.sh \
    /scripts/build/final-rocm.sh

# libdrm2 and libdrm-amdgpu1: every ROCm library built here links the versioned
# libdrm.so.2 and libdrm_amdgpu.so.1 SONAMEs, and this base image bundles only
# unversioned copies. Each build stage starts fresh, and only /opt/rocm crosses
# between them, so the runtime packages have to be installed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev libopenblas0 ffmpeg libsndfile1 locales \
        libdrm2 libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# The Polaris runtime environment. HSA_OVERRIDE_GFX_VERSION and
# ROC_ENABLE_PRE_VEGA make the stack accept the card; hipBLASLt has never had
# gfx8 kernels, so torch must not prefer it.
ENV HSA_OVERRIDE_GFX_VERSION=8.0.3
ENV ROC_ENABLE_PRE_VEGA=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=0
ENV LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim.so

ENV UV_NO_CACHE=1
ENV PYTHONPATH=/opt/rocm/lib
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV --python 3.12 --seed

COPY --from=ort /onnxruntime/dist/*.whl /tmp/ort/
COPY --from=pytorch /wheels/*.whl /tmp/torch/
COPY --from=torchvision /wheels/*.whl /tmp/torch/
COPY --from=torchaudio /wheels/*.whl /tmp/torch/

RUN --mount=type=bind,source=scripts/build/final-wheels.sh,target=/scripts/build/final-wheels.sh \
    /scripts/build/final-wheels.sh
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt
ENV UV_CONSTRAINT=/opt/pip-constraints.txt
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" final "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
