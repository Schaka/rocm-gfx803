# syntax=docker/dockerfile:1
#
# torchvision built from source against the torch wheel in the pytorch image.
#
# Its own target and its own image, rather than a step inside pytorch, so that it
# gets its own runner and its own time budget in CI.
#
# Named "builder" for the same reason as pytorch.Dockerfile's own stage of that
# name: final takes only /wheels/*.whl (see the "wheels" stage below), everything
# built to reach it does not need to leave this stage's own published image.
FROM pytorch AS builder

ARG ROCM_ARCH
ARG TORCHVISION_REF
ARG TORCHVISION_SHA
ARG GFX803_LINE

RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev libpng-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=scripts/git-pin.sh,target=/git-pin \
    /git-pin /vision https://github.com/pytorch/vision.git "${TORCHVISION_REF}" "${TORCHVISION_SHA}" \
    && git -C /vision submodule sync --recursive \
    && git -C /vision submodule update --init --recursive --depth 1 --jobs 4

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-torchvision \
    --mount=type=bind,source=scripts/build/torch-companion.sh,target=/scripts/build/torch-companion.sh \
    /scripts/build/torch-companion.sh /vision

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" torchvision "${GFX803_SOURCE_REV}" "${GFX803_PINS}"

# See pytorch.Dockerfile's own "wheels" stage for why this split exists.
FROM scratch AS wheels
COPY --from=builder /wheels /wheels
