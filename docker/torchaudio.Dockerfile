# syntax=docker/dockerfile:1
#
# torchaudio built from source against the torch wheel in the pytorch image.
#
# ROCm/audio, not upstream pytorch/audio: only audio has a ROCm specific fork.
# Its own target and its own image for the same reason as torchvision.
FROM pytorch

ARG ROCM_ARCH
ARG TORCHAUDIO_REF
ARG TORCHAUDIO_SHA
ARG GFX803_LINE

RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line verify /opt/rocm "${GFX803_LINE}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=scripts/git-pin.sh,target=/git-pin \
    /git-pin /audio https://github.com/ROCm/audio.git "${TORCHAUDIO_REF}" "${TORCHAUDIO_SHA}" \
    && git -C /audio submodule sync --recursive \
    && git -C /audio submodule update --init --recursive --depth 1 --jobs 4

RUN --mount=type=cache,target=/root/.ccache,id=gfx803-rocm10-torchaudio \
    --mount=type=bind,source=scripts/build/torch-companion.sh,target=/scripts/build/torch-companion.sh \
    USE_FFMPEG=1 /scripts/build/torch-companion.sh /audio

ARG GFX803_SOURCE_REV GFX803_PINS
RUN --mount=type=bind,source=scripts/gfx803-line.sh,target=/gfx803-line \
    /gfx803-line stamp /opt/rocm "${GFX803_LINE}" torchaudio "${GFX803_SOURCE_REV}" "${GFX803_PINS}"
