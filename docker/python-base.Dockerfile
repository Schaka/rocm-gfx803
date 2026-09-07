# syntax=docker/dockerfile:1
#
# Starting point for every other target: the ROCm base image plus a uv-managed
# Python 3.12.
#
# Ubuntu 26.04's own python3 is 3.14, and numpy and onnx dependency resolution
# needs 3.12. Several tools in this build also break outright on 3.14.
ARG BASE_IMAGE=rocm/dev-ubuntu-26.04:10.0.0-full
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore

# This base image's cmake is 4.x, which hard-errors on any
# cmake_minimum_required() below 3.5 instead of warning. Several dependencies
# fetched during this build still declare one. Set as an environment variable
# rather than a flag, because most of the places this bites are nested cmake
# calls with no way to pass extra flags through.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-dev python3-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12
