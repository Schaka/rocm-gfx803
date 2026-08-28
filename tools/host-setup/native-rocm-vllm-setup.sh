#!/usr/bin/env bash
# Install a working ROCm+PyTorch+vLLM stack directly on the gfx803 box's
# host OS, copied from an already-working container (ct-old by default),
# so real-hardware debugging (gdb, kprobes, ftrace, /sys/kernel/debug/...)
# no longer needs `docker exec` wrapping or PID-namespace translation.
#
# WHY this exists: this box is dedicated gfx803 test hardware, not a
# shared multi-tenant machine, so there's no isolation benefit to running
# the actual workload in a container here -- only cost (every debugging
# session needs docker exec, gdb/ftrace/kprobe PIDs don't match host PIDs,
# and container filesystem churn from repeated testing is harder to reason
# about than host state). Session found and worked around three
# container-relative-path traps doing this the first time; this script
# captures the fixes so they don't need rediscovering:
#   1. /opt/rocm's bin/lib/include/etc are symlinks through
#      /etc/alternatives/rocm-* -- those alternatives entries live outside
#      /opt/rocm itself and must be created separately on the host.
#   2. The venv's python binary is a symlink to a uv-managed interpreter
#      under /root/.local/share/uv/python/... -- /root is mode 700, so a
#      non-root user can never traverse into it even after chown'ing a
#      subdirectory inside. Relocated under /opt instead of touching
#      /root's own permissions.
#   3. The container's base image has libopenblas.so.0 installed via apt;
#      the host's distro packaging doesn't provide it under the same name
#      by default.
#
# Usage: sudo bash native-rocm-vllm-setup.sh [container-name]
# (default container name: ct-old)
set -euo pipefail

CONTAINER="${1:-ct-old}"
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

if [ "$(id -u)" -ne 0 ]; then
	echo "Run with sudo -- needs to create /opt/rocm, /opt/venv, /opt/uv-python" >&2
	exit 1
fi

echo "==> Creating /opt/rocm, /opt/venv, /opt/uv-python owned by $REAL_USER"
mkdir -p /opt/rocm /opt/venv /opt/uv-python
chown "$REAL_USER:$REAL_USER" /opt/rocm /opt/venv /opt/uv-python

echo "==> Copying $CONTAINER:/opt/rocm -> /opt/rocm (this is ~20GB, takes a couple minutes)"
sudo -u "$REAL_USER" docker cp "$CONTAINER:/opt/rocm/." /opt/rocm/

echo "==> Copying $CONTAINER:/opt/venv -> /opt/venv (~5GB)"
sudo -u "$REAL_USER" docker cp "$CONTAINER:/opt/venv/." /opt/venv/

echo "==> Copying $CONTAINER:/root/.local/share/uv/python -> /opt/uv-python"
sudo -u "$REAL_USER" bash -c "mkdir -p /tmp/uv-python-copy && docker cp '$CONTAINER:/root/.local/share/uv/python/.' /tmp/uv-python-copy/"
cp -a /tmp/uv-python-copy/. /opt/uv-python/
chown -R "$REAL_USER:$REAL_USER" /opt/uv-python
rm -rf /tmp/uv-python-copy

# The copied tree's own "cpython-3.12-..." entry is a symlink back to
# /root/... (its original absolute path in the container) -- repoint it
# to the sibling versioned directory that's now actually present here.
UV_PY_LINK=$(find /opt/uv-python -maxdepth 1 -type l -name 'cpython-3.12-*')
UV_PY_REAL=$(find /opt/uv-python -maxdepth 1 -type d -name 'cpython-3.12.*' -printf '%f\n' | head -1)
if [ -n "$UV_PY_LINK" ] && [ -n "$UV_PY_REAL" ]; then
	ln -sf "$UV_PY_REAL" "$UV_PY_LINK"
fi

echo "==> Repointing /opt/venv/bin/python at the relocated interpreter"
VENV_PY_REAL=$(find /opt/uv-python -mindepth 2 -maxdepth 2 -type f -path '*/bin/python3.12')
rm -f /opt/venv/bin/python
sudo -u "$REAL_USER" ln -s "$VENV_PY_REAL" /opt/venv/bin/python

echo "==> Setting up /etc/alternatives/rocm-* to match the container's layout"
ROCM_CORE=$(find /opt/rocm -maxdepth 1 -type d -name 'core-*' -printf '%f\n' | head -1)
if [ -z "$ROCM_CORE" ]; then
	echo "Couldn't find /opt/rocm/core-* -- check the copy succeeded" >&2
	exit 1
fi
for n in bin include lib libexec share amdgcn llvm; do
	ln -sf "/opt/rocm/$ROCM_CORE/$n" "/etc/alternatives/rocm-$n"
done
ln -sf "/opt/rocm/$ROCM_CORE" /etc/alternatives/core
ln -sf "/opt/rocm/$ROCM_CORE" /etc/alternatives/core-7

echo "==> Installing libopenblas (needed by the vendored torch build)"
dnf install -y --setopt=strict=0 openblas-serial.x86_64 openblas-openmp.x86_64 >/dev/null

ENV_SCRIPT="$REAL_HOME/gfx803-env.sh"
cat > "$ENV_SCRIPT" << EOF
# source this before running anything from /opt/venv against the native
# gfx803 ROCm install (see tools/host-setup/native-rocm-vllm-setup.sh)
export LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim.so
export LD_LIBRARY_PATH=/opt/rocm/lib
export PYTHONPATH=/opt/rocm/lib
export TORCH_BLAS_PREFER_HIPBLASLT=0
export ROCM_PATH=/opt/rocm
export PATH=/opt/venv/bin:/opt/rocm/bin:\$PATH
EOF
chown "$REAL_USER:$REAL_USER" "$ENV_SCRIPT"

echo "==> Done. Verify with:"
echo "    source $ENV_SCRIPT && rocminfo | grep gfx803"
echo "    source $ENV_SCRIPT && python3 -c 'import torch, vllm; print(torch.cuda.is_available())'"
