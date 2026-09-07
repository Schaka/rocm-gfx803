#!/bin/sh
# Install the ORT and torch wheels into the runtime venv, prove each one imports,
# and write the constraints file that keeps them installed.
set -eu

VENV="${VIRTUAL_ENV:?VIRTUAL_ENV is required}"

# One wheel per distribution, newest mtime wins, rather than a glob. All three
# torch stage images drop into the same directory, and a published stage image
# can carry a wheel from before its own most recent rebuild. A glob then hands
# pip two conflicting versions of the same package, and the resulting
# ResolutionImpossible message names neither file.
picked="$(for w in /tmp/ort/*.whl /tmp/torch/*.whl /tmp/triton/*.whl; do
              [ -e "$w" ] || continue
              printf '%s\t%s\t%s\n' "$(stat -c %Y "$w")" \
                  "$(basename "$w" | sed -E 's/-.*//' | tr 'A-Z.' 'a-z_')" "$w"
          done | sort -k1,1nr | awk -F'\t' '!seen[$2]++ {print $3}')"
echo "wheels to install:"
echo "$picked" | sed 's/^/  /'

# shellcheck disable=SC2086  # $picked is a newline-separated list of paths.
"$VENV/bin/pip" install --no-cache-dir numpy $picked
rm -rf /tmp/ort /tmp/torch /tmp/triton

"$VENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p"
"$VENV/bin/python3" -c "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip)"
"$VENV/bin/python3" -c "import torchvision; print('torchvision', torchvision.__version__)"
"$VENV/bin/python3" -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
"$VENV/bin/python3" -c "import migraphx; print('migraphx python module OK')"
# Import only, not a compile: this build has no GPU, so a real JIT compile is
# the on-hardware check (tools/correctness-suite/), not this gate. Confirms the
# wheel installs cleanly and torch can see it, which is everything a GPU-less
# build can prove.
"$VENV/bin/python3" -c "import triton; from torch.utils._triton import has_triton_package; print('triton', triton.__version__, 'has_triton_package:', has_triton_package())"

# This image is a base for downstream Dockerfiles, and any of them can pip
# install something that pulls torch in as a transitive dependency. pip then
# silently swaps this image's ROCm gfx803 build for a generic PyPI wheel. Pinning
# the exact installed version turns that into a loud resolution failure, because
# no index can satisfy the pin and only the existing install can.
#
# importlib.metadata, not `pip freeze`: these wheels were installed from a local
# path, which pip freeze renders as `name @ file:///...` rather than
# `name==version`.
#
# The distribution name is onnxruntime-migraphx. A downstream package that pulls
# in plain `onnxruntime` is not covered by this file.
"$VENV/bin/python3" -c "import importlib.metadata as m; [print(f'{p}=={m.version(p)}') for p in ('torch', 'torchvision', 'torchaudio', 'onnxruntime-migraphx', 'triton')]" > /opt/pip-constraints.txt
cat /opt/pip-constraints.txt
