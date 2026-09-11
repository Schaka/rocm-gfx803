# Building vLLM for gfx803

This guide explains how to build the gfx803 (Polaris) port of vLLM that
lives in this folder. This build ran on real hardware before. Follow
the steps below to repeat it.

vLLM here is vendored, not patched. This folder is a full copy of the
vLLM source with the gfx803 port already applied. You do not write a
patch file. You install this folder as a Python package and compile
three small kernel files.

If you have not read `NOTES.md`, read it first. It records that this
vLLM path is not usable end to end yet. The cause is an open hardware
bug, the gfx7/8 EOP-interrupt-loss erratum. The build steps below
still work. Only long, uninterrupted runs hit that separate bug.

## Where you build

Build on the gfx803 box (`192.168.1.214`). The box already has a
working ROCm and PyTorch install, set up by
`tools/host-setup/native-rocm-vllm-setup.sh`. That script gives you a
Python virtual environment at `/opt/venv` and a ROCm install at
`/opt/rocm`. This guide assumes that install already exists.

## Step 1: Load the environment

Every command below needs the settings in `tools/vllm-bench/env.sh`.
Load them into your shell first.

```
source /path/to/rocm-gfx803/tools/vllm-bench/env.sh
```

This sets `LD_PRELOAD` to a shim that rocBLAS needs on gfx803. Without
that shim, some GEMM calls return wrong numbers with no error. It also
points `PYTHONPATH` at the ROCm build of `amdsmi`. It moves build
caches to `/data`, so a large build does not fill the box's root disk.

## Step 2: Copy this folder onto the box

Copy this whole `vllm/` folder to the box, for example to
`/data/vllm-mobydick`. If you are already working on the box, skip this
step.

## Step 3: Install the vLLM Python package

From inside the copied folder, run:

```
cd /data/vllm-mobydick
pip3 install --no-build-isolation --no-deps -e .
```

This is an editable install. `pip3` builds the C++ and HIP extension
once and links it against the box's PyTorch build. It then points the
`vllm` Python package straight at this source folder. If you edit a
`.py` file afterward, you do not need to run this command again. If you
edit a `.cu`/`.cuh`/`.h`/`CMakeLists.txt` file, run it again.

## Step 4: Compile the three hand-written gfx803 kernels

The port adds three kernels that are not part of the C++ extension
above. Each one is a small HIP source file that you compile straight
to a `.so` file. Each `.so` file must land next to the Python file that
loads it. That Python file finds its `.so` using its own file path.

If you followed Step 2 and 3 as shown, the installed folder is
`/data/vllm-mobydick`. Run all three commands from inside it:

```
cd vllm/model_executor/layers
hipcc --offload-arch=gfx803 -O3 -shared -fPIC \
  -o libgfx803gemm.so ../../gfx803_kernels/gfx803_gemm_lib.hip

hipcc --offload-arch=gfx803 -O3 -shared -fPIC \
  -o libgfx803gemv_m.so ../../gfx803_kernels/gfx803_gemv_m.hip

cd ../../v1/attention/ops
hipcc --offload-arch=gfx803 -O3 -shared -fPIC \
  -o libgfx803attn.so ../../../gfx803_kernels/gfx803_attn_split.hip
```

Run these three commands again whenever you change a `.hip` file under
`vllm/gfx803_kernels/`. You do not need to repeat Step 3 for a kernel
change alone.

## Step 5: Check the build

Run this sanity check. It loads a small model and generates a few
tokens. It does not check speed, only that the build produces coherent
text.

```
cd /data/vllm-mobydick && /opt/venv/bin/python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/data/qwen2.5-1.5b', gpu_memory_utilization=0.7,
          max_model_len=1024, dtype='float16',
          compilation_config={'cudagraph_capture_sizes': [1]})
for p in ['The capital of France is', '2 + 2 =']:
    out = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=32))
    print(repr(out[0].outputs[0].text))
"
```

Use `dtype='float16'` always. Do not use `dtype='bfloat16'` on gfx803.
This hardware has no native bf16 support, and the first bf16 kernel
call hangs.

If you changed a kernel, also run the correctness probes under
`tools/vllm-bench/` (`probe_gemv_m.py`, `verify_gemv_m.py`) before you
trust any speed number. `NOTES.md` explains what each one checks and
why a speed-only test can miss a wrong answer.

## Building vLLM in CI

CI builds vLLM the same way it builds PyTorch and ONNX Runtime. It runs
as its own bake target, on its own runner, from the vendored source in
this folder. `docker-bake.hcl` defines the `vllm` target and its
trimmed `vllm-wheels` companion. `.github/workflows/build-pipeline.yml`
runs it as a job named `vllm`. That job needs only the `pytorch` job.
It starts as soon as PyTorch publishes. It then runs in parallel with
torchvision, torchaudio, and ONNX Runtime, instead of waiting behind
them.

The CI build compiles the same three kernels as Step 4 above, with the
same `hipcc` commands. It builds them against the same 10.0-line ROCm
and PyTorch that `docker/final.Dockerfile` assembles.
`docker/final.Dockerfile` then installs the resulting wheel. It also
copies the three compiled kernels into place on its own. A manual run
of the "Build gfx803 (ROCm 10.0)" workflow builds and wires in vLLM by
default. Uncheck "Build the vendored gfx803 vLLM fork" to skip it and
reuse the last published `vllm` image instead.

Nobody ran the resulting final image on real hardware yet. The image
builds and `import vllm` succeeds in CI, and that is a build check
only. Someone must run the checks below on the box and record the
result here before this counts as verified.

## Using vLLM in the final image

The final image now installs vLLM and its three compiled kernels by
default. `LD_PRELOAD` and `PYTHONPATH` are already set inside the image,
to the same values `tools/vllm-bench/env.sh` sets on the box. A
container started from the final image needs no extra environment
setup to import vLLM. Start it like this, with the GPU passed through:

```
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
    <final-image-tag> bash
```

Then run Step 5's sanity check inside the container, unchanged.

Are you iterating on a kernel change and want to test it before it
lands in CI? Keep using the box-native build in Steps 1 through 5. You
can also mount your working copy of this folder over the image's
installed package, for a quick check:

```
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /data/vllm-mobydick/vllm/model_executor:/opt/venv/lib/python3.12/site-packages/vllm/model_executor \
    <final-image-tag> bash
```

Use this mount only to preview a change. Commit the real fix to the
files under `vllm/vllm/` in this repo, so the next CI build picks it
up.
