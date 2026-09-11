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

## Using the build inside the final CI image

The steps above build vLLM against the box's own ROCm and PyTorch
install. `native-rocm-vllm-setup.sh` copied that install from an older
container line, 7.14. This repo's `docker/final.Dockerfile`
builds a separate, newer image (the 10.0 line) that does not contain
vLLM at all yet. Wiring vLLM into that Dockerfile is still open work
(see `NOTES.md`, open items). Until that happens, use the box build
inside a container from the final image like this:

1. Start a container from the final image, with the GPU passed through
   and the box's vLLM folder mounted in:

   ```
   docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
       -v /data/vllm-mobydick:/data/vllm-mobydick \
       <final-image-tag> bash
   ```

2. Inside the container, repeat Step 3 and Step 4 above, using the
   container's own `/opt/venv` and ROCm install. Do not skip this and
   try to import the box's already-built copy directly. The compiled
   C++/HIP extension and the three kernel `.so` files are linked
   against one specific PyTorch and ROCm build. The final image's
   10.0 line is a different build than the box's current 7.14-based
   install. A `.so` built on the box is not guaranteed to load inside
   the final image's container.

3. Before you trust the result, run Step 5's sanity check inside the
   container.

Nobody ran this container path on real hardware yet. Treat it as the
best known method, not a checked one. Someone must run it on the box
from start to end and record the result here.
