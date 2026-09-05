# rocm-gfx803

This repo keeps AMD Polaris (gfx803: RX 460/470/480/560/570/580/590 and close
relatives) working on the MIGraphX + ONNX Runtime + PyTorch stack. It was split
out of [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) into its own
repository.

## Prebuilt images: do not build this yourself

CI builds the final image and pushes it to GHCR. You only need a local build
when you change a patch. Pull the image you want:

```bash
# rocm10 (main line, TheRock 10.0), versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm10.0-gfx803

# rocm10, always the newest successful rocm10 build
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803

# rocm7.14 (older, hardware-verified line), versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx803

# rocm6.4.4 (older, hardware-verified line), versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm6.4.4-gfx803
```

The section "Repository layout" and `.github/workflows/gfx803-component.yml`
give the whole tag scheme: per-component images, cache tags, and dated tags.

## Why a separate repo

AMD stopped building gfx803 support after ROCm 6.0. ROCm 7 and newer reject the
card outright when HSA creates the agent. Every part that makes gfx803 work is a
local patch here. One patch restores the legacy doorbell, which ROCm 7 needs
just to start a kernel. A larger set of Tensile, MIOpen, and MIGraphX patches
correct bugs that only appear on this old GCN3 hardware.

That patch set changes faster than, and separately from, the mainline
nightly/release pipeline that `rocm-migraphx-ort-builder` runs for every other
architecture. Keeping it there meant that every gfx803 investigation added
noise to a repo that needs none of it. This repo now holds that investigation
and patch history.

The link between the two repos runs one way. The mainline docs point here for
gfx803. This repo does not track or copy the mainline per-architecture matrix.
The versions do follow the mainline release track. Every pinned ref here matches
what the mainline repo's `release.yml` ships for the same ROCm line: MIGraphX
`release/rocm-rel-10.0`, ORT `v1.29.0`, and PyTorch `2.14.0` /
`release/2.14`. So gfx803 does not silently lag the supported line it came from.

## Repository layout

```
rocm-gfx803/
├── Dockerfile              # ROCm 10.0 (TheRock), the line under active development
├── .dockerignore
├── patches/                # gfx803 patches for the 10.0 pin. Most came from 7.14.
│                           #   Each patch header states its own hardware-verification state.
├── scripts/                # build helpers: git-pin.sh (commit pinning), gfx803-line.sh (line provenance)
├── tools/                  # correctness-suite, host-setup, tc-staleness. Shared by all lines.
├── verify.py               # on-hardware smoke test
├── MIGRATION_NOTES.md      # investigation log for the 10.0 line
├── vllm/                   # the gfx803 vLLM hard fork, now on the 10.0 line
├── rocm7.14/               # the hardware-verified TheRock 7.14 line (archived, builds by hand)
│   ├── Dockerfile
│   ├── patches/
│   ├── verify.py
│   ├── README.md           # includes the full vLLM-on-gfx803 investigation
│   └── MIGRATION_NOTES.md
├── rocm6.4.4/              # the older hardware-verified line on classic ROCm 6.4.4 tags
│   ├── Dockerfile
│   ├── patches/
│   ├── tools/
│   ├── verify.py
│   ├── KERNEL_BUGS.md      # the original gfx803 bug-hunting method and record
│   └── wip_patches/        # rejected and superseded patch designs, kept for the record
├── llama-cpp-gfx803/       # llama.cpp gfx803 patches (arch level, shared)
├── RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md  # hardware level, shared by all lines
└── .github/workflows/      # CI builds the 10.0 line only
```

rocm10 is this repo's root and the line under active development. ROCm 10.0 is
a TheRock meta-release, not a classic per-repo tag. Every component is pinned to
a release branch instead of a frozen commit. ROCR-Runtime, rocBLAS, and MIOpen
(now in the `rocm-systems` and `rocm-libraries` monorepos) track
`release/therock-10.0`. MIGraphX tracks `release/rocm-rel-10.0`. That is the same
`release/rocm-rel-<major.minor>` convention the mainline repo uses for its own
manual releases. There is no nightly schedule and no prebuilt wheel. The build
takes the current tip of the named release branch when a person runs it.
`MIGRATION_NOTES.md` records how those refs were chosen.

`rocm7.14/` is the line this one replaced. It uses the same hand-built
pattern, it is hardware-verified, and the gfx803 vLLM fork and its investigation
started there. It is kept because it is the most recent line with a full
correctness-suite pass and long real-hardware use. rocm10 inherited its patches
and its method, but shares no files with it.

`rocm6.4.4/` is the older stable line. It uses the classic per-repo
`rocm-6.4.4` tags and the longest period of hardware verification. New
investigation does not start there.

The three lines are separate copies on purpose, not one shared asset. Each
was maintained at a different verification level. A shared file lets a bug found
while chasing 10.0 reach the hardware-tested 7.14 or 6.4.4 builds. Once
10.0 is confirmed at least as solid as 7.14 everywhere, the lines are diffed and
merged deliberately. See "Convergence" below.

## Status

- rocm10 (this repo's root): actively developed, and partly hardware-tested.
  The ROCm 10.0 stack on the test box runs rocBLAS, MIOpen, MIGraphX, PyTorch,
  ORT, and the vLLM fork on a real card. The torch correctness suite gives
  202/202 PASS with the current patch set on the box stack (see "Cross-dispatch
  coherence"). Inside the image the same suite reports 118 PASS, 0 BAD, 0 NONFINITE and
  84 ERRORs that are all `/ INDUCTOR`, because the image ships no Triton. The full image
  builds end to end, and `tools/imgvalidate.sh` now runs that whole gate against the
  image itself on the card: its `libamdhip64` and its `librocsolver.so.0` (13 of 13
  `torch.linalg` routines within 1.0e-5 of CPU) are the shipped binaries, not a
  hand-built pair swapped in. `tools/imgvalidate.sh <image-tag>` is the on-card gate
  for a whole image: it asserts the shipped libraries' markers, runs `verify.py`, the
  coherence probes with a control arm that must reproduce the corruption, the fp16 GEMM
  and convolution sweep with and without the shim's takeover, and the op suite. The last
  run of it (2026-09-05) came back with verify.py all-pass, 0 anomalies in 480
  cross-stream checks against 16 in the same-configuration control, 27/27 fp16 cases with
  the shim and 27/27 against real rocBLAS, op suite `BAD 0, NONFINITE 0`, and no GPU reset
  or ring timeout in dmesg. `MIGRATION_NOTES.md` has the details, and each patch header
  states its own verification state.
- rocm7.14: hardware-verified, and archived. CI no longer builds it.
  `rocminfo` lists the card as a real `KERNEL_DISPATCH` agent, and rocBLAS,
  MIOpen, MIGraphX, PyTorch, and ORT all do real GPU work on it. The full
  `tools/correctness-suite/` (23 MIOpen op and solver sweeps) passes clean. ORT's
  `onnx_backend_test_series.py` (3828 tests) has one open failure that is not
  gfx803-specific: `ConvTranspose`, an upstream MIGraphX bug reported upstream
  and not patched here. Real-model runs (faster-whisper/CTranslate2,
  whisper.cpp, parakeet.cpp) produce correct transcripts on real audio. The
  gfx803 vLLM fork is built and verified against this line's stack.
  `rocm7.14/README.md` has the full detail.
- rocm6.4.4: hardware-verified, the older and longest-running line, and
  archived. CI no longer builds it. See `rocm6.4.4/README.md` and
  `rocm6.4.4/KERNEL_BUGS.md`.

### Required host setup

- VBIOS and VRAM clock: this card's VRAM must run at or below its rated
  speed. The rating is 1750 MHz on the mining-tuned VBIOS that this box shipped
  with, and that limit was confirmed by core overdrive. A stock VBIOS from the
  correct vendor needs no overdrive. VRAM above spec causes real GPU VM faults in
  MIOpen (`pool_sweep`) and GPU hangs in vLLM. Both have the same hardware cause
  and neither is a software bug. See "Host VBIOS setting" below and
  `RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md`.
- `patches/rocm-systems/aql-ring-queue-full-workaround.patch` restores the AQL
  ring's double mapping for GFXIP 7 and 8. It raises the queue from 64 packets to
  131072, which is 2048 times the unpatched cap. With it,
  `graph-replay-batch-chunk-deadlock.patch` is not needed. It requires a kernel
  that does NOT carry `REFERENCE-amdkfd-gfx7-8-queue-size-writeback`. Do not
  combine it with `graph-replay-queue-size-cap.patch`.

### gfx803 test box (192.168.1.214)

- SSH user and password are both `user`, and the sudo password is `user`. There
  is no root login, so use `echo user | sudo -S <cmd>`. Put all work under
  `/data`. The host stack is 10.0: `/opt/rocm` plus `/opt/venv` (torch 2.14). The
  7.14 line is no longer on the box: `/data/rocm-7.14` and `/data/venv-7.14`
  were deleted on 2026-09-05 to reclaim space. Rebuild it from `rocm7.14/` if an
  A/B against 7.14 is needed again.
- `/opt/rocm/lib` is a symlink to `/etc/alternatives/rocm-lib`, which points at
  `/opt/rocm/core-10.0/lib`, so they are one directory. That symlink is broken
  inside a container, because `/etc/alternatives` is not part of the `/opt/rocm`
  mount. Pass the real prefix instead, for example
  `-DROCM_PATH=/opt/rocm/core-10.0`. Otherwise CMake fails to find `hip` and says
  very little about it.
- The GPU needs a module reload after every boot. Neither initramfs image
  contains `amdgpu/polaris10_sdma.bin`, so the boot-time SDMA load fails and KFD
  starts with no GPU node. You then see `Cannot create KFD process`, and torch
  reports no GPUs. Run `podman stop rocrfix2; modprobe -r amdgpu; modprobe amdgpu`
  and make sure that `/sys/class/kfd/kfd/topology/nodes` lists `0 1`. Stop
  `rocrfix2` and kill every python process first. Unbinding amdgpu while a
  process holds `/dev/kfd` reboots the box through the watchdog. A `rocm-smi`
  line for `00:01.0` does not prove that the GPU agent exists.
- `rocrfix2` is a build container that needs **two** mounts: `-v /data:/data` and
  `-v /opt/rocm:/opt/rocm`. Without the second there is no `hipcc` in it and its
  `/data/clrbuild4` tree cannot rebuild, because that tree points at
  `/opt/rocm/core-10.0` for the compiler. Recreating it with only `/data` looks
  like a working container until a build fails with `hipcc: No such file`.
- `pkill -f <name>` and `pgrep -f <name>` match the shell that runs them, because
  the pattern is in that shell's own command line. A cleanup line like
  `pkill -9 -f f16sweep` inside `sudo sh -c "..."` therefore kills the command
  before it launches anything. Write the first character as a bracket class
  (`pkill -9 -f "[f]16sweep"`) and the same for `pgrep`.
- A `podman rm -f` over `$(podman ps -aq)` also removes the long-lived build
  containers. Name the containers you mean.
- Do not set `amdgpu.noretry=1` on this card. GPUVM page-table retries are
  needed. With `gpu_recovery=0` a fault under it wedges the box completely, and
  pstore holds nothing.
- Kernels: the default is `7.1.12-200.fc44` with `gpu_recovery=1`. The debug
  build is `7.1.8-dbg3`, which runs as `7.1.8-dirty`. It sets `gpu_recovery=0` so
  that faults stall instead of resetting, and it adds the debugfs files
  `gfx803_ctxb`, `gfx803_flush_tlb`, `gfx803_shmem`, `gfx803_ptwalk`, and
  `gfx803_readphys`. Switch with `grubby --set-default=/boot/vmlinuz-...` and
  reboot. Its source tree is on the dev machine at `/usr/src/linux-7.1.8-local`,
  not on the box. Build the `.ko` there and copy it over.
- Do not extract an archive into `/` or `/lib` on the box. That replaced the
  `/lib` to `usr/lib` symlink with a real directory, which hid
  `/usr/lib/modules` and `/usr/lib/firmware` from the boot and crash-looped every
  kernel. If the box seems to have lost its kernels, look first at whether `/lib`
  is still a symlink.
- Saved libraries for A/B tests live in `/data/s6/`:
  `libamdhip64.TCINV-v3.so` (the deployed build, md5
  `a79a75631b40e6a731586f7feb03ae5d`),
  `libamdhip64.TCINV-v1-dispatchonly.so`, and
  `libamdhip64.PRE-TCINV-CIchain.so` (md5 `195f17d9ad85f94bbda58b3375c17a78`, the
  genuine pre-patch CI chain). The rocSOLVER work in progress is under
  `/data/rsbuild*`. Name a library backup after the real directory, not after
  `$(basename $D)`. Both `/opt/rocm/lib` and `/opt/rocm/core-10.0/lib` have the
  basename `lib`, so such a name overwrites one backup with the other.
- Treat the GPU as single-tenant. Two torch jobs at once invalidate timing and
  fault measurements. `timeout N podman exec ...` kills the client and leaves the
  process running inside the container holding its GPU context, so a later run
  inherits a second tenant: it fails as `CUDA error: out of memory` or as an
  illegal access, and looks like a library bug. Run
  `podman exec <c> pkill -9 -f 'python3 /data'` between arms.
- An SDMA ring timeout is terminal while `gpu_recovery=0`: dmesg says
  `ring sdma1 timeout` then `GPU recovery disabled`, new GPU work never starts,
  and processes go into `D` state, where even `kill -9` does not reach them and
  `modprobe -r amdgpu` blocks forever. Only a reboot clears it, and a forced
  reboot with those tasks present can leave the box down for a long fsck.

### vLLM on gfx803

The gfx803 vLLM hard fork lives at `vllm/` (repo root, the 10.0 line). It targets
the ROCm 10.0 stack and is assumed to work against it. The hand-written gfx803
kernels (`vllm/vllm/gfx803_kernels/*.hip`) are version-agnostic source. Each one
is compiled once with the stack's own
`hipcc --offload-arch=gfx803 -O3 -shared -fPIC`, and each loader's docstring
gives the exact call. `librocblas.so` resolves through the stack's
`LD_LIBRARY_PATH`, which is `/opt/rocm/core-10.0/lib` on 10.0. The compiled `.so`
files are built on the box next to their loaders and never committed, so this
repo pins nothing stack-specific and a fresh build on the 10.0 stack works.

Hardware validation of vLLM on the 10.0 stack is done (2026-09-02). Two crashes
blocked it, and both were in the ROCm 10.0 stack rather than in vLLM.
`patches/rocm-systems/va-reuse-defer-noremap.patch` and
`patches/rocm-systems/d2h-null-dsthost.patch` fix them, and `AGENTS.md` gives the
reason for each. Measured on the box with `qwen35_2b_bench_v3.py`: EXIT=0,
prefill 311.0 tok/s, decode 30.2 tok/s. The 7.14 record for the same bench is
331.7 and 24.4 tok/s. `rocm7.14/README.md` has the investigation and tuning notes
from the 7.14 period.

### Cross-dispatch coherence on gfx803 (knob `CLR_GFX8_TC_INVALIDATE`)

On gfx803 the CP ignores the AQL `SCACQUIRE` and `SCRELEASE` scope bits, and those
bits are ROCm's only way to express cache coherence across dispatches. The compute
shader's TC is also not maintained at a dispatch boundary. Three silent faults
follow, and one packet fixes all three.

- A kernel that reads a virtual address that an earlier kernel overwrote can get
  the pre-overwrite bytes from a stale TC line. The correct data is already in
  VRAM.
- A copy engine reads DRAM and never looks at the shader TC. So it copies out
  bytes that a compute kernel overwrote but has not written back. The GPU result
  is correct and the host copy of it is not.
- Same as the one above, but the kernel that wrote the bytes ran on another queue.
  Then there is no ring that holds both the writer and the read, so no writeback
  placed on the reader's side can be ordered behind the writer. It has to be
  published by the writing queue.

Anything that recycles virtual addresses is exposed. That is why this appears
under torch's caching allocator and not in a raw `hipMalloc`/`hipFree` program,
which never re-reads an address that a kernel cached.

`patches/rocm-systems/gfx803-tc-invalidate-acquire-mem.patch` fixes all three at
the source. It publishes one PM4 `ACQUIRE_MEM` (`TC_ACTION_ENA|TC_WB_ACTION_ENA`, full
address range) in its own AQL ring slot, in three places: ahead of every kernel
dispatch, inside `VirtualGPU::releaseGpuMemoryFence()` between that function's
barrier and a second barrier, and ahead of an event-record barrier in
`VirtualGPU::submitMarker()`. The second barrier makes sure that the completion
signal a copy engine waits on retires only after the writeback. The third site is
what closes the cross-stream case: a queue that is about to be waited on publishes
its own caches, because a writeback enqueued on the *reading* queue cannot be
ordered behind a kernel on the *writing* one. It is the raw slot only, with no
barrier and no completion signal, which is what keeps it out of the marker's
signal bookkeeping. All sites are on by default for ISA 8 and older.
`CLR_GFX8_TC_INVALIDATE=0` turns the whole thing off for A/B runs, and
`CLR_GFX8_TC_RECORD_FENCE=0` turns off the event-record site alone.

State on the 10.0 line (2026-09-05): the torch correctness suite
`tools/correctness-suite/torch_op_suite.py` gives 202/202 PASS, 0 BAD, 0
NONFINITE on the box stack. Inside the image the same suite reports 118 PASS, 0
BAD, 0 NONFINITE and 84 ERRORs that are all `/ INDUCTOR`, because the image ships
no Triton and inductor therefore cannot compile: no wrong answer, but not a
coverage claim either. With the knob off, the box stack fails 13 of them, 5 with
NONFINITE.
`tools/tc-staleness/` probes: bmm 0/40, soak 0/527 across 5 seeds, and a 60-step
Adam training run that is bit-identical to the CPU loss trajectory. The same
training run is not reproducible with the knob off. Cost: D2H +2.7%, H2D +1.5%,
launch-bound tiny ops +2.4%, and compute unchanged (fp16 2048^3 GEMM +0.04%, conv
and GPU-to-GPU clone unchanged).

The cross-stream case that this section used to leave open is closed. A producer on
another `torch.cuda.Stream` feeding a host copy on the default stream went from 21
anomalies in 960 checks to 0 in 2000, on one binary with
`CLR_GFX8_TC_RECORD_FENCE` toggled; `multistream2.py` 0/1600, and `ms4.py`, which
copies into pinned host memory and fans two producers into one consumer, 0/2400.
Copy bandwidth is unchanged (D2H pinned 2.07 to 2.03 GiB/s), because no copy moved
off the copy engine. Two fixes that look right and do not work are recorded in the
patch header, because someone will try them again: attaching the producer's
dependency to the consumer's fence barrier measures 11/960 either way, and a
CP-side `WAIT_REG_MEM` does not execute inside an AQL ring at all.

One path is still exposed, and the patch header lists it in its CAVEAT section.

- The HIP graph replay batch path (`dispatchAqlPacketBatchFlat`) issues no
  `ACQUIRE_MEM`. A torch `CUDAGraph` capture of the poisoned sequence does not
  reproduce (0/40 with the knob off), so this gap is unproven rather than known
  broken. `tools/tc-staleness/graphprobe.py` is the probe for it.

### torch.linalg on gfx803

Every `torch.linalg` entry point that reaches hipSOLVER and rocSOLVER SIGSEGVs.
That list is `qr`, `svd`, `svdvals`, `eigh`, `eigvalsh`, `cholesky`,
`cholesky_solve`, `cholesky_inverse`, `solve`, `solve_batched`, `inv`, `lstsq`,
`pinv`, `matrix_rank`, `det`, `slogdet`, `triangular_solve`, `norm('nuc')`, and
`cond`. Only `eigvals` runs. The fault lands in `hipLaunchKernel`, called from
`rocsolver::init_scalars<float>`. This is not a wrong number and it is not the
coherence bug above: the crash set is the same against the pre-patch
`libamdhip64`, with `CLR_GFX8_TC_INVALIDATE` on or off, and on the 7.14 stack too.
Two independent causes are involved.

1. There is no device code. The rocSOLVER in the pinned 10.0 stack has an empty
   `.hip_fatbin`. `objcopy --only-section=.hip_fatbin` yields 0 bytes (a NOBITS
   section) against 4.7 MB for our own rocBLAS build and 43 MB for
   `libtorch_hip`. Its CMake default target list is `gfx900 / gfx906 / gfx908`
   plus newer, so a gfx803 build of it has never existed in this stack, and the
   7.14 line carries the identical file (same md5). hipSOLVER needs no rebuild of
   its own. It has no `.hip_fatbin` at all and is a host-side wrapper over
   rocSOLVER.
2. The wave size is wrong once it does build. `lib_device_helpers.hpp` sets
   `WarpSize = 64` only under `#if defined(__GFX9__)`. That leaves gfx8xx, which
   is also a wave64 ISA, on the 32-lane branch while its reductions take the
   wave64 (`is_cdna`) DPP path on purpose. The consumers of the constant (`larfg`,
   `larf`, `lange`, `latrd`, and the LACN2 norm and condition helpers) then store
   each wave's full sum in two shared-memory slots, and the combine loop adds it
   twice. Measured on the card with a replica of that pattern: ratio 1.982 with
   `WarpSize=32`, and 0.991 with the constant corrected.

Cause 1 is fixed and measured. The `rocsolver-builder` stage builds rocSOLVER with
`-DAMDGPU_TARGETS=gfx803` and gates on `.hip_fatbin` size, the way the rocBLAS stage
does, so an empty payload can never ship quietly. With the library that stage
produces, all 13 tested `torch.linalg` routines run and match a CPU reference, with
a worst relative error of 1.0e-5, where the stock stack SIGSEGVs on all 13.
hipSOLVER needs no rebuild, because it resolves into this library by SONAME at load
time.

Cause 2 is patched but not yet proven to matter. `patches/rocsolver/
rocsolver-wavesize-gfx8.patch` fixes a constant that is objectively wrong for a
wave64 ISA, and the pattern it feeds double-counts in isolation, but an ablation
between patched and unpatched builds of the same tree gave byte-identical output on
all 39 checks, at n=128, 256 and 512, non-square, rank-deficient and batched. So no
`torch.linalg` result is known to depend on it, and it is recorded as latent rather
than as a measured correction. See its header for what to exercise next.

### fp16 GEMM and convolution on gfx803 (the SGEMM shim)

Tensile's own SGEMM kernels are unreliable on this card, so the stack ships an
`LD_PRELOAD` shim (`patches/rocblas/sgemm-shim/`) that answers three cases with
kernels verified on hardware: the standard-algo f32 `rocblas_sgemm` and
`rocblas_gemm_ex` path, the f16 `rocblas_gemm_ex` path, and the small-problem
`rocblas_gemm_strided_batched_ex` path that MIGraphX's batched attention dots
arrive in. Everything else falls through to the real rocBLAS symbol.

The shim is compiled in its own always-executed Dockerfile stage (`sgemm-shim-builder`),
because a build that supplies `ROCBLAS_IMAGE` skips `rocblas-builder` entirely: building
the shim there would ship whatever `.so` the published image happens to carry, however
current this tree's sources are. Each takeover is asserted by a marker string in the
compiled library, in that stage and again in the final stage, so a stale shim fails the
build instead of passing silently.

The f16 route had a caller bug, not a kernel bug. Under the column-major contract
that its own gate checks, the row-major product that lands in the output buffer is
`b @ a` with `M=n, N=m, K=k`; it passed `(m, n)`. That computes the transpose and
reads `m*k` elements from a `k*n` buffer, so it was only ever right when `m == n`.
Convolutions are the common case where it is not: im2col makes `K = Cin*R*S` and
`M = H*W`, so a 3-channel 3x3 conv arrives as `m=4096, n=64, k=27`, and the result
was garbage (max ~5e4 where the true value is ~28) with an out-of-bounds read that
later surfaced as `illegal memory access`, usually inside MIOpen's next kernel load
rather than at the GEMM itself. Non-square batched dots were wrong too. Measured
case by case against a float32 reference (`tools/correctness-suite/fp16_gemm_sweep.py`,
27 cases, both shims, one case per process because a fault kills the context): the
old mapping was correct on the 8 square cases and wrong or faulting on all 19
others; the fixed mapping is correct on all 27, and the 8 square cases are
unchanged, since for `m == n` the two calls are the same call. The kernel itself is
innocent: called directly on its documented contract it gives max error 7e-05 at
`M=4096, N=64, K=27`.

## Component images, pins, and line provenance

A component stage either builds from source or inherits a published
`ghcr.io/<owner>/rocm-<component>-builder` image, chosen by the `*_IMAGE`
build-args. That inheritance is not a cache. It is an artifact handoff, and a tag
name alone did not identify it well enough.

- The intermediate component tags now name the line (`:gfx803-rocm10`, through the
  workflow input `intermediate-suffix`). The main line used to publish and consume
  the unsuffixed `:gfx803`, which is also what an earlier line of this repo
  published under. So a component whose 10.0 job had not run since the switch was
  consumed into a 10.0 image as if it belonged there. This was seen directly:
  `rocm-migraphx-builder:gfx803` and `rocm-migraphx-torch-builder:gfx803` held
  `/opt/rocm/core-7.14` and a torch 2.13 wheel. A mixed-line image assembles,
  imports, and misbehaves only on real hardware. The final image keeps the names
  that downstream pulls: `latest-gfx803`, `rocm10.0-gfx803`, and
  `<date>-gfx803`.
- Every stage states which line it inherited. A component image carries
  `/opt/rocm/.gfx803-line` (written by `scripts/gfx803-line.sh`) with the line,
  the `rocm-gfx803` revision, and the resolved upstream commits, plus the
  `io.rocm.gfx803.*` image labels. A marker that names a different line stops the
  build. An inherited tree with no marker (published before this scheme) warns
  instead of stopping, because that is every image published to date. Set
  `GFX803_LINE_STRICT=1` to make the missing marker fatal too.
- Branch pins stay the policy, and CI adds the commit. The `*_REF` args are still
  release branches (see "Component pins" in `AGENTS.md`). A `git clone` of a
  branch happens inside a `RUN`, so the layer's cache key is the command text,
  and that text does not change when upstream pushes. So a stale layer can
  survive a tip move with nothing to show it. The component workflow resolves
  every ref to its commit once per run with `git ls-remote`, and it stops the run
  if a ref cannot be resolved. The commit is passed as `*_SHA`, and
  `scripts/git-pin.sh` fetches that commit with `--depth 1` instead of cloning
  full history. The resolved set is recorded in the image marker.

Nobody sets these values by hand, and nothing needs re-cutting when AMD pushes.
The resolution runs by itself on every run. The `*_SHA` build-args are optional
inputs to the Dockerfile, not to people. A plain local `docker build` with no
build-args still works: it follows the branch, and `git-pin.sh` says so on stderr.
A run that cannot resolve a pin fails after three retries, rather than producing
an image that cannot say what it holds.

## Building

All lines build the same way: a multi-stage Dockerfile, with patches applied per
project before that component compiles from source.

```sh
# rocm10 (this repo's root)
docker build -t rocm-gfx803:rocm10 .

# rocm7.14
docker build -t rocm-gfx803:rocm7.14 -f rocm7.14/Dockerfile rocm7.14/

# rocm6.4.4
docker build -t rocm-gfx803:rocm6.4.4 -f rocm6.4.4/Dockerfile rocm6.4.4/
```

Every component (ROCR-Runtime and CLR, rocBLAS, MIOpen, rocSOLVER, MIGraphX,
PyTorch, torchvision, torchaudio, ONNX Runtime) is compiled from source in its own
stage. There is no prebuilt gfx803 wheel anywhere upstream, unlike the newer and
more common architectures that the mainline repo builds, where a published wheel
sometimes replaces a recompile. CI keeps the same rule, and it publishes each
stage as its own image so a later stage can inherit it through `*_IMAGE` instead
of rebuilding it. See "Component images, pins, and line provenance".

For the on-hardware checks, run `verify.py` (also
`rocm7.14/verify.py` and `rocm6.4.4/verify.py`, the same in spirit) inside a
container started with `--device=/dev/kfd --device=/dev/dri --group-add video`.
It checks the paths that only real hardware can show, and each one can import
cleanly and still fail or fall back silently:

- The MIGraphX EP is present and does real GPU work.
- rocBLAS GEMM and MIOpen convolution match a CPU reference.
- The rocBLAS library carries no `_WGM8` kernels.
- rocSOLVER embeds real device code for gfx803.
- `torch.linalg` results match CPU.
- A host copy of a fresh GPU result is not stale.

## Patches: philosophy and conventions

Every patch under `patches/`, `rocm7.14/patches/`, and `rocm6.4.4/patches/`
carries its own header. The header gives the reason (what is broken, how it was
found, and the hardware measurements where they apply) before the change (the
diff). When a patch was carried from one line to another and the upstream source
moved, the header also carries a re-diff note. Read the header before you touch
the code it targets, because the diff alone rarely shows the reason.

Two apply styles exist on purpose. The `.sh` drivers under
`patches/rocm-systems/` use `git apply`, because their target `rocm-systems` is
cloned as a real git repository root. Everything else (`rocblas/`, `miopen/`,
`rocsolver/`, `migraphx/`, `pytorch/`) uses `patch -p1`, because those targets are
sparse-checked-out subdirectories of a monorepo. On this box's git version,
`git apply --check` in such a tree reports success and changes nothing ("Skipped
patch", exit 0) instead of failing loudly. Every driver checks its own result: it
greps for a marker string after the apply and fails the build when the marker is
absent. So a patch that quietly stopped applying cannot ship unpatched code.

## When a patch needs updating

A gfx803 patch stops applying, or starts applying with fuzz, when the pinned
upstream commit moves and the target file changed shape around it. That is
expected. It is not a sign that the patch is wrong. Before you re-diff:

1. Make sure that the bug is still there. A newer upstream commit sometimes fixes
   the underlying fault outright. This has happened: see the MIGraphX section of
   `rocm7.14/MIGRATION_NOTES.md`, where two 6.4.4-era ONNX parser patches were
   fully obsolete against 7.14. One was obsolete because the fix landed upstream,
   and one because the whole code path it patched was replaced. Grep the new
   source for the target function or struct before you assume a re-diff is
   needed.
2. Make sure that the fix is still gfx803-specific. Some of these bugs are
   architecture-general faults that gfx803's kernel and solver selection merely
   exposes, such as the WGM Tensile swizzle bug and the small-GEMM assembly
   miscompute. Others are real hardware gaps. If a new investigation shows that
   the same code path misfires on other architectures, report it upstream, and
   patch here only in addition to that.
3. Re-test on real hardware, and do not treat a clean apply as a fix. A patch
   that compiles proves nothing about correctness. Several patches here were
   re-diffed successfully and then marked NOT YET RE-VERIFIED ON REAL HARDWARE,
   until someone ran the original repro against the new binaries. Each patch
   header and the Status section above state the current state.

## Host BIOS setting: keep PCIe ASPM off

On at least one gfx803 host, PCIe ASPM (link power management) enabled in the
BIOS caused rare stalls and hangs under GPU load that were extremely hard to
diagnose. They look like a driver or kernel bug, and they can cost hours before
you find a power-management setting that is outside the software stack entirely.
Keep ASPM disabled in the BIOS on any gfx803 host until that specific board
proves otherwise. If the board will not hand OS-level control of ASPM to Linux,
clear the already-programmed register bits with `setpci` or the kernel cmdline.
`tools/host-setup/` has a working `setpci`-based systemd unit for such boards.

## Host VBIOS setting: mining-tuned VRAM clocks cause random GPU faults and hangs

At least one gfx803 card used with this repo (Sapphire RX 470 8GB Mining UEFI,
Hynix `H5GQ8H24MJR` VRAM) shipped with a mining-tuned VBIOS that runs VRAM (MCLK)
at 2000-2100 MHz. That is above the rating of a 7 Gbps Hynix chip. Under a
correctness-checked compute workload this produced two symptoms. Both needed real
investigation to rule out as software bugs: ioctl tracing, PM4 dispatch tracing,
a kernel-side TLB-flush review, and GPU-side wave-state capture through debugfs.
`RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md` holds that investigation.

- `tools/correctness-suite/pool_sweep`: an intermittent GPU VM fault at a
  deterministic address, in about 50% of runs.
- vLLM and other sustained loads: an intermittent hang that cannot be recovered,
  where a wave waits forever in `s_waitcnt vmcnt(0)` for a vector-memory op that
  never returns. Killing the stuck process fails a KFD queue eviction, so a
  reboot is needed.

A mining workload tolerates occasional VRAM bit errors that a
correctness-checked or long-running workload does not. In the latter it surfaces
as a fault, a hang, or a silently wrong result.

Flash a VBIOS whose VRAM clock matches the real rating of the installed memory. Use `amdvbflash`'s force-flash mode (`amdvbflash -f -p 0 <rom>`), and
always dump and keep the existing ROM first. Two independent hardware tests
confirm this:

- The card's own mining VBIOS, with MCLK capped at 1750 MHz through core
  overdrive (`amdgpu.ppfeaturemask=0xffffffff` plus `pp_od_clk_voltage`): 64 of
  64 clean runs, against repeated hangs and crashes in the same boot at 2000 MHz
  with every other binary held identical.
- A real Sapphire RX570 Nitro VBIOS with correct-vendor Hynix straps
  (`113-2E366AU-X56`, from
  https://www.techpowerup.com/vgabios/212597/212597), whose stock MCLK table ends
  at 1750 MHz: 75 of 75 clean runs at stock settings, no overdrive. This is the
  recommended fix for the same Sapphire RX 470 8GB Mining UEFI card with Hynix
  memory, and it needs no software workaround at all.

Two other RX570 VBIOS files with Samsung straps (the wrong vendor for this
card's Hynix chips) did not probe at all (`SMU load firmware failed`,
`probe with driver amdgpu failed with error -22`) instead of hanging. That is a
different and harder failure mode. Match the VBIOS memory-vendor strap to the
chips that are physically installed, and not only to the card model and VRAM
size.

Read the VRAM clock with `cat /sys/class/drm/card*/device/pp_dpm_mclk` and compare
it with the card's real rating before you accept a gfx803 GPU fault or hang report
as a software bug.

## What needs real gfx803 hardware to validate, and what does not

- The card is needed for anything that dispatches a GPU kernel: `verify.py`,
  `tools/correctness-suite/`, `tools/tc-staleness/`, any real transcription or inference
  run, and MIOpen's own `MIOpenDriver -V 1` check. `tools/imgvalidate.sh <image-tag>`
  runs the whole set against a built image in one pass, including the control arms that
  must reproduce each bug with its fix switched off; run it before calling a release
  validated. Silent miscompute is the common bug
  class here, where `rocblas_status_success` returns with wrong numbers. A CPU or
  an emulator cannot reproduce it, and a patch that only "applies clean" and
  "compiles" has proved nothing about correctness.
- The card is not needed for these: whether a Dockerfile builds at all, whether a
  patch applies against a given pin, and where a bug lives, which source-level
  tracing answers. MIOpen's own `MIOPEN_ENABLE_LOGGING_CMD` traces and
  `MIGRAPHX_TRACE_COMPILE` plus upstream source diffs located several root causes
  in this repo's history without touching a GPU. Cross-architecture differential
  tests against an image for a different card also need no gfx803 card, and
  `rocm7.14/MIGRATION_NOTES.md` uses them to separate "this line broke it" from
  "upstream never worked here".

## Convergence

Once the rocm10 line is confirmed at least as solid as `rocm7.14/` (clean
correctness suite, ORT suite parity, comparable real-model results), the lines are
diffed and merged deliberately. What is still only in 7.14 or 6.4.4, and belongs in
the shared structure, moves up, and the separate copies collapse back into a shared
structure. This is not done, and each line is still maintained at its own
verification level.

## See also

- `MIGRATION_NOTES.md`: the 10.0 migration log, with the pins, what was
  inherited, and what is still open.
- `rocm7.14/MIGRATION_NOTES.md`: the detailed as-found 7.14 investigation log.
  Read it before you assume something on that line is broken or fixed. It is the
  main record of what was tested and how.
- `rocm7.14/README.md`: the 7.14 line in full, including the vLLM on gfx803
  investigation and tuning notes.
- `rocm6.4.4/KERNEL_BUGS.md`: the original gfx803 bug-hunting method and bug
  record for the 6.4.4 line.
- [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder): the mainline
  (gfx900 and newer) build that this repo split from and follows version-wise.
