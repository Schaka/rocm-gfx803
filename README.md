# rocm-gfx803

This repo keeps AMD Polaris (gfx803: RX 460/470/480/560/570/580/590 and close
relatives) working on the MIGraphX + ONNX Runtime + PyTorch stack. It was split
out of [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) into its own
repository.

## Prebuilt images: do not build this yourself

CI builds the final image and pushes it to GHCR. You only need a local build
when you change a patch. Pull the image you want:

```bash
# versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm10.0-gfx803

# always the newest successful build
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803
```

`docker-bake.hcl` gives the whole tag scheme: per-component images, cache tags, and
dated tags.

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

## Status

rocm10 (this repo's root) is the only line, under active development and partly
hardware-tested. The ROCm 10.0 stack on the test box runs rocBLAS, MIOpen,
MIGraphX, PyTorch, ORT, and the vLLM fork on a real card. The torch correctness
suite gives 202/202 PASS with the current patch set on the box stack. The image
ships Triton, verified on the real card: the JIT-compile, GEMM, and hang-repro
checks all pass (`patches/triton/README.md`).
The full image builds end to end, and `tools/imgvalidate.sh` runs that whole
gate against the image itself on the card: its `libamdhip64` and its
`librocsolver.so.0` (13 of 13 `torch.linalg` routines within 1.0e-5 of CPU) are
the shipped binaries, not a hand-built pair swapped in. `tools/imgvalidate.sh <image-tag>` is the on-card gate for a whole
image: it asserts the shipped libraries' markers, runs `verify.py`, the coherence
probes with a control arm that must reproduce the corruption, the fp16 GEMM and
convolution sweep with and without the shim's takeover, and the op suite. The
last run of it (2026-09-05) came back with verify.py all-pass, 0 anomalies in 480
cross-stream checks against 16 in the same-configuration control, 27/27 fp16
cases with the shim and 27/27 against real rocBLAS, op suite `BAD 0, NONFINITE
0`, and no GPU reset or ring timeout in dmesg. `MIGRATION_NOTES.md` has the
details, and each patch header states its own verification state.

### Required host setup

VBIOS and VRAM clock: this card's VRAM must run at or below its rated speed. The
rating is 1750 MHz on the mining-tuned VBIOS that this box shipped with, and that
limit was confirmed by core overdrive. A stock VBIOS from the correct vendor
needs no overdrive. VRAM above spec causes real GPU VM faults in MIOpen
(`pool_sweep`) and GPU hangs in vLLM. Both have the same hardware cause and
neither is a software bug. See "Host VBIOS setting" below and
`RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md`.

Optional, informational only: `patches/rocm-systems/aql-ring-queue-full-workaround.patch`
restores the AQL ring's double mapping for GFXIP 7 and 8. It raises the queue
from 64 packets to 131072, which is 2048 times the unpatched cap, and the build
applies it automatically, so there is no host action to take. It requires a
kernel that does NOT carry `REFERENCE-amdkfd-gfx7-8-queue-size-writeback`, and it
must not be combined with `graph-replay-queue-size-cap.patch`. With it,
`graph-replay-batch-chunk-deadlock.patch` is not needed.

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

Hardware validation of vLLM on the 10.0 stack is done (2026-09-02). Measured on
the box with `qwen35_2b_bench_v3.py`: EXIT=0, prefill 311.0 tok/s, decode
30.2 tok/s. vLLM's runs on this stack depend on
`patches/rocm-systems/va-reuse-defer-noremap.patch` and
`patches/rocm-systems/d2h-null-dsthost.patch`; `AGENTS.md` gives the reason for
each.

The published `latest-gfx803` image carries that fork too: `final` installs the
wheel and the three compiled gfx803 kernels into `/opt/venv`, so running vLLM
needs no box-native install. What it does need is
`patches/triton/gfx803-vdot-gate.patch` inside the image's triton: without it an
fp16 `tl.dot` is a fatal LLVM abort and no engine starts, because the attention
backend's prefill kernel is one. Measured on the image (2026-09-12) with that
patch: engine init 88 s and coherent greedy generation at 46.6 tok/s on
Qwen3-0.6B. A bf16 run of the same model starts but produces incoherent output;
that is a separate, open problem, and fp16 is the dtype the numbers above use.

## Component images, pins, and line provenance

A component target either builds from source or consumes a published
`ghcr.io/<owner>/rocm-<component>-builder` image as a named build context, chosen
by the `WITH_*_IMAGE` variables in `docker-bake.hcl`. That handoff is not a
cache. It is an artifact handoff, and a tag name alone did not identify it well
enough.

- The intermediate component tags name the line (`:gfx803-rocm10`, from the `LINE`
  variable). The main line used to publish and consume
  the unsuffixed `:gfx803`, which is also what an earlier line of this repo
  published under. So a component whose 10.0 job had not run since the switch was
  consumed into a 10.0 image as if it belonged there. This was seen directly:
  `rocm-migraphx-builder:gfx803` and `rocm-migraphx-torch-builder:gfx803` held
  `/opt/rocm/core-7.14` and a torch 2.13 wheel. A mixed-line image assembles,
  imports, and misbehaves only on real hardware. The final image keeps the names
  that downstream pulls: `latest-gfx803`, `rocm10.0-gfx803`, and
  `<date>-gfx803`.
- Every target states which line it inherited. A component image carries
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
  survive a tip move with nothing to show it. `scripts/ci/resolve-pins.sh` reads
  the refs from `docker-bake.hcl` and resolves each one to its commit once per
  run with `git ls-remote`. It stops the run if a ref cannot be resolved. The commit is passed as `*_SHA`, and
  `scripts/git-pin.sh` fetches that commit with `--depth 1` instead of cloning
  full history. The resolved set is recorded in the image marker.

Nobody sets these values by hand, and nothing needs re-cutting when AMD pushes.
The resolution runs by itself on every run. The `*_SHA` values are optional
inputs to the build, not to people. A plain local `docker buildx bake` still
works: it follows the branch, and `git-pin.sh` says so on stderr.
A run that cannot resolve a pin fails after three retries, rather than producing
an image that cannot say what it holds.

## Building

The build applies its patches per project before that component compiles from
source. It is a Docker Bake graph.

```sh
# the whole graph
docker buildx bake

# one component and its dependencies
docker buildx bake rocblas

# the resolved graph, with no build: tags, contexts, args, cache refs
docker buildx bake --print final

# every version pin in one place
docker buildx bake --print pins
```

Every component (ROCR-Runtime and CLR, rocBLAS, MIOpen, rocSOLVER, MIGraphX,
PyTorch, torchvision, torchaudio, ONNX Runtime) is compiled from source. There is
no prebuilt gfx803 wheel anywhere upstream. The mainline repo's newer
architectures sometimes have one, and a published wheel replaces a recompile
there. CI keeps the same rule here. It publishes each component as its own image,
so a later component consumes it as a build context instead of rebuilding it. See
"Component images, pins, and line provenance".

`docker-bake.hcl` is the single source of truth: image names, tags, cache refs,
version pins, and which target gets its dependency from a prebuilt image. Each
`docker/<name>.Dockerfile` declares bare ARGs and takes its values from there.
The long build steps live in `scripts/build/<name>.sh`, mounted into the build
rather than copied into the image. The workflows set variables and name one
target, and carry no build logic of their own.

For the on-hardware checks, run `verify.py` inside a container started with
`--device=/dev/kfd --device=/dev/dri --group-add video`. It checks the paths
that only real hardware can show, and each one can import cleanly and still
fail or fall back silently:

- The MIGraphX EP is present and does real GPU work.
- rocBLAS GEMM and MIOpen convolution match a CPU reference.
- The rocBLAS library carries no `_WGM8` kernels.
- rocSOLVER embeds real device code for gfx803.
- `torch.linalg` results match CPU.
- A host copy of a fresh GPU result is not stale.

## Patches: philosophy and conventions

Every patch under `patches/` carries its own header. The header gives the reason
(what is broken, how it was found, and the hardware measurements where they
apply) before the change (the diff). When the upstream source moved and a patch
needed re-diffing, the header also carries that note. Read the header before you
touch the code it targets, because the diff alone rarely shows the reason.

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
   the underlying fault outright, or replaces the whole code path a patch
   targeted with something new. This repo's history has both outcomes; see
   `MIGRATION_NOTES.md`. Grep the new source for the target function or struct
   before you assume a re-diff is needed.
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
  in this repo's history without touching a GPU. A cross-architecture
  differential test against an image for a different card also needs no gfx803
  card, and separates "this repo broke it" from "upstream never worked here".

## See also

- `MIGRATION_NOTES.md`: the migration log, with the pins, what was inherited,
  and what is still open.
- [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder): the mainline
  (gfx900 and newer) build that this repo split from and follows version-wise.
