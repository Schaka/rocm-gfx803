# Polaris / gfx803 variant

Experimental image for Polaris cards -- RX 460, 470, 480, 550, 560, 570, 580,
590, plus the R9 Fury/Nano generation that shares the gfx803 ISA. Built from
`Dockerfile.gfx803` by the manual `gfx803.yml` workflow, entirely separate from
the nightly matrix in the [main README](../README.md).

> **This is not a supported configuration.** Officially supported is gfx900 and
> above, i.e. what AMD lists in the ROCm supported-GPU matrix. gfx803 is
> best-effort and slow by construction (see [Expectations](#expectations)).
> It has been verified on real hardware (Sapphire RX 470 8GB Mining UEFI) with
> [`verify_gfx803.py`](#verifying-on-hardware) -- MIGraphX EP inference,
> rocBLAS GEMM, and MIOpen convolution all ran correctly. Issues against it
> are welcome as reports, not as regressions.

## Why it's a separate image, not another arch in the matrix

ROCm 7 removed Polaris support outright. ROCR-Runtime rejects agents reporting
"legacy" HSA doorbell types 0 and 1, which is exactly what Polaris hardware
reports, so the GPU agent is dropped before it is ever enumerated --
`rocminfo` fails with `HSA_STATUS_ERROR` and `clinfo` reports zero devices. See
[ROCm/clr#269](https://github.com/ROCm/clr/issues/269), where an AMD engineer
root-causes it and publishes a patched ROCR in a
[TheRock fork](https://github.com/lucbruni-amd/TheRock/tree/lb/gfx803-polaris-support).

That patch restores enumeration, but the same commit's build config has to
exclude MIOpen, composable_kernel, hipBLASLt, hipSPARSELt, rocWMMA and
rocprofiler-compute for gfx803 -- which is most of what MIGraphX needs. On
ROCm 7, a Polaris image would enumerate the card and then have nothing to
dispatch convolutions to.

ROCm 6.x has no doorbell check at all. A stock `rocm/dev-ubuntu-24.04:6.4.x`
image enumerates a Polaris card as shipped, no runtime patch required, and
MIOpen still works. So this variant pins the last ROCm 6 release rather than
carrying a patched runtime forward. That means it can't be an entry in
`ROCM_ARCH` -- it isn't a different GPU target on the same stack, it's a
different ROCm major, base image and dependency set.

## What gets rebuilt, and why

| Component | Source | Reason |
| --- | --- | --- |
| rocBLAS | `rocm-6.4.4` tag | Dropped gfx803 from its default `TARGET_LIST` at ROCm 6.0, but the gfx803 Tensile logic (`Logic/asm_full/r9nano/*.yaml`) is still in the tree, so `rmake.py -a gfx803` builds it back. The base image's prebuilt copy has no gfx803 code objects. |
| MIGraphX | `release/rocm-rel-6.4` | Prebuilt for gfx900+ only, same as on the main image. |
| PyTorch | `ROCm/pytorch` `release/2.8` | No gfx803 wheel has ever been published. torchvision and torchaudio are built alongside it. |
| ONNX Runtime | `v1.21.1` | Built against the MIGraphX above. |

**MIOpen is deliberately *not* rebuilt** -- it comes from the base image
untouched. It still carries its `Ellesmere`/`Baffin`/`Polaris10`/`Polaris11` →
`gfx803` device map and its gfx803 asm and Winograd convolution solvers. What
it lacks is a gfx803 `.kdb`, which is the *tuning* database: untuned, not
broken. This is the single component that is dead on ROCm 7 and alive here, and
it's why the 6.x route is viable at all.

## What is switched off

None of these have ever supported gfx8, on any branch:

- **rocMLIR** -- `AmdArchDb.cpp` knows gfx9/gfx10/gfx11/gfx12 only, verified
  identical from `rocm-5.7.0` through `rocm-6.4.0`.
- **composable_kernel** -- floor is gfx908 across the 6.x line (gfx900 on
  `develop`).
- **hipBLASLt** -- gfx90a and newer.

Turning them off in MIGraphX's cmake is necessary but not sufficient: `rbuild`
builds everything `requirements.txt` lists regardless of MIGraphX's own
options, so `composable_kernel` and `rocMLIR` are stripped from that file
before the build runs. Without that they'd still be compiled for gfx803, and
fail.

hipBLASLt can't be excluded at PyTorch build time -- torch 2.8 links it
unconditionally in `cmake/Dependencies.cmake` -- so the final image sets
`TORCH_BLAS_PREFER_HIPBLASLT=0` to keep GEMMs off a library with no gfx803
kernels.

## Expectations

Untuned MIOpen convolutions, no kernel fusion (no CK, no MLIR), on hardware
with no packed-fp16 and no dot-product instructions. This exists to make a
20-euro card usable, not competitive. For small models it may still beat CPU;
for anything large, measure before believing.

## Versions

| | Pinned to | Why |
| --- | --- | --- |
| Base | `rocm/dev-ubuntu-24.04:6.4.4-complete` | Last ROCm 6 release. `-complete` because MIGraphX needs MIOpen's dev files. |
| Python | 3.12 | Ubuntu 24.04's native interpreter, so no uv overlay is needed here (unlike the main image, whose 26.04 base ships 3.14). |
| PyTorch | `release/2.8` + torchvision `v0.23.0` + torchaudio `v2.8.0` | Matches AMD's own `rocm/onnxruntime:rocm6.4.4_ub24.04_ort1.21_torch2.8.0`. |
| ONNX Runtime | `v1.21.1` | Newest release AMD pairs with ROCm 6.4. Also still has the pre-removal `--use_rocm`/`--rocm_home` flags, unlike the v1.27 the main image builds. |

> **PyTorch 2.8 vs 2.6.** 2.8 is verified against ROCm 6.4.4, not against
> gfx803. The combination actually run on Polaris hardware is torch
> `release/2.6` + torchvision `v0.21.0` + torchaudio `v2.6.0`, by the
> [gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm) project. If
> torch misbehaves on-device, that triple is the known-good fallback and is
> exposed as workflow inputs (below) and build args.

## Packages

The same packages the main pipeline publishes to, with `gfx803` as just another
arch tag:

- `rocm-rocblas-builder:gfx803` -- ROCm with rocBLAS rebuilt for gfx803
- `rocm-migraphx-builder:gfx803` -- the above plus from-source MIGraphX
- `rocm-migraphx-torch-builder:gfx803` -- torch, torchvision, torchaudio wheels
- `rocm-migraphx-ort-builder:gfx803` -- ONNX Runtime wheel
- `rocm-migraphx-ort-torch-builder:latest-gfx803` / `:<YYYYMMDD>-gfx803` -- the
  combined image

Mind that these tags are the only ones in those packages built from
`Dockerfile.gfx803` against ROCm 6.4.4 -- every other arch tag is the ROCm 7.x
main pipeline. The versions pinned here have nothing to do with the ones the
nightly builds.

The first four are build plumbing. Downstream consumers want the last one,
which is the same layout as the main image (`/opt/venv` on `PATH`, `/opt/rocm`
with from-source MIGraphX) plus torchvision and torchaudio.

```dockerfile
ARG BASE_IMAGE=ghcr.io/<owner>/rocm-migraphx-ort-torch-builder:latest-gfx803
FROM ${BASE_IMAGE}
RUN python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Building

Manual only. There is deliberately no schedule: every version above is pinned
to an immutable ref, so a nightly run would rebuild identical inputs.

```
gh workflow run gfx803.yml
```

The chain is `rocblas` → {`migraphx`, `pytorch`} → `ort` → `final`. rocBLAS
comes first because both MIGraphX and PyTorch link it and both need the gfx803
rebuild; MIGraphX and PyTorch are independent and run in parallel.

Each component is individually skippable via the workflow's checkboxes --
rocBLAS in particular only changes when the ROCm version does, so a torch-only
rebuild shouldn't spend hours regenerating Tensile kernels. A skipped
component's downstream jobs reuse whatever it last published. The workflow also
exposes `pytorch-ref` / `torchvision-ref` / `torchaudio-ref` for the 2.6
fallback, and `debug` for a tmate session.

Locally:

```
docker build -f gfx803/Dockerfile.gfx803 -t rocm-gfx803-builder gfx803
```

## Running

Needs `/dev/kfd` and `/dev/dri` passed through, as any ROCm container does:

```
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    ghcr.io/<owner>/rocm-migraphx-ort-torch-builder:latest-gfx803 \
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The image already sets the Polaris runtime environment
(`HSA_OVERRIDE_GFX_VERSION=8.0.3`, `ROC_ENABLE_PRE_VEGA=1`,
`TORCH_BLAS_PREFER_HIPBLASLT=0`); no extra `-e` flags should be needed.

## Verifying on hardware

The build-time import check in the final stage of `Dockerfile.gfx803` only
proves the wheels import -- not that MIGraphX, rocBLAS, and MIOpen actually
dispatch to the card correctly. `verify_gfx803.py` in this repo runs a real
MIGraphX EP inference, a rocBLAS GEMM, and a MIOpen convolution, each checked
against expected output:

```
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$(pwd)/gfx803/verify_gfx803.py:/verify_gfx803.py" \
    ghcr.io/<owner>/rocm-migraphx-ort-torch-builder:latest-gfx803 \
    python3 /verify_gfx803.py
```

Confirmed passing on a Sapphire RX 470 8GB Mining UEFI: all four checks
(provider list, MIGraphX EP inference, rocBLAS GEMM, MIOpen convolution) ran
correctly.

If it hangs or the process dies without a Python traceback, check `dmesg` on
the host before suspecting the image -- see the kernel-version caveat below.

## Known runtime issue: ROCMExecutionProvider crashes on fused conv (musicnn-class models)

`ROCMExecutionProvider` (the plain kernel-based HIP EP, distinct from
MIGraphX) is present in this image's ONNX Runtime build as a fallback for
graphs MIGraphX can't compile -- see the [AudioMuse-AI ROCm
plugin](https://github.com/Schaka/audiomuse-rocm-plugin)'s
`plugin/rocm_accelerator/README.md` for the CLAP `keep_aspect_ratio_policy`
parse failure that motivated it. On real gfx803 hardware (Sapphire RX 470
8GB), that fallback **intermittently SIGSEGVs on any graph where ORT's
optimizer fuses Conv+Bias+Activation into a FusedConv node** -- constantly on
CNN-style models (musicnn's embedding/prediction heads), and rarely but
reproducibly on CLAP, whose conv stem is small next to its transformer body
but still enough to reach the fused path.

Root cause, isolated via `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6`: ORT's
graph optimizer fuses Conv+Bias+Activation into one node, which ONNX
Runtime's ROCM EP hands to MIOpen's Fusion Plan API rather than three
separate conv/bias/activation calls. On gfx803 that Fusion Plan path --
whether it lands on a fused Winograd solver (`ConvBinWinogradRxSFused`) or
falls through to "No supported fusion solvers found" and a plain-conv
fallback -- corrupts GPU state and eventually faults
(`Memory access fault ... Page not present or supervisor privilege`).

What was ruled out, each confirmed on real hardware, not assumed:

- **Not a stale/mismatched find-db.** `/opt/rocm/share/miopen/db/gfx803_*.fdb.txt`
  ships in 36-CU and 64-CU variants; the RX 470 is 32-CU and correctly falls
  through to its own per-run user db (`gfx803_32...ufdb.txt`) instead of a
  mismatched file -- deleting the shipped fdb files entirely made no
  difference to the crash.
- **Not a single bad solver.** `MIOPEN_DEBUG_CONV_WINOGRAD=0` (disables the
  non-fused Winograd solver) didn't help; `MIOPEN_DEBUG_AMD_FUSED_WINOGRAD=0`
  (disables the fused variant specifically) reduced but did not eliminate the
  crash -- it still faults on the "no fusion solver found" fallback path for
  some shapes.
- **Not a simple async race.** `HIP_LAUNCH_BLOCKING=1` (forces synchronous
  kernel launches) did not prevent the crash.
- **Not deterministic.** The exact same command, same audio sample, same
  shape, same run count crashes on one invocation and completes cleanly on
  the next. Session churn amplifies it heavily (creating and destroying
  sessions reshuffles ORT's arena layout), but a first-session crash in a
  fresh process has been observed too.
- **Not unstable VRAM.** Tried non-mining BIOS, 1750 Mhz VRAM clock, added tons
  of extra cooling for stability and saw no difference.
- **Not SDMA and not an async race** (from the later CLAP investigation):
  `HSA_ENABLE_SDMA=0` and `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` both
  still crash, and the serialized `AMD_LOG_LEVEL=4` trace pins the fault on
  the fused kernels themselves (`miopenSp3AsmConvRxSU_CBA`,
  `MIOpenConvUniBatchNormActiv`) reading past the end of the arena chunk
  holding their weights -- the HSA fault address lands exactly on the chunk's
  end boundary, and dmesg shows `read from 'TC'` (a compute kernel, not a
  copy engine). Whether the over-read faults depends on whether the
  neighbouring page happens to be mapped, which is where the non-determinism
  comes from.

**Practical conclusion: keep MIOpen's Fusion Plan API out of the picture
entirely.** Two complementary measures, both in the AudioMuse-AI ROCm plugin:
don't route CNN/conv-heavy models (musicnn) through `ROCMExecutionProvider`
at all -- MIGraphX is both faster for them (~22ms vs ~26-30ms mean per
inference, when ROCM doesn't crash) and stable -- and for graphs that must
use the ROCM EP (CLAP, which MIGraphX refuses to parse), disable ORT's
`ConvActivationFusion` graph optimizer per session via the
`optimization.disable_specified_optimizers` session config entry, so no
FusedConv nodes exist and every conv runs unfused (verified stable over a
200-iteration session-churn loop that kills an unpatched session within a
couple of iterations). This is a MIOpen/ROCm bug on unsupported-generation
hardware, not something fixable from an image build -- worth an upstream
report against ROCm/MIOpen if anyone has cycles, but no open issue is known
to track it yet.

## Host-side caveats

Neither of these can be fixed inside the image:

- **Linux 6.12 and 6.13 are reported to segfault on gfx803** (per the
  gfx803_rocm project). If the stack crashes on an otherwise correct setup,
  check the host kernel before suspecting the image.
- Mining-edition cards often have no display outputs, which is irrelevant for
  compute, but confirm the board is a genuine 8GB Polaris 10 and not a 4GB
  rebadge.

## Credit

The ROCm-6.x-plus-rebuilt-rocBLAS approach, and the specific knowledge of which
knobs Polaris needs, comes from two community projects:

- [robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm)
  -- actively maintained, ROCm 6.4, the source of the torch 2.6 hardware
  validation and the kernel-version warning
- [woodrex83/ROCm-For-RX580](https://github.com/woodrex83/ROCm-For-RX580) --
  the earlier ROCm 6.1.2 take

Neither builds MIGraphX or ONNX Runtime; that part is new here, and has since
been verified on real hardware (a Sapphire RX 470 8GB Mining UEFI) with
[`verify_gfx803.py`](#verifying-on-hardware).
