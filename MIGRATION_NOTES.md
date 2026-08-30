# ROCm 10.0 migration notes (this repo's main line)

Running investigation log for the rocm10 (root) line, written as findings
happen. The 7.14 line's log is `rocm7.14/MIGRATION_NOTES.md`; the 6.4.4
line's bug-hunting methodology record is `rocm6.4.4/KERNEL_BUGS.md`.

## 2026-08-29 -- TheRock 10.0 pin-bump, nothing built or verified yet

This line is a pin-bump of the hardware-verified 7.14 line
(`rocm7.14/`) to TheRock 10.0, keeping this repo's hand-rolled
per-component build pattern. The mainline repo (`rocm-migraphx-ort-builder`)
moved its release track to ROCm 10.0 (BASE_IMAGE `rocm/dev-ubuntu-26.04:
10.0.0-full`, MIGraphX `release/rocm-rel-10.0`, ORT `v1.29.0`, PyTorch
`2.14.0`) and this line now tracks the same defaults so gfx803 doesn't lag
the supported line it's split off from.

Pins (all branch pins, same policy as the 7.14 line -- release branch tips
at build time, not frozen SHAs):

| component | ref | confirmed to exist upstream |
|---|---|---|
| BASE_IMAGE | `rocm/dev-ubuntu-26.04:10.0.0-full` | mainline release.yml default |
| rocm-systems (ROCR/CLR) | `release/therock-10.0` | `git ls-remote` 2026-08-29 |
| rocm-libraries (rocBLAS/MIOpen) | `release/therock-10.0` | `git ls-remote` 2026-08-29 |
| MIGraphX | `release/rocm-rel-10.0` | `git ls-remote` 2026-08-29 |
| PyTorch | `release/2.14` | `git ls-remote` 2026-08-29 |
| torchvision | `release/0.28` (unchanged from 7.14; mainline maps torch 2.14 -> 0.28) | `git ls-remote` 2026-08-29 |
| torchaudio | `release/2.11.0.2` (unchanged from 7.14) | mainline torch-package-build-decide.sh |
| ORT | `v1.29.0` (bumped from 7.14's v1.28.0) | mainline ORT_VERSION default |

ORT bump rationale: the mainline repo's release track moved to v1.29.0 and
this line tracks its ORT default. ORT stays MIGraphX-EP-only -- `--use_rocm`
left build.py's flag set at 1.28 regardless, so the 6.4.4 line's ROCm-EP
fallback (v1.22.2) is unportable to any line newer than 6.4.4. See
`rocm7.14/MIGRATION_NOTES.md` for the full v1.28-vs-v1.22.2 history this
inherits.

## Open / unverified on this line (2026-08-29)

- **Nothing has been built.** The Dockerfile is a pin-bump; the patch set is
  copied verbatim from `rocm7.14/patches/` with no re-diff and no build
  against the 10.0 refs. Every patch driver is self-verifying (marker grep,
  exit 1), so a patch that no longer applies fails the build loudly instead
  of silently shipping -- but "applies cleanly" confirms nothing about
  correctness on 10.0 (or on any line; the recurring gfx803 bug class is
  silent miscompute).
- The whole 10.0 line needs the same validation arc the 7.14 line already
  completed: a successful build, then `verify.py` + `tools/correctness-suite/`
  + ORT backend-test series + real-model transcription runs against a real
  gfx803 card, before it can be called verified.
- MIOpen's `ConvOclDirectFwd` grouped-conv OOB fix is still NOT applied
  (solver removed upstream between 6.4.4 and 7.14, replaced by
  `ConvHipDirectFwd`); whether the same OOB exists in the new solver is
  unknown. Same state as the 7.14 line.
- The MLIR-stub-function gap check (`src/targets/gpu/{mlir,jit/mlir}.cpp`,
  `is_module_fusible`/`dump_mlir_to_*`) noted as un-verified on 7.14 is
  likewise un-verified on the 10.0 refs.

## 2026-08-29 (later) -- patch-set source validation against the 10.0 refs

Ran every patch driver against fresh sparse checkouts at the actual 10.0
refs (rocm-systems @ 6b0e43f3, rocm-libraries @ 8d1ae90e, AMDMIGraphX @
becdb3d). Findings:

- **4 of 7 rocm-systems patches apply as-is**: hsa-agent-rejects-legacy-
  doorbell, sdma-doorbell-missing-sfence (offset 14), va-reuse-defer (offset
  20), aql-ring-queue-full-workaround. Offsets are context shifts; the
  driver marker greps passed.
- **opencl-gfx8-hardcoded-rejection**: target still present but `device.hpp`
  moved up a directory (`device/rocm/device.hpp` -> `device/device.hpp`) and
  `populateOCLDeviceConstants()`'s image block shifted ~1382 -> ~1463.
  Re-diffed, intent preserved (image-query failures now degrade to
  disabled-image-support instead of aborting device init). Note: upstream
  already moved the imagePitchAlignment_/imageBaseAddressAlignment_/
  bufferFromImageSupport_ assignments out of the image_is_supported block;
  the re-diff keeps them where 10.0 has them rather than reverting that.
  Verified clean on a fresh 10.0 checkout.
- **graph-replay-batch-chunk-deadlock**: target present, `kPeriod =
  DEBUG_HIP_GRAPH_BATCH_SIZE` still unclamped at 10.0 (gfx803 deadlock
  mechanism unchanged), but the function moved to ~1624, its packet-buffer
  parameter type changed (`std::vector<uint8_t>` ->
  `amd::AlignedVector64<uint8_t>`), and upstream added a ramp-up/lead-chunk
  path below the kPeriod line. Re-diffed the same clamp. Verified clean on a
  fresh 10.0 checkout.
- **blit-kernel-eop-interrupt-retry**: DELIBERATELY NOT CARRIED to 10.0.
  The patch failed to apply (target code substantially evolved), and its own
  header's CORRECTION + SECOND CORRECTION sections document that the
  lost-EOP-interrupt premise was disproven -- the hang it mitigated is
  VRAM-clock marginality, fixed at the VBIOS level -- and that enabling the
  mitigation is actively harmful (`device lost from bus` cascade, full-box
  lockups; the README's "never set ROCR_GFX8_EOP_MITIGATION" guidance
  exists because of it). It was off by default anyway, so dropping it
  changes no default behavior. Removed the Dockerfile invocation and the
  patch files from this line's `patches/`; the 7.14 line keeps its own
  copy. Decision made 2026-08-29 after asking.

rocblas/miopen/migraphx patch validation was in progress when this note was
written; see the build + this log for results.

## 2026-08-29 (evening) -- full build + on-box regression results

### Build: SUCCESS

Full image built locally (24 cores, `TENSOR_TOPK_OPT_LEVEL=-O1`), pushed to
the gfx803 box (192.168.1.214, rootless podman) as `rocm-gfx803:rocm10`.
All final-stage import gates passed (ORT providers =
[MIGraphXExecutionProvider, CPUExecutionProvider], torch 2.14.0, torchvision
0.28.0, torchaudio 2.11.0.2).

Three build fixes landed during this run (beyond the patch re-diffs above):

1. **ORT flatbuffers collision**: ROCm 10.0's base image ships flatbuffers
   v25.9.23 in /opt/rocm. ORT's flatbuffers FetchContent uses
   `FIND_PACKAGE_ARGS 23.5.9` (a *minimum* version), so with
   CMAKE_PREFIX_PATH=/opt/rocm (MIGraphX provider) find_package picks the
   ROCm v25 config instead of downloading ORT's pinned v23.5.26, and the
   v25 headers fail ORT's generated-schema static_assert (`(25 == 23)`).
   Fixed in ort-builder by removing the whole ROCm flatbuffers footprint
   (headers + cmake config + lib + pkgconfig). The 7.14 base shipped no
   flatbuffers, which is why this never bit before.
2. **PyTorch 2.14 refuses `setup.py bdist_wheel`**: 2.14-2.15 deprecation
   makes every setup.py command except install/develop fail outright.
   Replaced with `python3 -m build --wheel --no-isolation` (+ `build`
   package in the venv). The mainline repo's source tier is unaffected
   because gfx900+ has a prebuilt-wheel tier that gfx803 never has.
3. **blit-kernel-eop-interrupt-retry dropped on this line** (see above).

### On-box hardware results

- Card enumerates as a real gfx803 KERNEL_DISPATCH agent (RX 570) on the
  10.0 runtime -- the legacy-doorbell restore works.
- MIGraphX EP inference correct (relu returns [0 2 0 4]).
- rocBLAS GEMM numerics correct (512/1024/2048 square, max rel err ~1e-6),
  with and without the sgemm shim.
- correctness-suite: **20/23**.
  - groupnorm "2 wrong" = self-labeled expected boundary rejects ("EXEC
    FAILED on rejected-boundary cases is expected, not a bug"). Not a
    regression.
  - pool_sweep GPU fault: ALSO reproduces on the box's 7.14 host stack
    (same deterministic ~39GB address range) -- box-level, not a 10.0
    regression. The box's documented intermittent VRAM-marginality fault
    class is still present here despite the correct 1750MHz VBIOS.
  - activ_sweep: one hipMalloc OOM mid-run (then continued OK). Needs
    re-check.
  - tensorop_sweep: 32/32 correct, then GPU fault at teardown/exit.

### Root-caused 10.0-line fault (reproducible 3/3)

Reproducible GPU fault ("Page not present", write from TC units, ~39GB VA)
under torch's memory-churn pattern (repeated large alloc + `.double()`
elementwise convert + `.cpu()` + free). Differential on the SAME card:

- 7.14 container (torch 2.8): passes the identical workload 3/3.
- 10.0 image (torch 2.14): faults 3/3.
- Pure HIP equivalent (hipMalloc + fp32->fp64 convert kernel + memcpy +
  free churn): passes on the 10.0 image.
- torch's randn fill loops and `.cpu()` copy loops pass on 10.0; only the
  elementwise `.double()` convert path under churn faults.

So the fault is in torch 2.14's gfx803 elementwise-convert path -- either a
torch-2.14 kernel bug or a hipcc-10.0 codegen regression for gfx803 --
NOT the patched ROCR/CLR/rocBLAS/MIOpen (which all validated clean). The
activ OOM and tensorop teardown crash are likely the same fault class and
should be re-checked once this one is understood.

### Arch differential: gfx803-SPECIFIC, not a general ROCm 10.0 issue

Re-ran the memory-churn fault (double_test) and GEMM numerics on the local
machine's gfx1201 (RX 9070 XT), using a scratch image built from the cached
rocm/dev-ubuntu-26.04:10.0.0-full base with AMD's pip-SDK torch devreleases
snapshot (torch 2.11.0 + devrocm10.0.0.dev, resolved by the mainline repo's
own rocm-devrelease-snapshot.py -- the arch test only needs a recent torch
on ROCm 10, and 2.11 is actually closer to the 7.14 line's 2.8 than 2.14):

| stack | GEMM numerics | memory-churn double_test |
|---|---|---|
| gfx803, ROCm 7.14, torch 2.8 (container) | correct | clean 3/3 |
| gfx803, ROCm 10.0, torch 2.14 (this line) | correct | **faults 3/3** |
| gfx1201, ROCm 10.0, torch 2.11 (pip-SDK) | correct | clean |

So the fault is gfx803-specific, NOT a general ROCm-10/torch issue (the
gfx1201 result rules that out). Still not yet split between the two gfx803
10.0-line variables: (a) torch 2.14's gfx803 elementwise kernels compiled by
hipcc-10.0, vs (b) the patched ROCR/CLR rebuilt against therock-10.0.
Evidence so far favors the torch kernel build: raw hipcc-10.0 convert-churn
kernels (hand-written HIP, same operation, same gfx803 box) run clean, so
hipcc-10.0 gfx803 output and the patched 10.0 runtime handle the operation
fine in general -- it is torch's *specific* elementwise-convert kernel
(TensorIterator launch config / vectorization) that faults. Torch 2.14 stays
(user requirement); the fix target is the torch gfx803 kernel build, not the
ROCm runtime or the rocBLAS/MIOpen/MIGraphX patches. Next step if pursued:
reduce the fault to the single kernel and compile it with the 7.14-era hipcc
to confirm the codegen hypothesis.

### UPDATE 2026-08-29 (late): torch 2.13 does NOT fix it -- fault is the 10.0 stack

Built a gfx803 10.0 image with PYTORCH_REF=release/2.13 (the 7.14 line's
proven torch, exactly what the mainline resolves for 2.13:
ROCm/pytorch + release/2.13). It builds fine (the `python -m build` command
handles both 2.13 and 2.14) and the same memory-churn double_test
**faults identically** (Page not present, ~39GB VA). So the fault is NOT
torch-version-specific -- the 10.0 gfx803 stack is the variable, as the
user suspected. Decision: keep torch 2.14 as the line default (it's the
mainline pin; 2.13 buys nothing).

Operation-level bisection on the gfx803 10.0 image: `.double()` alone
(convert) clean, `.cpu()` alone (D2H copy) clean, `.clone()` clean, but
`.double().cpu()` in a churn loop faults. So the fault lives in the
convert+D2H-copy combination under memory churn -- pointing at the
SDMA/D2H path (notable given patches/rocm-systems/sdma-doorbell-missing-
sfence.patch targets SDMA doorbell) or a convert-then-SDMA-read mapping
race, rather than the convert kernel alone. A hand-written hipcc-10.0 HIP
churn doing convert+hipMemcpy(D2H) with the same sizes passed once (single
run -- may also be intermittent), so this needs a real investigation:
prime suspects are the patched ROCR/CLR on 10.0 (va-reuse-defer or SDMA
interaction) or hipcc-10.0 codegen for the specific torch kernel. Not yet
split; a build with va-reuse-defer / sdma-sfence disabled would be the next
experiment, or compiling the isolated kernel with the 7.14 hipcc.

## Ref resolution method (recorded once, reused for all pins)

`git ls-remote --heads <repo> <ref>` for every branch pin above, run
2026-08-29. AMD bumped TheRock from 7.15 to 10.0 directly (no 8/9), so the
7.14 -> 10.0 jump is the same major-version skip the mainline repo's
"Nightly ROCm versioning" section documents; the release branches exist and
are the active lines.