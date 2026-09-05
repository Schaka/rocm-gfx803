# ROCm 10.0 migration notes (this repo's main line)

This is the running investigation log for the rocm10 (root) line, written as
findings happen. Each entry keeps its date, because later entries correct earlier
ones. The 7.14 line's log is `rocm7.14/MIGRATION_NOTES.md`. The 6.4.4 line's
bug-hunting method and record is `rocm6.4.4/KERNEL_BUGS.md`. Current status lives
in `README.md`, not here.

## 2026-08-29: TheRock 10.0 pin bump, nothing built or verified yet

This line is a pin bump of the hardware-verified 7.14 line (`rocm7.14/`) to
TheRock 10.0. It keeps this repo's hand-built per-component pattern. The mainline
repo (`rocm-migraphx-ort-builder`) moved its release track to ROCm 10.0. Its
defaults are BASE_IMAGE `rocm/dev-ubuntu-26.04:10.0.0-full`, MIGraphX
`release/rocm-rel-10.0`, ORT `v1.29.0`, and PyTorch `2.14.0`. This line tracks the
same defaults, so gfx803 does not lag the supported line it came from.

Pins. Each one is a branch pin, which is the same policy the 7.14 line uses: the
release branch tip at build time, and not a frozen SHA.

| component | ref | how the ref was confirmed |
|---|---|---|
| BASE_IMAGE | `rocm/dev-ubuntu-26.04:10.0.0-full` | mainline release.yml default |
| rocm-systems (ROCR/CLR) | `release/therock-10.0` | `git ls-remote` 2026-08-29 |
| rocm-libraries (rocBLAS/MIOpen) | `release/therock-10.0` | `git ls-remote` 2026-08-29 |
| MIGraphX | `release/rocm-rel-10.0` | `git ls-remote` 2026-08-29 |
| PyTorch | `release/2.14` | `git ls-remote` 2026-08-29 |
| torchvision | `release/0.28`, unchanged from 7.14, because mainline maps torch 2.14 to 0.28 | `git ls-remote` 2026-08-29 |
| torchaudio | `release/2.11.0.2`, unchanged from 7.14 | mainline torch-package-build-decide.sh |
| ORT | `v1.29.0`, bumped from 7.14's v1.28.0 | mainline ORT_VERSION default |

Reason for the ORT bump: the mainline release track moved to v1.29.0, and this
line tracks that default. ORT stays MIGraphX-EP only. `--use_rocm` left build.py's
flag set at 1.28 in any case, so the 6.4.4 line's ROCm-EP fallback (v1.22.2) cannot
be ported to any line newer than 6.4.4. `rocm7.14/MIGRATION_NOTES.md` holds the
v1.28 and v1.22.2 history that this inherits.

## 2026-08-29: open questions at that time

- Nothing had been built. The Dockerfile was a pin bump. The patch set was copied
  verbatim from `rocm7.14/patches/`, with no re-diff and no build against the 10.0
  refs. Every patch driver checks its own result (marker grep, then `exit 1`), so a
  patch that no longer applies fails the build loudly instead of shipping
  unpatched code. A clean apply still says nothing about correctness. The common
  gfx803 bug class is a silent wrong answer.
- The whole line needed the validation arc that the 7.14 line had already
  completed: a successful build, then `verify.py`, `tools/correctness-suite/`, the
  ORT backend-test series, and real-model transcription runs on a real gfx803 card.
- MIOpen's `ConvOclDirectFwd` grouped-conv out-of-bounds fix was still not applied.
  Upstream removed that solver between 6.4.4 and 7.14 and replaced it with
  `ConvHipDirectFwd`. Whether the same out-of-bounds read exists in the new solver
  was unknown. The 7.14 line was in the same state.
- The MLIR stub function gap check (`src/targets/gpu/{mlir,jit/mlir}.cpp`,
  `is_module_fusible`, and `dump_mlir_to_*`), noted as unverified on 7.14, was also
  unverified against the 10.0 refs.

## 2026-08-29 (later): patch-set check against the 10.0 refs

Every patch driver was run against fresh sparse checkouts at the real 10.0 refs:
rocm-systems at 6b0e43f3, rocm-libraries at 8d1ae90e, and AMDMIGraphX at becdb3d.
Findings:

- 4 of 7 rocm-systems patches applied as-is: hsa-agent-rejects-legacy-doorbell,
  sdma-doorbell-missing-sfence (offset 14), va-reuse-defer (offset 20), and
  aql-ring-queue-full-workaround. The offsets are context shifts, and the driver
  marker greps passed.
- opencl-gfx8-hardcoded-rejection: the target was still present, but `device.hpp`
  moved up one directory (`device/rocm/device.hpp` to `device/device.hpp`) and the
  image block in `populateOCLDeviceConstants()` moved from about line 1382 to about
  line 1463. It was re-diffed with the intent kept: an image-query failure now
  degrades to disabled image support instead of aborting device init. Upstream had
  also moved the imagePitchAlignment_, imageBaseAlignment_ and
  bufferFromImageSupport_ assignments out of the image_is_supported block. The
  re-diff keeps them where 10.0 puts them, and does not revert that. It applies
  clean on a fresh 10.0 checkout.
- graph-replay-batch-chunk-deadlock: the target was present, and `kPeriod =
  DEBUG_HIP_GRAPH_BATCH_SIZE` was still unclamped at 10.0, so the gfx803 deadlock
  mechanism is unchanged. The function moved to about line 1624, its packet-buffer
  parameter type changed from `std::vector<uint8_t>` to
  `amd::AlignedVector64<uint8_t>`, and upstream added a ramp-up and lead-chunk path
  below the kPeriod line. The same clamp was re-diffed. It applies clean on a fresh
  10.0 checkout.
- blit-kernel-eop-interrupt-retry: deliberately not carried to 10.0. The patch
  failed to apply, because the target code had evolved a lot. Its own header's
  CORRECTION and SECOND CORRECTION sections record that the lost-EOP-interrupt
  premise was disproven. The hang it mitigated was VRAM clock marginality, and the
  VBIOS fix handles it. The mitigation is also actively harmful: a `device lost from
  bus` cascade and full-box lockups. The README's rule to never set
  ROCR_GFX8_EOP_MITIGATION exists because of that. It was off by default anyway, so
  dropping it changes no default behavior. The Dockerfile call and the patch files
  were removed from this line's `patches/`. The 7.14 line keeps its own copy. This
  decision was made 2026-08-29, after asking.

rocBLAS, MIOpen, and MIGraphX patch checks were still in progress when this entry
was written. The build output and later entries carry the results.

## 2026-08-29 (evening): full build and on-box regression results

### Build: success

The full image was built locally (24 cores, `TENSOR_TOPK_OPT_LEVEL=-O1`) and sent
to the gfx803 box (192.168.1.214, rootless podman) as `rocm-gfx803:rocm10`. Every
final-stage import gate passed: ORT providers
[MIGraphXExecutionProvider, CPUExecutionProvider], torch 2.14.0, torchvision
0.28.0, and torchaudio 2.11.0.2.

Three build fixes landed during this run, besides the re-diffs above:

1. ORT and flatbuffers collided. The ROCm 10.0 base image ships flatbuffers
   v25.9.23 in /opt/rocm. ORT's flatbuffers FetchContent uses
   `FIND_PACKAGE_ARGS 23.5.9`, which is a minimum version. With
   CMAKE_PREFIX_PATH=/opt/rocm (the MIGraphX provider sets it), find_package takes
   the ROCm v25 config instead of downloading ORT's pinned v23.5.26, and the v25
   headers fail ORT's generated-schema static_assert (`(25 == 23)`). The fix in
   ort-builder removes the whole ROCm flatbuffers footprint: headers, cmake config,
   library, and pkgconfig. The 7.14 base shipped no flatbuffers, which is why this
   never appeared before.
2. PyTorch 2.14 refuses `setup.py bdist_wheel`. The 2.14 and 2.15 deprecation makes
   every setup.py command fail except install and develop. It was replaced with
   `python3 -m build --wheel --no-isolation`, and the `build` package was added to
   the venv. The mainline repo's source tier is not affected, because gfx900 and
   newer have a prebuilt-wheel tier that gfx803 never has.
3. blit-kernel-eop-interrupt-retry was dropped on this line, as recorded above.

### On-box hardware results

- The card enumerates as a real gfx803 KERNEL_DISPATCH agent (RX 570) on the 10.0
  runtime, so the legacy-doorbell restore works.
- MIGraphX EP inference is correct (relu returns [0 2 0 4]).
- rocBLAS GEMM numerics are correct (512, 1024, and 2048 square, max relative error
  about 1e-6), with and without the sgemm shim.
- The correctness suite gave 20 of 23 clean.
  - groupnorm reported 2 wrong, and both are self-labeled expected boundary
    rejects: "EXEC FAILED on rejected-boundary cases is expected, not a bug". Not a
    regression.
  - pool_sweep GPU fault also reproduces on the box's 7.14 host stack, in the same
    deterministic address range near 39 GB. It is box-level, and not a 10.0
    regression. The box's documented intermittent VRAM-marginality fault class is
    still present here, despite the correct 1750 MHz VBIOS.
  - activ_sweep hit one hipMalloc out-of-memory during the run, then continued
    clean. It needed a re-check.
  - tensorop_sweep gave 32 of 32 correct, then faulted at teardown or exit.

### One fault root-caused on this line (reproducible 3/3)

A GPU fault ("Page not present", write from TC, at a VA near 39 GB) appeared under
torch's memory-churn pattern: repeated large allocations, a `.double()` elementwise
convert, a `.cpu()`, then free. The differential was run on the same card:

- The 7.14 container (torch 2.8) passes the identical workload 3/3.
- The 10.0 image (torch 2.14) faults 3/3.
- A pure HIP equivalent (hipMalloc, an fp32 to fp64 convert kernel, memcpy, and
  free churn) passes on the 10.0 image.
- torch's randn fill loops and `.cpu()` copy loops pass on 10.0. Only the
  elementwise `.double()` convert path under churn faults.

That put the fault in torch 2.14's gfx803 elementwise-convert path: either a torch
2.14 kernel fault or a hipcc 10.0 codegen regression for gfx803. The patched ROCR,
CLR, rocBLAS, and MIOpen were clear, because all of them validated clean. The
activ_sweep out-of-memory and the tensorop teardown crash were probably the same
fault class, and were to be re-checked once this one was understood.

### Architecture differential: gfx803-specific, not a general ROCm 10.0 problem

The memory-churn fault (double_test) and the GEMM numerics were rerun on the local
machine's gfx1201 (RX 9070 XT). The test image was built from the cached
rocm/dev-ubuntu-26.04:10.0.0-full base with the pip SDK torch devrelease snapshot
(torch 2.11.0 with devrocm10.0.0.dev, resolved by the mainline repo's own
rocm-devrelease-snapshot.py). The architecture test only needs a recent torch on
ROCm 10, and 2.11 sits closer to the 7.14 line's 2.8 than 2.14 does.

| stack | GEMM numerics | memory-churn double_test |
|---|---|---|
| gfx803, ROCm 7.14, torch 2.8 (container) | correct | clean 3/3 |
| gfx803, ROCm 10.0, torch 2.14 (this line) | correct | faults 3/3 |
| gfx1201, ROCm 10.0, torch 2.11 (pip SDK) | correct | clean |

So the fault is gfx803-specific and not a general ROCm 10 or torch problem, which
the gfx1201 result rules out. It was not yet split between the two remaining gfx803
10.0 variables: torch 2.14's gfx803 elementwise kernels compiled by hipcc 10.0, or
the patched ROCR and CLR rebuilt against therock-10.0. The evidence then favored
the torch kernel build, because hand-written hipcc 10.0 convert-churn kernels, the
same operation on the same gfx803 box, run clean. So hipcc 10.0 gfx803 output and
the patched 10.0 runtime handle the operation in general, and torch's specific
elementwise-convert kernel (its TensorIterator launch config and vectorization) is
what faults. Torch 2.14 stays, because it is a user requirement. The fix target is
the torch gfx803 kernel build, and not the ROCm runtime or the rocBLAS, MIOpen, and
MIGraphX patches. The next step, if pursued, was to reduce the fault to one kernel
and compile it with the 7.14-era hipcc, to confirm the codegen idea.

## 2026-08-30: RESOLVED, root cause found and fixed (d2h-staged-copy plus va-reuse-defer-mapping)

The operation-level bisection (`.double().cpu()` in a churn loop faults, and each
operation alone is clean) led to a full root cause on the gfx803 box.

1. It is not a 10.0 regression, and it is not tied to a torch version. The same
   `.double().cpu()` churn fault reproduces 5/5 on the real published 7.14 image
   (ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx803, torch
   2.13.0+git18c52f2, which is the exact commit the 10.0 experiment used). The
   "7.14 clean" baseline images on the box were in fact ROCm 6.4.4 (torch 2.8). The
   fault appears in the 6.4.4 to 7.14 transition, and 10.0 inherited it unchanged.
   It is not a kernel-codegen problem: compute kernels that write the same host
   buffer, scalar and vectorized, are clean. Only the copy engine's direct host
   write faults.
2. Mechanism. The fault address is the GPUVM mapping of the registered or pinned
   host buffer, and the fault says "write from TC". The D2H copy engine, blit shader
   or SDMA, writes the pinned host destination directly. On gfx803 the engine's
   stores to host memory drain from L2 into the host buffer after the copy's
   completion signal. A system-scope release fence does not change that on this
   hardware. This was measured: it faults with the shader path forced, with the SDMA
   path forced, and with a hipDeviceSynchronize before the unlock. So the write-back
   races the host buffer's unlock and teardown, and faults on the removed mapping.
   The 6.4.4 line escaped only because it recycled the VA fast enough that the
   write-back landed on a page that had been mapped again. The 7.14 and later VA
   reuse defer removes that accidental tolerance. The minimal repro is: register a
   pageable buffer with hipHostRegister, hipMemcpy D2H into it, unregister it, and
   repeat with device allocation and free churn. It faults within a few rounds.
3. Fix, verified on hardware. Force the GPU-staging D2H path for every copy whose
   destination is host memory. The copy then writes a GPU staging buffer, whose
   mapping fmm keeps alive, and a CPU memcpy lands the data, so no engine write is
   ever outstanding against the host buffer's lifetime. This is the same path that
   GPU_PINNED_MIN_XFER_SIZE=4096 forces, and that path was already proven clean. It
   is implemented as patches/rocm-systems/d2h-staged-copy.patch: kEnablePin=false in
   both `DmaBlitManager::readBuffer` and `KernelBlitManager::readBuffer`, plus
   routing registered and locked host destinations through the staged readBuffer in
   both `submitReadMemory` and `copyMemory`. It costs one extra copy, and that is
   the right trade, because the direct host write is broken on this hardware.
4. The va-reuse-defer's park contract was also broken, and is fixed by
   patches/rocm-systems/va-reuse-defer-mapping.patch. The defer parks a freed object
   and withholds the VA, but hsa-runtime's `KfdDriver::FreeMemory` calls
   `hsaKmtUnmapMemoryToGPU` before `hsaKmtFreeMemory`. So parked objects ended up
   reserved and unmapped, and the park's promise to keep the mapping alive was
   false. parktest showed it: a write to a freed but parked VA faults. The patch
   re-establishes the mapping in the park branch. This is independent of the D2H
   fault, because parktest now passes on 10.0, but it is the same "mapping alive
   through the window" contract.

Both patches applied and were measured on the 10.0 line. d2h-staged-copy and
va-reuse-defer-mapping were also applied, and source-checked, on the 7.14 line. The
10.0 on-hardware results, from a spliced test image: dblcpu churn 8/8 clean,
regchurn 8/8 clean, parktest passes, dcheck 8/8 data-correct, GEMM numerics about
1e-6, reduce_extreme_sweep 32/32 (8 expected rejects), and groupnorm sweep boundary
rejects as expected. A full rebuild of the 10.0 image with the new patches was the
next step, and the 7.14 line needed a build and hardware rerun.

## How a ref was resolved (recorded once, used for every pin)

`git ls-remote --heads <repo> <ref>` was run for every branch pin above, on
2026-08-29. AMD bumped TheRock from 7.15 to 10.0 directly, with no 8 or 9, so the
7.14 to 10.0 jump is the same major-version skip that the mainline repo's "Nightly
ROCm versioning" section documents. The release branches exist and are the active
lines.
