# Triton patches for gfx803 (Polaris)

Hardware-verified support for running Triton on AMD Polaris (gfx803). Triton
upstream has never targeted pre-RDNA GCN. This is the wiring that makes it
work, measured on a real RX 470.

## Files

| File | Purpose |
| --- | --- |
| `gfx803-isa-family.patch` | Adds `ISAFamily::GCN3` (gfx801-gfx810) to triton's AMD backend: enum, family deduction, and wave64 warp size. |
| `apply-gfx803-isa-family.sh` | Applies the patch and verifies it. |
| `gfx803-dpp-broadcast-warpreduce.patch` | Routes GCN3 to the DPP row_bcast warpReduce path instead of the GFX10+-only `v_permlanex16` intrinsic. 1-line condition change. |
| `apply-gfx803-dpp-broadcast-warpreduce.sh` | Applies the patch and verifies it. Requires `gfx803-isa-family.patch` first. |
| `fold-true-cmpi-while-nested-in-for-hang.patch` | Fixes a real GPU hang (BACO reset, hardware-verified). A `while` loop nested inside a `for`-loop with trip count 1 gets its own exit condition folded to a constant by `TritonAMDFoldTrueCmpI`. This produces an infinite loop, or a zero-iteration loop. The bug is not gfx803-specific in mechanism, only proven on gfx803 hardware. |
| `apply-fold-true-cmpi-while-nested-in-for-hang.sh` | Applies the patch and verifies it. Independent of the other two patches. |
| `gfx803-vdot-gate.patch` | Emissions of `llvm.amdgcn.fdot2`/`sdot4` are gated on the target actually having v_dot. Without it an fp16 `tl.dot` is a fatal LLVM abort on gfx803, and no vLLM engine can start. |
| `apply-gfx803-vdot-gate.sh` | Applies the patch and verifies it. Requires `gfx803-isa-family.patch` first. |

Apply order: apply `gfx803-isa-family.patch` before
`gfx803-dpp-broadcast-warpreduce.patch` and before `gfx803-vdot-gate.patch`,
because both of those patches' context assumes `ISAFamily::GCN3` already
exists. Apply `fold-true-cmpi-while-nested-in-for-hang.patch` in any order,
because it is independent of the other three.

This repo pins triton to one exact commit, not a branch. See `TRITON_REF` in
`docker-bake.hcl` for why. It mirrors PyTorch's own triton pin. All four
patches are re-diffed against that exact commit. Each patch header's RE-DIFF
section says what moved since the patch was first written and
hardware-verified. Check that the header's WHY still holds before you trust a
future re-diff. Apply the same check to any other patch in this repo.

## What the patch does

Triton's AMD backend rejects any arch whose `TargetFeatures::getISAFamily()`
returns `Unknown` with a hard "unsupported target" error
(TritonGPUToLLVM.cpp, ConvertWarpPipeline.cpp). gfx803 is pre-RDNA/pre-CDNA,
so it was Unknown. The patch:

1. `TargetFeatures.h`: adds `GCN3` to the `ISAFamily` enum.
2. `TargetFeatures.cpp`: `getISAFamily()` maps every `gfx8xx` arch
   (GCN3/GCN4: Tonga/Fiji/Polaris) to `GCN3`.
3. `TargetFeatures.cpp`: `getWarpSize()` returns 64 for `GCN3` (wave64).
   Without this, the default of 32 is used, and kernels run half-dead.
4. `TargetFeatures.cpp`: `supportDppBroadcast()` returns true for `GCN3`, so
   warpReduce's cross-row broadcast step uses the DPP `row_bcast:15/31`
   lowering instead of the GFX10+-only `v_permlanex16` intrinsic (see
   `gfx803-dpp-broadcast-warpreduce.patch`'s own header).

Feature flags are already correct by default for GCN3. `isCDNA()` and
`isRDNA()` return false, so GCN3 gets no MFMA, and the `v_dot`-gated choices
that do ask the family stay off. What does not stay off by itself is the packed
intrinsic emission: `lm.amdgcn.fdot2` and `llvm.amdgcn.sdot4` are emitted
whenever a dot takes the FMA path, whatever the family, and on a target without
v_dot that is not a slow path but a fatal one -- LLVM finds no instruction to
select and aborts codegen, taking the whole compile with it. So triton does not
lower every `tl.dot` to FMA on this arch by default: an fp16 dot with f32
accumulation goes to those intrinsics, and `gfx803-vdot-gate.patch` is what puts
it back on the FMA path. See that patch's header for the hardware measurements.

## Build requirements (triton's LLVM must be built a specific way)

`scripts/build/triton.sh` builds triton's own pinned LLVM commit from source.
The pin comes from `cmake/llvm-info.json` in the triton checkout. This is the
same pin that triton's own `python/build_helpers.py` uses to download a
prebuilt LLVM for `setup.py`. The script builds that commit from source
instead of downloading it, using the same project set and distribution
component list as triton's own `.github/workflows/llvm-build.yml` recipe.

Do not read the LLVM hash from `cmake/llvm-build-info.json`. That file holds a
separate, newer pin. Triton uses it only to pre-build its next LLVM version
ahead of a future release. A build against that newer commit produced a real
compile error, not a stale patch. Its
`mlir/Interfaces/SideEffectInterfaces.h` had already dropped
`Resource::getName() const` and `Resource::getParent()`. This triton commit's
own `Dialect.h` still needs both methods.

Two flags are added on top of the llvm-build.yml recipe. Three rebuilds found
both, and neither one is optional:

- `LLVM_ENABLE_RTTI=ON`: triton's pybind module compiles WITHOUT `-fno-rtti`
  (the flag applies only to `add_triton_library` targets). Without RTTI,
  `libLLVMSupport.a` lacks `llvm::cl::GenericOptionValue` typeinfo, and
  libtriton.so fails to import.
- `LLVM_ABI_BREAKING_CHECKS=FORCE_OFF`: with assertions ON, LLVM otherwise
  auto-enables ABI-breaking checks and emits `EnableABIBreakingChecks`.
  Triton expects the `Disable` symbol instead.

`LLVM_TARGETS_TO_BUILD` must include `NVPTX`, even though this stack has no
NVIDIA card. Triton's root CMakeLists links `LLVMNVPTXCodeGen` and MLIR's
NVVM dialect into the core triton library unconditionally. This is confirmed
by reading CMakeLists.txt at the pinned commit. The link is not only into the
nvidia backend, which is also always built and here unused. An earlier
version of this note said
`X86;AMDGPU` was sufficient. That was accurate for whatever triton commit was
pinned at the time, not for the one pinned now. Read the pinned source before
you trust this list again.

## Runtime notes

- Backend discovery: when you install triton without entry points
  (hand-copied installs), set `TRITON_BACKENDS_IN_TREE=1`. Make sure
  `triton/backends/amd` and `triton/language/extra/hip` are real directories,
  not the editable-install's symlinks into the build tree.
- `torch.cuda.get_device_properties(0).gcnArchName` returns `gfx803` on real
  Polaris, so triton picks the right arch automatically.

## Verification (real hardware, 2026-08-21)

- Add kernel JIT-compiled for gfx803, ran, allclose PASS.
- GEMM kernels (tl.dot, 3 shapes) ran with exact match (maxerr=0.0000).
- Naive BLOCK=64 fp32 GEMM: 0.589 TFLOPS vs rocBLAS torch.matmul's 0.518
  TFLOPS. Triton is slightly ahead, at about 12% of gfx803's ~5 TFLOPS peak.
  The remaining headroom is vLLM's card-specific tuning work: tile sizes and
  wave64-aware configs.

## Verification (real hardware, 2026-08-23): while-nested-in-for hang fix

- vLLM's `_topk_topp_kernel` (the kernel that surfaced the bug) went from
  a 100%-confirmed hardware hang (dmesg `ring gfx timeout` -> BACO reset)
  to running correctly on every tested grid/batch combination.
- Minimal structural repro (trivial 5-iteration while nested in a
  `tl.range` for-loop, no data dependency) went from hang to correct
  output.
- Full vLLM end-to-end generation with the triton top-k/top-p path
  enabled (no pytorch-fallback stopgap): coherent output, clean shutdown.

See `fold-true-cmpi-while-nested-in-for-hang.patch`'s own header for the
full root-cause writeup.

## Verification (real hardware, 2026-09-07): re-diffed patches, shipped image

The two passes above predate the 2026-09-07 re-diff and ran against the
original file layout. This pass closes that gap. The `triton` Docker Bake
stage built the re-diffed patches, and the `final` image shipped them.
Every test below ran from that `final` image on the real RX 470.

- Add kernel JIT-compiled for gfx803, ran, allclose PASS.
- GEMM kernels (tl.dot, 3 shapes: 256x256x256, 512x512x512, 1024x1024x1024)
  ran with exact match (maxerr=0.0000) on every shape.
- `torch.cuda.get_device_properties(0).gcnArchName` reported `gfx803` inside
  the `final` image's own venv. Triton saw the same arch string as the rest
  of the stack.
- The while-nested-in-for minimal structural repro (5-iteration while, no
  data dependency, nested in a `tl.range` for-loop with trip count 1) ran
  to completion with the correct output. No hang.

This closes the re-diff gap for the three patches re-diffed at that
point, including the while-nested-in-for fix.

## Verification (real hardware, 2026-09-12): v_dot gate on the pinned triton

Found while validating the published `final` image on the card: `vllm
0.20.1+gfx803` could not start an engine at all, because the ROCm attention
backend's prefill kernel is an fp16 dot with f32 accumulation and triton emitted
`llvm.amdgcn.fdot2` for it. LLVM has no instruction to select, and aborts instead
of falling back:

    LLVM ERROR: Cannot select: t1002: f32 = AMDGPUISD::FDOT2
    In function: _fwd_kernel

Per-dtype 16x16 `tl.dot`, compiled and run on the RX 570, GPU result against a
float64 CPU reference, before and after `gfx803-vdot-gate.patch`:

| dtype | before | after |
| --- | --- | --- |
| fp16 | LLVM abort (`FDOT2`) | OK, max_err 1.19e-06 |
| i8 | LLVM abort (`sdot4`) | OK, max_err 0.0 |
| fp32 | OK | OK, max_err 1.69e-06 |
| bf16 | OK | OK, max_err 4.77e-07 |

That table also states what the 2026-09-07 pass above did not cover: its `tl.dot`
GEMM shapes cannot have been fp16, because an fp16 dot on that image aborts.
fp32 and bf16 dots both reach the f32 FMA path, so both passed. fp16 is the one
every attention and prefill kernel uses.

End to end, on the same image with only triton replaced by the build carrying
this patch: the engine starts and generates coherent text (greedy, 64 tokens,
46.6 tok/s) where it previously died in core init with "Engine core
initialization failed".

Two limits on this pass. The wheel was built against triton's prebuilt LLVM for
the same `cmake/llvm-info.json` pin and with gcc, not through CI's self-built
LLVM and clang, so only a CI rebuild of the `triton` and `final` images makes a
published `latest-gfx803` carry a verified version of this patch. And a bf16 run
of the same model, prompt and settings produced incoherent output, identically
before and after this patch -- a separate, still unidentified problem that this
patch neither causes nor fixes.

## Known limitations

- The GCN3 family is grouped for feature purposes (all of gfx801-gfx810 are
  wave64/no-vdot/no-MFMA, so one family suffices). gfx803 itself is the
  only target tested.
- Direct-to-LDS, vectorized-atomic, and other newer-arch features are
  correctly disabled by the Unknown-equivalent defaults.