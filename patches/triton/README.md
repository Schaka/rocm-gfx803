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

Apply order: apply `gfx803-isa-family.patch` before
`gfx803-dpp-broadcast-warpreduce.patch`, because the second patch's context
assumes `ISAFamily::GCN3` already exists. Apply
`fold-true-cmpi-while-nested-in-for-hang.patch` in any order, because it is
independent of the other two.

This repo pins triton to one exact commit, not a branch. See `TRITON_REF` in
`docker-bake.hcl` for why. It mirrors PyTorch's own triton pin. All three
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
`isRDNA()` return false, so GCN3 gets no MFMA. Every `v_dot`-gated path in
`AccelerateAMDMatmul.cpp` is reached only through `isCDNA()`/`isCDNA4()`, so
it stays off. Triton lowers `tl.dot` to FMA on this arch. That is correct. When a
dot-product instruction is needed, the packed-dp4a trick from llama.cpp
applies instead.

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

This closes the re-diff gap for all three patches, including the
while-nested-in-for fix.

## Known limitations

- The GCN3 family is grouped for feature purposes (all of gfx801-gfx810 are
  wave64/no-vdot/no-MFMA, so one family suffices). gfx803 itself is the
  only target tested.
- Direct-to-LDS, vectorized-atomic, and other newer-arch features are
  correctly disabled by the Unknown-equivalent defaults.