# Triton patches for gfx803 (Polaris)

Hardware-verified support for running Triton on AMD Polaris (gfx803). Triton
upstream has never targeted pre-RDNA GCN; this is the wiring that makes it
work, measured on a real RX 470.

## Files

| File | Purpose |
| --- | --- |
| `gfx803-isa-family.patch` | Adds `ISAFamily::GCN3` (gfx801-gfx810) to triton's AMD backend: enum, family deduction, and wave64 warp size. 7 lines. |
| `apply-gfx803-isa-family.sh` | Applies the patch and verifies it. |
| `gfx803-dpp-broadcast-warpreduce.patch` | Routes GCN3 to the DPP row_bcast warpReduce path instead of the GFX10+-only `v_permlanex16` intrinsic. 1-line condition change. |
| `apply-gfx803-dpp-broadcast-warpreduce.sh` | Applies the patch and verifies it. Requires `gfx803-isa-family.patch` first. |
| `fold-true-cmpi-while-nested-in-for-hang.patch` | Fixes a real GPU hang (BACO reset, hardware-verified): a `while` loop nested inside a `for`-loop with trip count 1 gets its own exit condition folded to a constant by `TritonAMDFoldTrueCmpI`, producing an infinite (or zero-iteration) loop. Not gfx803-specific in mechanism, only proven on gfx803 hardware. |
| `apply-fold-true-cmpi-while-nested-in-for-hang.sh` | Applies the patch and verifies it. Independent of the other two patches. |

Apply order: `gfx803-isa-family.patch` before `gfx803-dpp-broadcast-warpreduce.patch`
(the latter's context assumes `ISAFamily::GCN3` exists);
`fold-true-cmpi-while-nested-in-for-hang.patch` is independent, any order.

## What the patch does

Triton's AMD backend rejects any arch whose `deduceISAFamily()` returns
`Unknown` with a hard "unsupported target" error (TritonGPUToLLVM.cpp:95,
ConvertWarpPipeline.cpp:361). gfx803 is pre-RDNA/pre-CDNA, so it was Unknown.
The patch:

1. `TargetUtils.h`: adds `GCN3` to the `ISAFamily` enum.
2. `TargetUtils.cpp`: `deduceISAFamily()` maps `GK_GFX801..GK_GFX810`
   (GCN3/GCN4: Tonga/Fiji/Polaris) to `GCN3`.
3. `TargetInfo.cpp`: `getWarpSize()` returns 64 for `GCN3` (wave64). Without
   this the default 32 is used and kernels run half-dead.

Feature flags are already correct by default for GCN3: `supportsVDot()`
returns false (gfx803 has no `v_dot4`), `isCDNA()`/`isRDNA()` return false
(no MFMA). Triton lowers `tl.dot` to FMA on this arch, which is correct —
the packed-dp4a trick from llama.cpp applies when a dot-product instruction
is needed.

## Build requirements (triton's LLVM must be built a specific way)

These were found by three rebuilds; they are not optional:

- `LLVM_ENABLE_RTTI=ON` — triton's pybind module compiles WITHOUT `-fno-rtti`
  (the flag only applies to `add_triton_library` targets); without RTTI the
  `libLLVMSupport.a` lacks `llvm::cl::GenericOptionValue` typeinfo and
  libtriton.so fails to import.
- `LLVM_ABI_BREAKING_CHECKS=FORCE_OFF` — asserts ON otherwise auto-enables
  ABI breaking checks, emitting `EnableABIBreakingChecks`; triton expects
  the `Disable` symbol.
- `LLVM_TARGETS_TO_BUILD="X86;AMDGPU"` — triton references
  `X86MCRegisterClasses` even for AMD-only work.
- `LLVM_ENABLE_ASSERTIONS=ON` — matches triton's `TritonRelBuildWithAsserts`.
- Copy `build/bin/FileCheck` to `install/bin/` — triton's CMake requires it.
- `LLVM_ENABLE_LLD=OFF` works if the host has no lld (bfd linker is fine).

## Runtime notes

- Backend discovery: use `TRITON_BACKENDS_IN_TREE=1` when installing triton
  without entry points (hand-copied installs), and make sure
  `triton/backends/amd` + `triton/language/extra/hip` are real directories,
  not the editable-install's symlinks into the build tree.
- `torch.cuda.get_device_properties(0).gcnArchName` returns `gfx803` on real
  Polaris, so triton picks the right arch automatically.

## Verification (real hardware, 2026-08-21)

- Add kernel JIT-compiled for gfx803, ran, allclose PASS.
- GEMM kernels (tl.dot, 3 shapes) ran with exact match (maxerr=0.0000).
- Naive BLOCK=64 fp32 GEMM: 0.589 TFLOPS vs rocBLAS torch.matmul 0.518 —
  triton slightly ahead, ~12% of gfx803's ~5 TFLOPS peak. The headroom is
  the vLLM card-specific tuning work (tile sizes, wave64-aware configs).

## Verification (real hardware, 2026-08-23) -- while-nested-in-for hang fix

- vLLM's `_topk_topp_kernel` (the kernel that surfaced the bug) went from
  a 100%-confirmed hardware hang (dmesg `ring gfx timeout` -> BACO reset)
  to running correctly on every tested grid/batch combination.
- Minimal structural repro (trivial 5-iteration while nested in a
  `tl.range` for-loop, no data dependency) went from hang to correct
  output.
- Full vLLM end-to-end generation with the triton top-k/top-p path
  enabled (no pytorch-fallback stopgap): coherent output, clean shutdown.

See `fold-true-cmpi-while-nested-in-for-hang.patch`'s own header for the
full root-cause writeup and `rocm-gfx803` repo `SESSION_HANDOFF.md`
section 9 for the session narrative.

## Known limitations

- The GCN3 family is grouped for feature purposes (all of gfx801-gfx810 are
  wave64/no-vdot/no-MFMA, so one family suffices). gfx803 itself is the
  only target tested.
- Direct-to-LDS, vectorized-atomic, and other newer-arch features are
  correctly disabled by the Unknown-equivalent defaults.