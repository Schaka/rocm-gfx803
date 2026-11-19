# Triton patches for gfx803 (Polaris)

Support for running Triton on AMD Polaris (gfx803). Triton upstream has never
targeted pre-RDNA GCN. These patches are the wiring that makes it work. Each one
applies to triton-lang/triton `0a6e13e6e31f18fa75a17ac72dee0dd1bcb507b8`
(release/3.9.x). None of them ran on real hardware at that commit yet. See
"Verification" below.

## Files

| File | Purpose |
| --- | --- |
| `gfx803-isa-family.patch` | Adds `ISAFamily::GCN3` (gfx801-gfx810) to triton's AMD backend: the enum value, the family mapping, and the wave64 warp size. |
| `apply-gfx803-isa-family.sh` | Applies the patch and verifies it. |
| `fold-true-cmpi-while-nested-in-for-hang.patch` | Fixes a GPU hang. When a `while` loop sits inside a `for` loop with trip count 1, `TritonAMDFoldTrueCmpI` folds the exit condition of the `while` loop to a constant. The result is an infinite loop or a zero-iteration loop. The mechanism is not gfx803-specific. Only gfx803 hardware showed it. |
| `apply-fold-true-cmpi-while-nested-in-for-hang.sh` | Applies the patch and verifies it. Independent of the other two patches. |
| `gfx803-vdot-gate.patch` | Emits `llvm.amdgcn.fdot2`/`sdot4` only on targets that have v_dot. Without it, an fp16 or i8 `tl.dot` is a fatal LLVM abort on gfx803, and no vLLM engine can start. |
| `apply-gfx803-vdot-gate.sh` | Applies the patch and verifies it. Requires `gfx803-isa-family.patch` first. |

Apply order: apply `gfx803-isa-family.patch` before `gfx803-vdot-gate.patch`,
because the vdot gate's context keeps a line of the ISA-family patch. Apply
`fold-true-cmpi-while-nested-in-for-hang.patch` in any order.

This repo pins triton to one exact commit, not a branch. See `TRITON_REF` in
`docker-bake.hcl` for why. It mirrors PyTorch's own triton pin. All three patches
are diffed against that exact commit. Each header names the commit, states its
hardware verification status, and gives the repro that settles it. When the
pin moves, check that a header's WHY still holds in the new source before you
trust a re-diff that applies. Apply the same check to any other patch in this
repo.

## What the ISA-family patch does

Triton's AMD backend rejects any arch whose ISA family is Unknown with a hard
"unsupported target" error (TritonGPUToLLVM.cpp, ConvertWarpPipeline.cpp,
Membar.cpp). gfx803 is pre-RDNA and pre-CDNA, so it is Unknown. The patch:

1. `TargetFeatures.h`: adds `GCN3` to the `ISAFamily` enum.
2. `TargetFeatures.cpp`: `computeISAFamily()` maps every `gfx8xx` arch
   (GCN3/GCN4: Tonga, Fiji, Polaris) to `GCN3`.
3. `TargetFeatures.cpp`: `getWarpSize()` returns 64 for `GCN3` (wave64). Without
   this, the default of 32 is used, and kernels run half-dead.

Everything else follows from the defaults:

- `isCDNA()` and `isRDNA()` return false for `GCN3`, so it gets no MFMA and no
  WMMA lowering. Every `tl.dot` takes the FMA path.
- Buffer atomics, direct-to-LDS loads, permlane swaps and the other newer-arch
  features are off, because their predicates name only the families that have
  them.
- Warp reductions lower through triton's generic shuffle loop.
  `TargetInfo::warpReduce` uses its DPP sequence only on gfx90a and newer and on
  gfx11 and newer. For `GCN3`, the shuffle uses `ds_bpermute`, which GCN3 has.

The one emission that does not follow from the family is the packed v_dot cell
of the FMA path. Triton emits `llvm.amdgcn.fdot2` and `llvm.amdgcn.sdot4`
whenever a dot takes that path, whatever the family. On a target without v_dot,
that is not a slow path but a fatal one: LLVM finds no instruction to select and
aborts codegen, taking the whole compile with it. `gfx803-vdot-gate.patch` puts
those dots back on the plain FMA path. Its header states what that costs for i8
dots.

## Build requirements

`scripts/build/triton.sh` builds everything triton needs from source. Without
it, triton's own setup downloads a prebuilt LLVM and a prebuilt AMD
code-generation library.

### Two LLVMs

Triton links two separate LLVMs, and each pin comes from the triton checkout:

- `cmake/llvm-info.json` is the LLVM/MLIR that `libtriton` links. This is the
  same pin that triton's own `python/build_helpers.py` uses to download a
  prebuilt LLVM.
- `cmake/amd-llvm-info.json` is an LLVM that only the AMD code-generation
  library `libtriton_amd_codegen.so` links. The AMD backend hands its finished
  LLVM IR to that library through ctypes, and the library turns it into gfx ISA.
  gfx803's instruction selection, and so the LLVM abort that the vdot gate
  prevents, happens in this second LLVM.

Do not read either hash from `cmake/llvm-build-info.json` or
`cmake/amd-llvm-build-info.json`. Those files name what triton's own
`.github/workflows/llvm-build.yml` pre-builds. That can be the candidate for the
next LLVM bump, not the commit this source tree is written against. A build
against such a newer candidate failed with a real compile error in triton's MLIR
dialect headers.

The first LLVM follows the "llvm" package recipe of triton's `llvm-build.yml`:
the same project set, target list and distribution components, minus `not` and
`split-file`. Two flags are added on top, and neither one is optional:

- `LLVM_ENABLE_RTTI=ON`: triton's Python module compiles WITHOUT `-fno-rtti`
  (the flag applies only to `add_triton_library` targets). Without RTTI,
  `libLLVMSupport.a` lacks `llvm::cl::GenericOptionValue` typeinfo, and
  libtriton.so fails to import.
- `LLVM_ABI_BREAKING_CHECKS=FORCE_OFF`: with assertions ON, LLVM otherwise
  auto-enables ABI-breaking checks and emits `EnableABIBreakingChecks`. Triton
  expects the `Disable` symbol instead.

`LLVM_TARGETS_TO_BUILD` must include `NVPTX`, even though this stack has no
NVIDIA card. Triton's root CMakeLists links `LLVMNVPTXCodeGen` and MLIR's NVVM
dialect into the core triton library unconditionally, next to
`LLVMAMDGPUCodeGen`. That link is not only into the nvidia backend, which is
also always built and here unused. Read the pinned source before you trust this
list again.

The second LLVM follows the "amd-codegen" package recipe: no extra projects, the
AMDGPU target only, and headers, static libraries and CMake exports as the whole
distribution. The codegen library's CMake refuses any LLVM library that is not
static. The library links LLVM into itself and hides it, so that the two LLVMs
can live in one process. Neither of the two extra flags above applies to it: the
library compiles against this LLVM's headers alone and picks `-fno-rtti` and
`-fno-exceptions` to match how this LLVM was built.

### The AMD code-generation library

`triton.sh` builds `third_party/amd/backend/codegen` against the second LLVM,
with the same revision string triton's own build gives it
(`<amd llvm hash>-<build number>`). The AMD backend folds that string into every
kernel cache key. The script copies the library to
`third_party/amd/backend/lib/`, the path where triton's build helper puts its own
copy. Then it sets `TRITON_AMD_CODEGEN_PATH` for the wheel build. With that set,
the helper neither downloads nor copies a library, so the wheel packages the
self-built one.

Two checks make sure of it. The wheel build runs with an empty `TRITON_HOME`.
Setup keeps every AMD codegen download or bootstrap build under `.triton/amd`
in that directory. If anything is there after the build, the script stops. The
library inside the wheel must also match the self-built file byte for byte and
carry its revision string.

### Wheel verification

The end of `triton.sh` checks that each patch reached the wheel. All three
patches compile into `triton/_C/libtriton.so`. None reaches
`libtriton_amd_codegen.so`, whose only sources are `amdgpu_codegen.cc` and
`lib/Tools/LLVMOptions.cpp`.

| Patch | Marker in `libtriton.so` |
| --- | --- |
| `gfx803-isa-family.patch` | `GCN3` |
| `fold-true-cmpi-while-nested-in-for-hang.patch` | `isOwnWhileConditionOperand` |
| `gfx803-vdot-gate.patch` | `supportsVDot` |

Each marker is a name that exists only in the patched sources. It reaches the
.so through the debug info of triton's default build type (`-O2 -g`). The check
matches it as a whole line of `strings` output. A substring match is not enough,
because the LLVM AMDGPU code linked into libtriton carries symbols such as
`isGCN3Encoding` that contain the bare name.

## Runtime notes

- Backend discovery: when you install triton without entry points
  (hand-copied installs), set `TRITON_BACKENDS_IN_TREE=1`. Make sure
  `triton/backends/amd` and `triton/language/extra/hip` are real directories,
  not the editable-install's symlinks into the build tree.
- Leave `TRITON_AMD_CODEGEN_PATH` unset at runtime. When set, triton loads the
  code-generation library from that path instead of the one in
  `triton/backends/amd/lib/`.
- `torch.cuda.get_device_properties(0).gcnArchName` returns `gfx803` on real
  Polaris, so triton picks the right arch automatically.

## Verification

NOT YET RE-VERIFIED ON REAL HARDWARE on triton 3.9
(`0a6e13e6e31f18fa75a17ac72dee0dd1bcb507b8`). The patches apply exactly at that
commit, and nothing more is known there. Each patch header gives the repro that
must run on gfx803 against a build of this commit.

The last hardware results come from triton 3.8 (`675c59878`), on an RX 470 and
an RX 570, from the published `final` image:

- An add kernel JIT-compiled for gfx803, ran, and passed allclose.
- fp32 `tl.dot` GEMMs at 256x256x256, 512x512x512 and 1024x1024x1024 ran with
  exact results (maxerr 0.0). At BLOCK=64 the fp32 GEMM reached 0.589 TFLOPS
  against 0.518 TFLOPS for rocBLAS `torch.matmul`, about 12% of gfx803's
  ~5 TFLOPS peak.
- A 16x16 `tl.dot` per dtype against a float64 CPU reference, with the v_dot
  gate: fp16 max_err 1.19e-06, i8 max_err 0.0, fp32 max_err 1.69e-06, bf16
  max_err 4.77e-07. Without the gate, fp16 and i8 abort in LLVM.
- vLLM's `_topk_topp_kernel`, and a minimal 5-iteration `while` inside a
  `tl.range` loop with trip count 1, ran to completion with correct output,
  where they hang without the fold-true-cmpi patch.
- `vllm 0.20.1+gfx803` started an engine on Qwen3-0.6B and generated coherent
  text (greedy, 64 tokens, 46.6 tok/s).

## Known limitations

- The GCN3 family is grouped for feature purposes. All of gfx801-gfx810 are
  wave64, with no v_dot and no MFMA, so one family suffices. gfx803 itself is the
  only target tested.
- An i8 `tl.dot` on gfx803 accumulates in f32 and rounds once its accumulator
  exceeds 2^24 in magnitude. `gfx803-vdot-gate.patch` explains why.
- On triton 3.8, a bf16 run of Qwen3-0.6B produced incoherent output, identically
  with and without the v_dot gate. The cause is unidentified.
