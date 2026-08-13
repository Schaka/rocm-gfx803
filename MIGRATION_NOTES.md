# gfx803 on ROCm 7.14 -- migration notes

What it took to port this line from ROCm 6.4.4 to 7.14: patches carried,
dropped, or added, and the build-arg/tooling changes that came with them.
Not a bug tracker or a validation log -- correctness findings and open
investigations live elsewhere.

## Version pin resolution

ROCm 7.14 has no per-component release tags the classic way 6.4.4 did.
Confirmed via `git ls-remote --tags`:

- `ROCm/ROCR-Runtime`, `ROCm/rocm-libraries`, `ROCm/clr`: tags stop at
  `rocm-7.2.4`. No `rocm-7.14` tag exists anywhere.
- `ROCm/TheRock`: has `therock-7.14`. `version.json` at that tag reads
  `"rocm-version": "7.14.0"` -- this is what "ROCm 7.14" actually refers
  to now: a TheRock meta-release, not a per-repo tag.
- Docker Hub: `rocm/dev-ubuntu-26.04` has a `7.14.0-full` tag (no
  `-complete` variant exists for 7.14; that was the old per-repo-tag-build
  naming, `-full` is TheRock's). Confirmed to have the classic
  `/opt/rocm/{bin,lib,include,llvm}` symlink layout (via
  `/etc/alternatives/`), a drop-in for every path this build's stages
  already hardcode.
- TheRock's `.gitmodules` at `therock-7.14` pins `rocm-libraries` and
  `rocm-systems` by exact commit SHA (not branch), since both track
  `develop` in the module URL. Resolved via `git ls-tree HEAD
  rocm-libraries rocm-systems`:
  - `rocm-libraries` @ `cd9574023093742434e8c992d13b89ab9a6c1cf8`
  - `rocm-systems` @ `2b22ab0195cc1461cd9abf3b969e9dd7c10af350`

  Both used directly (not via TheRock's submodule machinery -- a blobless
  sparse clone straight from the upstream repo at the pinned SHA gets the
  same source at a fraction of the size/time).
- MIGraphX stayed a standalone repo (not folded into rocm-libraries) and
  does publish its own tags: `rocm-7.14` exists on `ROCm/AMDMIGraphX`.
- These pins match the main (gfx900+) repo's own manual release track:
  `.github/workflows/release.yml`'s default `rocm_version` is already
  `7.14.0`, `base-image` is computed as
  `rocm/dev-ubuntu-26.04:${rocm_version}-full`, and `pytorch_version`
  defaults to `2.13.0`.

**Re-pinning later**: read `therock-7.14`'s `.gitmodules`-pinned commits
again (or whatever the next `therock-7.1X` tag is), don't just bump a
version string -- "ROCm 7.14.0" the docker tag and the exact
rocm-libraries/rocm-systems state it was built from can drift
independently until you re-resolve them together.

## Tooling: `git apply` vs. `patch` on this box

Every `.sh` wrapper in `patches/` uses `patch -p1`, not `git apply`,
unlike the 6.4.4-era scripts. Reason: `git apply --check` (git 2.55.0)
reproducibly printed `Skipped patch '<file>'.` and exited **0** without
modifying anything, for at least one patch independently confirmed
correct (content and placement verified by hand, applies cleanly via
`patch -p1 --dry-run` with an exact line match). A git-apply-specific
defect or overly-lenient heuristic on this git version, not a defect in
the affected patches.

## Layer 1: ROCR-Runtime + CLR (new for this line, no 6.4.4 equivalent)

Real dispatch, not just enumeration, needs a 4-file, ~76-line change, not
a single guarded exception.

**Source of truth for the fix**: restored verbatim from an AMD engineer's
(lucbruni-amd) fork, `lucbruni-amd/TheRock@lb/gfx803-polaris-support`,
commit `3d4ad60`. That commit has two patches (`patches/rocm6.4.4/rocm-systems/
0001-...` and `0002-...`), both re-diffed and validated against this
build's own pinned `rocm-systems` commit (`2b22ab0195...`, not lucbruni's
fork's) -- `git apply --check` / `patch --dry-run` both pass clean, no
fuzz.

The struct fields the fix wires up -- `amd_queue_t::
max_legacy_doorbell_dispatch_id_plus_1`, `amd_queue_t::
legacy_doorbell_lock`, `AMD_SIGNAL_KIND_LEGACY_DOORBELL` -- already exist,
unused, in the pinned 7.14 source (confirmed via grep against
`runtime/hsa-runtime/inc/amd_hsa_queue.h` and `amd_hsa_signal.h`). AMD's
own dormant ABI scaffolding for legacy doorbells; the fix reactivates
disabled logic rather than retrofitting an incompatible layout.

Files: `patches/rocm-systems/hsa-agent-rejects-legacy-doorbell.{patch,sh}`,
`patches/rocm-systems/opencl-gfx8-hardcoded-rejection.{patch,sh}`.

The Dockerfile builds HIP only (`CLR_BUILD_OCL=OFF`); the OpenCL patch is
applied (harmless) but the OCL runtime itself isn't built, since this
stack doesn't use it.

## Layer 3: rocBLAS

Two patches carried over from the 6.4.4 line; one intentionally not
carried over.

### wgm-miscompute.sh -- unchanged, no port needed

Path/pattern-based (rewrites every `WorkGroupMapping:` entry in the
Tensile `Logic/` YAML tree via `sed`), self-verifying (fails the build if
its target pattern disappears). Tested against the pinned 7.14 rocBLAS
source: 103,741 non-1 `WorkGroupMapping` entries found, self-check passes
unchanged. Copied as-is.

### small-gemm-assembly-miscompute.patch -- re-diffed, two real drifts found

`git apply --check` reported "Skipped patch" with exit 0 and zero files
modified -- the `git apply` quirk above, not a real failure; `patch
--dry-run` correctly reported "Hunk #2 FAILED".

Two real drifts found by reading the current source:

1. **Reindentation** (cosmetic): `tensile_host.cpp`'s
   `runContractionProblem()` gained an `is_device_memory_size_query()`
   branch that pushed the patched block one level deeper (16-space vs.
   12-space body indent). Same control flow, just shifted.
2. **Real API change**: `runContractionProblem`'s template signature was
   `template <typename TiA, typename To, typename Tc, typename TiB,
   typename TcA, typename TcB>` in 6.4.4 -- separate input types for the A
   and B GEMM operands. Collapsed to `template <typename Ti, typename To,
   typename Tc>` in 7.14. The patch's `std::is_same<TiA, ...> &&
   std::is_same<TiB, ...>` gate wouldn't compile against 7.14 at all;
   rewritten to a single `std::is_same<Ti, float>::value` check.

Re-diffed patch applies clean (`patch -p1 --dry-run`, exact line match, no
offset).

### gsu-workspace-not-zeroed.patch -- not wired, matches 6.4.4

Re-diffed for completeness (kept in `patches/rocblas/` for reference), but
this patch was never wired into `rocm6.4.4/Dockerfile` either -- checked
directly, that Dockerfile's `rocblas-builder` stage only calls
`wgm-miscompute.sh` and `small-gemm-assembly-miscompute.sh`. Not wired
into this line's `Dockerfile` either, for the same reason.

## Layer 4: MIOpen

Three patches from the 6.4.4 line evaluated: two ported clean, one
blocked outright, two not attempted.

### winograd-fused-conv-miscompute.patch, reduce-prod-wrong-identity.patch -- ported unchanged

Both apply clean against the pinned 7.14 source with only small line
offsets (6 and 1 lines respectively), no content drift.
`ConvBinWinogradRxSFused::IsApplicable` and
`calculationfwdcontiguous`/`calculationparallelfwdcontiguous` (the exact
functions/classes these patches target) are present with matching
signatures. Copied as-is into `patches/miopen/`, `.sh` wrappers rewritten
to use `patch` instead of `git apply`.

### conv-direct-fwd-grouped-oob.patch -- blocked, target solver removed upstream

The whole `ConvOclDirectFwd*` solver family (OpenCL-source direct
convolution) was deleted from MIOpen between 6.4.4 and 7.14 --
`src/solver/conv/conv_ocl_dir2Dfwd.cpp` no longer exists; `src/solver.cpp`
explicitly comments each removed ID
(`++id; // removed ConvOclDirectFwd`, etc.). A replacement exists
(`ConvHipDirectFwd`, `src/solver/conv/conv_hip_dir2Dfwd.cpp`,
HIP-source, not OpenCL-source), but it is not the same code, and whether
it has the same grouped-convolution out-of-bounds weights-buffer read the
original patch fixed is unconfirmed. **Not applied, not ported** --
porting the old fix onto a different solver without re-running the
original repro would be inventing an unverified assumption.

### reduce-program-bound-eviction.patch, reduce-kernel-cache-eviction.patch -- not attempted

`reduce-program-bound-eviction` was opt-in/experimental even on the 6.4.4
line (gated behind `--build-arg ENABLE_REDUCE_BOUND=1`, off by default,
never validated against the real ORT test suite per its own header).
`patch --dry-run` against the pinned 7.14 `src/reducetensor.cpp` shows
real drift (hunk 1 failing, a large offset on the rest) -- not ported.
`reduce-kernel-cache-eviction` was already abandoned/not-shipped on the
6.4.4 line (its own `*** ABANDONED ***` header) -- not evaluated for
porting.

## MIGraphX

`parse-resize-fixes.patch` (two ONNX-parser backports from 6.4.4's
`release/rocm-rel-6.4` pin) is fully obsolete against `rocm-7.14`:

- Fix 1 (`keep_aspect_ratio_policy="stretch"` acceptance): already present
  verbatim in `rocm-7.14`'s `src/onnx/parse_resize.cpp`.
- Fix 2 (`.as_standard()` before `calc_neighbor_points`): the target
  function `calc_neighbor_points` no longer exists at all -- the
  linear-mode resize lowering it patched was replaced by a JIT-backed
  resize op upstream.

No port of the 6.4.4-era patch needed. Two new, 7.14-specific build
patches were needed instead (`patches/migraphx/`):

### gfx-default-rocblas-hipblaslt-off-build-failure.patch

`gpu::gfx_default_rocblas()` is declared behind `#if
MIGRAPHX_USE_HIPBLASLT` in `device_name.hpp` but called unconditionally
from `lowering.cpp` -- fails the build (undefined reference at link time)
whenever hipBLASLt is disabled, true for gfx803 since hipBLASLt has never
had gfx8 kernels. Fix moves the declaration outside the `#if` and adds a
stub `#else` returning `true` (dead at runtime: `hipblaslt_supported()`
always returns `false` when disabled, so `or gfx_default_rocblas()`
short-circuits away).

### mlir-stub-missing-symbols.patch

Same class of bug, different function: `is_module_fusible`,
`dump_mlir_to_file`, `dump_mlir_to_mxr`, `adjust_param_shapes` are
declared unconditionally in `mlir.hpp` and called unconditionally from
`jit/mlir.cpp` (always compiled, globbed via `file(GLOB JIT_GPU_SRCS
.../jit/*.cpp)`). Their only definitions live inside `#ifdef
MIGRAPHX_MLIR`; unlike sibling functions in the same file, these four
have no stub in the `#else` branch. Doesn't fail the build (nothing else
in `libmigraphx_gpu.so` references the missing symbols to trip the
linker) -- surfaces as `ImportError: ... undefined symbol` the first time
anything `dlopen`s it, caught by this Dockerfile's final-stage `import
migraphx` sanity check. Fix: added the missing stubs to `mlir.cpp`'s
`#else` branch, matching the existing stub style (`is_module_fusible`
returns `false`, the dump functions are no-ops).

Both are real upstream gaps, not gfx803-specific -- disabling a feature
via `-DMIGRAPHX_USE_X=Off` in this codebase doesn't reliably mean "no dead
code paths remain."

## ONNX Runtime -- switched to v1.28.0, MIGraphX-only (dropped ROCm EP)

Originally pinned to `v1.22.2` with `--use_rocm` kept alive, matching the
6.4.4 line. **Reverted after a real build failure**: v1.22.2's ROCm EP
source does not compile against ROCm 7.14's HIP headers --
`onnxruntime/core/providers/rocm/cu_inc/common.cuh`'s `constexpr int
GPU_WARP_SIZE = warpSize;` fails because `warpSize` changed from a
directly-constexpr-usable value to a non-constexpr accessor object
(`hip/amd_detail/amd_warp_functions.h`'s `operator int()`).

**Switched to `v1.28.0`, `--use_migraphx` only**, matching the main
(gfx900+) build's `docker/ort.Dockerfile` / `scripts/build/ort.sh`
exactly. `--use_rocm`/`--rocm_home` are gone from ORT's own build flags as
of 1.28 regardless (ROCm EP folded away upstream,
https://github.com/microsoft/onnxruntime/issues/26801).

Both 6.4.4-era ORT patches dropped, not ported: both patch
ROCm-EP-exclusive source
(`contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh`
and `core/providers/cuda/math/topk_impl.cuh`, the latter only compiled
when hipified for `--use_rocm`) -- neither file is reached at all once
`--use_rocm` isn't passed. `patches/onnxruntime/` is intentionally empty.

## PyTorch build: `TensorTopK.hip` compile-time fix

`clang-23 -cc1` compiling
`caffe2/CMakeFiles/torch_hip.dir/__/aten/src/ATen/native/hip/TensorTopK.hip.o`
peaks at ~19.6GB RSS alone -- `TensorTopK.hip`'s CUB radix-sort template
instantiations are expensive for LLVM's AMDGPU backend to
register-allocate at `-O3`. Fixed in-repo:
`patches/pytorch/tensortopk-hip-build-oom.patch` --
`set_source_files_properties()` on `TensorTopK.hip`, forcing `-O1` for
that one file only (trailing flag wins over the default `-O3`). Wired
into `pytorch-builder` the same way every other stage applies its patches.

**First attempt was silently ineffective**: patched
`aten/src/ATen/CMakeLists.txt` right after `ATen_HIP_SRCS` is assembled,
applied cleanly, but the live compile still showed `-O3`.
`set_source_files_properties()` is directory-scoped in CMake -- that file
only assembles the `ATen_HIP_SRCS`/`Caffe2_HIP_SRCS` variable list and
hands it up via `PARENT_SCOPE`; the actual `add_library(torch_hip
${Caffe2_HIP_SRCS})` call lives in `caffe2/CMakeLists.txt` (a sibling
directory, not a child of `aten/src/ATen`), so the property never reached
the target that compiles the file. Fixed by moving the override there
instead, right next to upstream's own `set_source_files_properties(...
PROPERTIES LANGUAGE HIP)` call, referenced via `${TORCH_ROOT}` (this
file's own established convention) rather than
`${CMAKE_CURRENT_SOURCE_DIR}`. Lesson for this codebase: when overriding a
per-source compile property, confirm the property is set in the same
directory scope as the `add_library()`/`add_executable()` call that
actually consumes the source list.

## va-reuse-defer

Carried over from 6.4.4 and wired into the `Dockerfile`
(`patches/rocr/va-reuse-defer.patch`) -- confirmed still needed on stock
ROCm 7.14 gfx803 via `rocm6.4.4/tools/reduce-harness/`'s 52-shape sweep,
at the same failure rate as 6.4.4 (~6-10/52) unpatched, 0/52 patched.
