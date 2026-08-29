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

### AQL ring queue-full workaround: `hsa_queue_create` stuck at a 64-packet floor

`hsa_queue_create` failed with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` for
every requested queue size except a hard floor of 64 AQL packets, while
`HSA_AGENT_INFO_QUEUE_MAX_SIZE` reported a normal 131072 -- the agent
wasn't the thing capping it. Root-caused via `kfd_queue_acquire_buffers()`
(`drivers/gpu/drm/amd/amdkfd/kfd_queue.c`): for GFX7/8 AQL compute queues
it validates the ring against `expected_queue_size = PAGE_ALIGN(queue_size
/ 2)` -- the kernel's own comment: "AQL queues on GFX7 and GFX8 appear
twice their actual size". Upstream deleted the userspace half of that
contract (`HsaMemFlags.ui32.AQLQueueMemory` -> `MemoryRegion::
AllocateDoubleMap`, which actually double-maps the ring's upper VA half
onto the same physical pages) along with the rest of gfx7/8 support, while
KFD kept expecting it. Every queue size above 1 page silently failed the
kernel's exact-match check; only 64 packets happened to round-trip through
`PAGE_ALIGN(size/2)` correctly.

Fix: restore `queue_full_workaround_` -- request `AllocateDoubleMap` from
the driver, and report the doubled span back to `CreateQueue` so KFD's own
halving lands on the real allocation. Verified on hardware: max queue size
64 -> 131072 packets, rocclr's live compute queue 64 -> 16384, a real
2048x increase. This also removes the constraint
`graph-replay-batch-chunk-deadlock.patch` exists to work around.

This patch was originally written believing it also explained the
long-standing gfx803 silent dispatch hang (a GFXIP 7/8 CP-can't-tell-a-
full-ring-from-an-empty-one theory). That was tested on hardware and is
false -- the hang reproduced unchanged with this patch applied and a
16384-packet ring, where the mechanism it addresses is unreachable. The
hang's actual cause was VRAM-clock marginality in the test hardware's
mining-tuned VBIOS, unrelated to this fix -- see README.md's "Host VBIOS
setting" section. This patch is still correct and still shipped, on its
own merits, for the queue-size fix alone.

Files: `patches/rocm-systems/aql-ring-queue-full-workaround.{patch,sh}`.
Requires a kernel WITHOUT the reference
`REFERENCE-amdkfd-gfx7-8-queue-size-writeback` patch (which forces the HQD
back to the single, un-doubled ring size and cancels this fix out), and
must never be combined with the superseded
`graph-replay-queue-size-cap.patch` (double-doubles the reported size and
faults the GPU under load).

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

### sgemm-shim strided-batched interceptor -- new for this line, attention GQA fix

The blanket `libgfx803_sgemm_shim.so` (routes f32 rocBLAS sgemm/gemm_ex to
the verified `gfx803_sgemm` kernel) left one broken path uncovered:
`rocblas_gemm_strided_batched_ex`, which is what MIGraphX's batched
`gpu::gemm` lowering calls for attention QK/QV dots (batch dims collapsed
into the gemm's M/N). On 7.14 that path still hit the GSU workspace-reuse
miscompute class (see `gfx803_sgemm.h` for the root cause), failing the
two `attention_*_gqa_with_past_and_present_expanded_cpu` ORT tests with
maxdiff up to ~4.5x.

The fix is a third interceptor in `sgemm_shim.cpp` for
`rocblas_gemm_strided_batched_ex`, looping `gfx803_sgemm` over the batch,
scoped to max(m,n,k) <= 32 (large batched GEMMs stay on real rocBLAS).

**The one thing that matters and is easy to get wrong**: the take-over gate
must NOT require `algo == rocblas_gemm_algo_standard`. `rocblas_gemm_algo_standard`
is 0, and MIGraphX passes `algo=1`, so that gate silently rejects every
take-over and falls through to the broken real rocBLAS kernel -- exactly
what happened during the investigation (the interceptor existed and looked
engaged, but the gate never matched, so nothing was actually fixed). The
shipped gate is `all_f32 && solution_index == 0 && c == d &&
small_problem && stride_c == stride_d && ldc == ldd`, no algo check. A
`sb-takeover-no-algo-gate` marker is embedded in the debug fprintf and the
Dockerfile build guard greps the built `.so` for it, so a regressed build
fails loudly instead of silently shipping a dead interceptor.

Verified on real gfx803: both attention GQA tests pass, full ORT suite
failures 13 -> 11 with exactly those two tests fixed and zero new
failures. This line's shim now differs from `rocm6.4.4/`'s copy (which
has no strided-batched interceptor) -- deliberate per the two-independent-
copies rule; the 6.4.4 line should get the same fix when converged or
independently re-verified.

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

## gfx7/8 EOP-completion-notification-loss, vLLM hang investigation, and
## the decision to deprioritize vLLM on gfx803

Long investigation, starting from an intermittent Qwen3.5-2B hang under
vLLM. Full chain:

1. **BAR stuck at 256MB.** A `nocrs` kernel boot flag was holding the
   card's PCIe BAR down to 256MB instead of a real 8GB Resizable BAR.
   Removing it enabled the full BAR -- but this *exposed* two hang sites
   that the small-BAR code path had apparently been routing around,
   rather than fixing anything on its own.
2. **`BlitKernel::SubmitLinearCopyCommand` and `AqlQueue::ExecutePM4`
   hangs.** Root cause: an undocumented gfx7/8 firmware erratum -- the
   GPU genuinely finishes the dispatched work, but the completion
   interrupt for that specific dispatch never arrives. Confirmed via live
   kernel-fence tracing showing real completion with no corresponding
   notification. Not reachable/fixable from software (no register or
   driver-level workaround found), so the fix is a bounded-retry
   mitigation, not a real fix: `patches/rocm-systems/
   blit-kernel-eop-interrupt-retry.patch`, gated behind
   `ROCR_GFX8_EOP_MITIGATION=1` (off/pristine by default -- on a system
   with a real Resizable BAR, code-object loads mostly route around the
   erratum-prone path anyway). Settled defaults after extensive sweep
   testing (with real output-correctness validation via greedy-decode
   text diffing against a known-good baseline, not just "didn't hang"):
   `ROCR_GFX8_EOP_MITIGATION_TIMEOUT_US=250`,
   `ROCR_GFX8_EOP_MITIGATION_MAX_ATTEMPTS=1`. Applied to both
   `BlitKernel::SubmitLinearCopyCommand` (size-scaled timeout, since a
   copy size is known there) and `AqlQueue::ExecutePM4`'s gfx8 branch
   (flat timeout -- no size context available at that call site).
3. **`sdma-doorbell-missing-sfence.patch` wired into the main
   Dockerfile.** A real, separate bug found and fixed independently:
   `SdmaQueue::RingDoorbell()` was missing an `_mm_sfence()` before its
   doorbell BAR write -- a release fence alone doesn't drain
   write-combined stores, only SFENCE orders them against the doorbell
   write. Same *bug class* as an existing fix in
   `hsa-agent-rejects-legacy-doorbell.patch`'s legacy-doorbell branch
   (missing SFENCE around a doorbell write), but a genuinely different
   bug: that one is CPU-write-ordering-before-the-doorbell-is-rung; the
   EOP mitigation above is GPU-finishes-but-the-completion-notification-
   never-arrives. Confirmed via code-path tracing that they don't share a
   root cause before shipping this as a separate patch rather than folding
   it into the EOP mitigation.
4. **Still-open, deliberately unmitigated: `hipMemcpyWithStream` /
   ROCclr's `WaitForSignal`.** This call path hangs too, but bypasses
   both of the mitigations above entirely -- it goes through the raw
   6-argument `SubmitLinearCopyCommand` overload with no built-in wait, so
   there's no size context to thread a safe timeout through without a
   real refactor of ROCclr's `Barriers`/signal-tracking architecture. A
   naive flat-timeout attempt was tried and caused a genuine GPU page
   fault on a large tensor copy (reverted immediately, `libamdhip64.so`
   restored to the pristine backup). Left unmitigated -- fixing this
   properly is a bigger task than one session, and shipping an unsafe
   mitigation here would trade a hang for a worse failure mode (silent
   corruption / page fault) on exactly the path that most needs to be
   trustworthy.
5. **`HSA_ENABLE_INTERRUPT=0` tested and ruled out.** Some profiling
   tools default to this (software/busy-poll signal waiting instead of
   hardware-interrupt-based) to sidestep unrelated deadlock classes.
   Tested via a 6-run batch with the EOP mitigation active: 0/6 clean,
   6/6 wedged -- no better than (and possibly worse than) the known
   ~30-35% clean-run baseline with the mitigation alone. Not shipped.

6. **Follow-up session: `ROCR_GFX8_EOP_MITIGATION=1` caught causing a
   full GPU bus death, worse than the hang it mitigates.** Testing with
   the mitigation *on* went worse than off: its bounded give-up-and-proceed
   logic let a `hipMemcpy` return early while its SDMA job was still
   genuinely in flight; the calling process then exited and closed its
   device fd, and the *kernel's own* `drm_sched_entity_flush` wait
   (unrelated to and not bounded by this userspace mitigation) blocked on
   that same still-outstanding job. Enough real time passed for the
   kernel's own ring-timeout watchdog to fire a GPU reset, which failed
   repeatedly (`device lost from bus`, `GPU Recovery Failed: -19`, 405+
   times) until the GPU was permanently wedged off the PCIe bus,
   unrecoverable short of a full reboot. This finding stands regardless of
   anything below: do not enable `ROCR_GFX8_EOP_MITIGATION=1` or
   `ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US` outside a disposable test
   environment.

**Superseded: the "deprioritize vLLM" decision above no longer holds.**
Items 1-6 correctly investigated and documented real software-level
behavior (the EOP mitigation's own failure mode, the missing SFENCE, the
unmitigated ROCclr call path) -- that work is accurate and the mitigations
above are still shipped. But none of it was the actual cause of the
underlying hang. A later investigation (2026-08-29) found the real cause:
VRAM-clock marginality in the test hardware's mining-tuned VBIOS, the same
condition also responsible for the `pool_sweep` GPU VM fault documented
below. Confirmed on hardware: with a VBIOS/clock combination that respects
the installed VRAM's real rated speed, 30/30 fresh-process vLLM launches
completed clean, versus the ~30-35% per-launch wedge rate this
investigation measured. vLLM is viable on gfx803 once the card's VRAM
clock is confirmed within spec -- see README.md's "Host VBIOS setting"
section for the general warning and fix, and the AQL ring queue-full
workaround above for the one real software fix that came out of this
investigation (does not fix this hang, but is a correct fix in its own
right).

### Qwen3.5-2B "hang": LLVM AMDGPU cold-compile is slow on gfx803, not stuck

Looked identical to a real GPU wedge at first glance: vLLM's engine-init
memory-profiling pass for this model would sit for minutes with the GPU at
0% busy. Every confirmed real gfx803 hang in this repo's history sits at
100% GPU busy (wave parked, needs a reboot) -- 0% busy for minutes pointed
somewhere else, so `py-spy dump --native --pid <pid>` was used to read the
live process's actual native stack rather than guessing.

That showed the process genuinely busy, not blocked: two threads pegged at
~200% combined CPU, native stack rooted in
`llvm::SLPVectorizerPass`/`SIFixSGPRCopies` inside Triton's bundled
`libtriton.so`, called from `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
(vLLM's GDN chunked-attention kernel, `vllm/model_executor/layers/fla/ops/
chunk_delta_h.py`) via its `@triton.autotune` config sweep.

Isolated per autotune config (`num_warps` in `[2, 4]`, `num_stages` in `[2,
3, 4]`, `BV` in `[32, 64]`, 12 configs total), compiled standalone via the
kernel's own `.warmup()` (`triton.runtime.autotuner.Autotuner.warmup`,
reached through `kernel.fn` -- calling `.warmup()` on `kernel` itself hits
the outer `Heuristics` wrapper, whose `.warmup()` forwards into
`Autotuner.run(warmup=True)`, which still calls `do_bench()` and dispatches
real launches; going through `.fn` avoids that, needed here since a
target-patched compile must never be dispatched onto real gfx803 silicon):

| num_warps | compile time (this box) |
|---|---|
| 2 | 0.00 - 1.54s per config |
| 4 | 19.18 - 66.51s per config |

Confirmed gfx803-specific, not just this kernel being generically expensive
to compile, by monkeypatching `triton.runtime.driver.active.get_current_target`
to report `gfx942` for a compile-only run (same box, same Triton/LLVM
build, same CPU, `.fn.warmup()` throughout so nothing reaches real
hardware): all 12 configs together compiled in 14.86s, against 203.4s for
the same 12 on `gfx803`. GCN's narrower SGPR file relative to gfx9+ is the
likely reason `num_warps=4` (256 threads/workgroup, more simultaneous live
ranges) blows up register-copy legalization and vectorization search cost
in LLVM's AMDGPU backend -- this is upstream LLVM/Triton compile-time
scaling, not a bug in anything this repo ships or patches.

Triton caches compiled kernels to disk (`~/.triton/cache`) keyed by
kernel/config/shape, so the cost is paid once per unique combination, not
every run: a cold `qwen35_2b_bench.py` run measured `init engine ... took
389.23 s`; an immediately following warm run (same box, same cache)
measured `13.45 s` and produced clean `prefill_tok_s=77.8` /
`decode_tok_s=56.1`. See README.md's Qwen3.5-2B status entry for the
user-facing summary.

## ldconfig silently reverting the patched libamdhip64/libhiprtc/
## libhiprtc-builtins back to stock

Found during this session's final regression pass, in the main
`Dockerfile`'s final stage. The stock ROCm dev base image ships its own
`-0000000`-suffixed build of `libamdhip64.so`/`libhiprtc.so`/
`libhiprtc-builtins.so` (three real regular files, not symlinks) sitting
in the same directory as the `-<commit>`-suffixed ones CLR's `make
install` adds -- both share the same SONAME (`libamdhip64.so.7`, etc.),
so which one the SONAME symlink actually resolves to is `ldconfig`'s call,
not a fixed fact about the image.

`ldconfig`'s version-comparison algorithm prefers the stock `-0000000`
suffix over a git-commit-hash suffix like `-ca887ee80a` -- reproduced
directly and minimally: run *only* `ldconfig` (no COPY, no cache, no
Docker layering involved) against a directory containing both files, and
`libamdhip64.so.7`'s symlink flips from the patched target to the stock
one. The final stage's own pre-existing verification check (grep for the
gfx8 opencl patch marker string in the *resolved* library, not the
symlink name) caught this for `libamdhip64` -- but `libhiprtc.so` and
`libhiprtc-builtins.so` have the identical stock/patched duplicate-file
problem with **no verification at all**, so a silent revert of those two
would have shipped undetected.

Fixed at the source: delete the stock `-0000000` duplicates immediately
before `ldconfig` runs in the final stage (`Dockerfile`, ~line 1041),
removing the ambiguity instead of hoping `ldconfig`'s version comparison
goes the right way. Not a new problem introduced this session -- this bug
would have hit any from-scratch build of this Dockerfile; it was
undetected purely because no prior build had reached this exact
final-stage check with the corrected patch chain from a fresh cache
state until this session's regression pass forced a real rebuild.

## `tools/correctness-suite/pool_sweep`: pre-existing GPU VM fault, not a
## regression -- RESOLVED, root cause was hardware (mining VBIOS VRAM clock)

Found during this session's regression pass: `pool_sweep` (MIOpen
pooling-forward correctness sweep) crashes with a real GPU VM fault --
not a soft miscompute -- on its very first, simplest test case (Max
pooling, C=8 H=16 W=16, kernel 2x2 stride 2 pad 0). `dmesg` on the box:

```
amdgpu 0000:02:00.0: GPU fault detected: 146 0x0ab8040c
amdgpu 0000:02:00.0:  Process pool_sweep pid 34658 thread pool_sweep pid 34658
amdgpu 0000:02:00.0:   VM_CONTEXT1_PROTECTION_FAULT_ADDR   0x009347FF
amdgpu 0000:02:00.0:   VM_CONTEXT1_PROTECTION_FAULT_STATUS 0x100C400C
amdgpu 0000:02:00.0: VM fault (0x0c, vmid 8, pasid 1116) at page 9652223, read from 'TC3' (0x54433300) (196)
```

Same fault address every single run (`0x9347ff000`/`0x934800000`),
deterministic, not flaky. Reproduces identically with
`ROCR_GFX8_EOP_MITIGATION=1` set, so it is **not** the EOP-completion-
notification-loss erratum this session's mitigations target -- it's a
different, unrelated bug.

**Confirmed pre-existing, not a regression from this session's changes**,
via two independent checks: (1) the identical fault, at the identical
address, reproduces against the currently-published
`ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803` image --
i.e. the image this repo has been shipping, untouched by anything from
this session; (2) `dmesg` shows the exact same fault class (`VM fault ...
read from 'TC3'`, `IH ring buffer overflow`) already hit by a real vLLM
workload (`Process VLLM::EngineCor`) earlier in this session, at a nearby
but different address (`0x00A751AB`) -- establishing this as a recurring,
pre-existing fault class on this hardware, not something newly
introduced. `README.md`'s Status section has been corrected to note this
exception to the previously-claimed clean 23/23 pass.

**Root cause identified in a follow-up session: hardware, not software.**
Extensive tracing (ioctl-level, PM4-dispatch-level, kernel TLB-flush
review, delay-based mitigation) ruled out every software explanation --
see `RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md` problem 2 for the full
investigation.
The actual cause: this card's mining-tuned VBIOS ran VRAM (MCLK) at
2100MHz, above its real correctness margin -- a clock mining workloads
tolerate occasional bit errors at, but this correctness-checked workload
does not. Reflashing to a non-mining VBIOS running VRAM at 1750MHz fixed
it completely: 20/20 clean runs afterward, vs ~50% crashing on the mining
VBIOS. No software fix exists in this repo for this issue because there
was never a software bug -- see `README.md`'s "Host VBIOS setting"
section.

## Final regression pass (this session)

Run against `rocm-gfx803:rocm7-regression`, a rebuild of the main
Dockerfile (with the ldconfig fix above, `sdma-doorbell-missing-sfence`,
and `blit-kernel-eop-interrupt-retry` all wired in) built against the
*live* `release/therock-7.14` branch tip, not a frozen commit -- per this
repo's own branch-pin convention. PyTorch/torchvision/torchaudio stages
were deliberately excluded from this particular build (via a build-only,
not-committed Dockerfile variant) since nothing in this regression pass
tests against them and skipping them saved substantial build time; the
tracked `Dockerfile` itself is unchanged in that respect.

- `tools/correctness-suite/`: 22/23 sweeps clean. `pool_sweep` exception
  documented above (pre-existing, unrelated).
- `verify.py`'s non-torch-dependent checks (ONNX Runtime MIGraphX EP
  provider check + Relu inference, and the static WGM8 kernel-name symbol
  scan for the historically dangerous silent-rocBLAS-miscompute class):
  all clean. The torch-based GEMM/MIOpen-conv numeric checks in
  `verify.py` were not run this pass (no PyTorch in this particular
  build) -- a known, deliberate gap in this one regression pass, not a
  claim that those checks now pass or fail.
- ORT's `onnx_backend_test_series.py`: attempted but not completed --
  blocked on a chain of missing pip dependencies for `onnx`'s own pytest
  report plugin (`pytest`, then `tabulate`), and even once collection
  succeeded, the script's test-case generation (`globals().update(...)`
  at module scope) doesn't populate any test items when invoked via a
  bare `pytest <file>` -- it expects to be driven through its own CLI
  entrypoint, not pytest's collector. Not investigated further given time
  budget; deferred. The historical 3-way diff against the 6.4.4 and
  gfx1201 lines documented earlier in `README.md`'s Status section is
  unaffected by this -- that was prior work, not redone or invalidated
  this session.
- `llama-cpp-gfx803` rebuilt (base config, no `gfx803-packed-dp4a.patch`
  performance patch) against `rocm-gfx803:rocm7-regression` as its base
  image. 3/3 `llama-bench` runs against
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`: clean, no hangs, no crashes,
  consistent numbers each run (~506 t/s pp128, ~141 t/s tg64). Confirms
  llama.cpp/HIP remains a working path on this hardware post-patch-chain.
