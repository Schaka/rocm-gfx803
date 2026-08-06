# gfx803 kernel bugs: what's broken, how we found it, how to fix more of it

gfx803 (Polaris) has been unsupported since ROCm 6.0 and untouched by AMD's
own testing for years before that. Its Tensile/rocBLAS kernel library --
generated code, some hand-written assembly -- has never had a reliable
correctness bar applied to it on this hardware. We are not going to get
upstream fixes for any of this. Nobody is going to report these bugs to AMD:
the platform is abandoned, and there is no realistic path to a fix landing.
The only way gfx803 stays usable is patching this ourselves, kernel by
kernel, the same way [`patches/rocblas/wgm-miscompute.sh`](patches/rocblas/wgm-miscompute.sh)
already did for one bug and [`patches/rocblas/sgemm-shim/`](patches/rocblas/sgemm-shim/)
does for a second, larger one. This document is the playbook for finding the
next one.

## Known-broken so far

| Bug | Scope | Symptom | Status |
|---|---|---|---|
| WGM8 (`WorkGroupMapping != 1`) | rocBLAS assembly GEMM kernels, all dtypes | Silent wrong results, `rocblas_status_success` | Fixed: [`patches/rocblas/wgm-miscompute.sh`](patches/rocblas/wgm-miscompute.sh) |
| Pervasive Tensile fp32 GEMM miscompute | Every rocBLAS/Tensile fp32 GEMM solution (assembly and source-kernel), any shape, any GSU, any tile config | First call in a process correct, every call after intermittently wrong (1%-100% of output elements, error magnitude up to 1e5) | Fixed for the standard-algo fp32 path: [`patches/rocblas/sgemm-shim/`](patches/rocblas/sgemm-shim/) |
| MIOpen `ConvOclDirectFwd` grouped-conv OOB | `MIOpenConvUniBatchNormActiv` kernel (`ConvOclDirectFwd`/`ConvOclDirectFwdFused` solver), grouped/depthwise convolutions specifically | GPU memory access fault, page not present -- an actual out-of-bounds read past the weights buffer, not stale/uninitialized data | Fixed: [`patches/miopen/conv-direct-fwd-grouped-oob.sh`](patches/miopen/conv-direct-fwd-grouped-oob.sh), MIOpen rebuilt from source (`miopen-builder` stage in the Dockerfile). Confirmed NOT the same bug as the GEMM one above -- the sgemm-shim fix only measurably delayed this crash (1-4 sessions -> 11 sessions) because it never touched MIOpen at all. See "MIOpen ConvActivationFusion investigation" below for the full root-cause chase. |
| int8_float32 GEMM (CT2) | rocBLAS int8_r/int32_r GEMM path | Garbled/repeating output, not a crash | **Confirmed fixed as of the MIOpen grouped-conv fix**, though not independently re-isolated. Retested end-to-end via the real audiomuse-rocm-plugin image (JFK sample, `faster_whisper`, `int8_float32`) after rebuilding the base image with the MIOpen fix: correct transcription, whereas this was previously garbled. Whisper's encoder has real Conv1D layers, so this was plausibly the same `ConvOclDirectFwd` grouped-conv defect corrupting the int8 path's output, not a separate rocBLAS int8 bug -- but that's an inference from the retest, not a confirmed root cause the way the fp32 GEMM and MIOpen bugs above were. Worth a real isolation pass if it regresses. |

## How we found the pervasive GEMM bug

Short version of a long investigation: MIOpen's fused Conv+Bias kept crashing
or silently corrupting on gfx803 even after WGM8 was fixed. Chasing it all
the way down (ruling out cache/invoker reuse, kernel-arg packing, hardware
memory/LDS/atomic reliability, alignment, split-K/GSU, compiler version, and
source-vs-assembly kernels one at a time) eventually landed on: **every one
of the 55 rocBLAS/Tensile solutions returned for a real problem shape is
broken**, not just an unlucky subset. Below is the reusable version of that
process.

### 1. Write a minimal, MIOpen/ORT/CT2-independent repro

Don't debug through four layers of framework. Call rocBLAS directly:

```c
rocblas_status st = rocblas_gemm_ex(handle,
    rocblas_operation_none, rocblas_operation_none,
    M, N, K, &alpha,
    d_a, rocblas_datatype_f32_r, M,
    d_b, rocblas_datatype_f32_r, K,
    &beta,
    d_c, rocblas_datatype_f32_r, M,
    d_c, rocblas_datatype_f32_r, M,
    rocblas_datatype_f32_r,
    rocblas_gemm_algo_standard, 0, 0);
```

Check every output element against a CPU double-precision reference computed
from the exact same random inputs. Loop many iterations with fresh random
data each time -- **the defining symptom of these bugs is that the first call
in a process is correct and later calls are not**, so a single-shot test will
miss it entirely.

**Tolerance matters.** Use a combined absolute+relative threshold
(`err_abs > 0.05 && err_rel > 1e-2`), not relative-only. Float32 GPU
accumulation order legitimately differs from a double-precision CPU
reference; relative-only checks flag that noise as corruption when the true
reference value is near zero. Real corruption from these bugs is unmistakably
larger (error magnitudes in the hundreds to tens of thousands), so this isn't
a fine needle to thread.

### 2. Enumerate every candidate solution, not just the default pick

`rocblas_gemm_ex_get_solutions` (a beta API -- `#define
ROCBLAS_BETA_FEATURES_API` before including the header) returns every
solution index that can solve a given problem. Run each one individually via
`rocblas_gemm_algo_solution_index`, many reps each:

```c
rocblas_int list_size = 0;
rocblas_gemm_ex_get_solutions(handle, ..., nullptr, &list_size);
std::vector<rocblas_int> solutions(list_size);
rocblas_gemm_ex_get_solutions(handle, ..., solutions.data(), &list_size);
// then call rocblas_gemm_ex with algo=rocblas_gemm_algo_solution_index,
// solution_index=solutions[i], for each i, REPEATS times each.
```

**Check `rocblas_status` on every call and report it explicitly.** Silently
`continue`-ing past a failed status (meaning that solution doesn't apply to
this problem) and only reporting "clean" for solutions with zero *observed*
mismatches will make an inapplicable solution look identical to a genuinely
correct one -- we hit this exact bug in our own harness and it produced a
false "two identical-looking solutions, one clean one broken" result that
took an extra hour to debug before realizing the "clean" one had never
actually run a single real computation.

**Use enough reps.** 20 reps is not enough -- a genuinely ~15%-broken
solution has a real chance of showing 0/20 mismatches by pure luck. We saw
this happen. Use 100+ before trusting a "clean" result.

### 3. Correlate solution index to kernel identity

The installed library's `.dat` files (msgpack format) map solution index to
kernel name and full parameter set (`sizeMapping`: `globalSplitU`,
`macroTile`, `depthU`, etc.):

```python
import msgpack
with open("/opt/rocm-6.4.4/lib/rocblas/library/TensileLibrary_Type_SS_..._gfx803.dat", "rb") as f:
    data = msgpack.unpack(f, raw=False, strict_map_key=False)
for s in data["solutions"]:
    print(s["index"], s["name"], s["sizeMapping"])
```

Solution index ranges are scattered across multiple `.dat` files (one per
transpose combination); check `min(index)`/`max(index)` per file to find
which one covers your solution IDs. Group broken-vs-clean solutions by
`GlobalSplitU`, tile size, prefetch flags, etc. to look for a pattern --
though be aware the pattern may not exist (ours didn't cleanly correlate with
any single parameter; see below).

### 4. Rule out the obvious explanations empirically, not by assumption

Each of these is a real, cheap, decisive test -- don't skip straight to
reading Tensile's C++ source before trying them:

- **Alignment/tail-loop bugs**: test a shape that's an exact multiple of
  every plausible tile/depth size. If it *also* fails, it's not a
  remainder-handling bug.
- **Split-K (GlobalSplitU)**: rebuild the Tensile logic tree with
  `GlobalSplitU` forced to 1 (`sed` the yaml, matching
  `wgm-miscompute.sh`'s approach), rebuild rocBLAS, and re-test. **Scope the
  edit to the arch-specific logic directory** (`Logic/asm_full/r9nano` for
  gfx803), not the whole tree -- GlobalSplitU is more deeply coupled to other
  solution fields than WorkGroupMapping was, and a whole-tree rewrite broke
  Tensile's own fallback source-kernel codegen for us
  (`use of undeclared identifier 'gsuSumIdx'`).
- **Hardware/thermal instability**: write a hand-rolled HIP kernel that
  stresses the same primitives GEMM kernels use (LDS tile staging + sustained
  FMA accumulation; separately, global atomics + cross-workgroup flag
  coordination) with NO Tensile/rocBLAS involvement. If these run clean for
  hundreds/thousands of iterations while every real GEMM kernel doesn't,
  that rules out generic hardware unreliability under load -- the defect is
  specific to what Tensile generates, not the hardware's ability to compute
  under stress.
- **Compiler regression**: rebuild rocBLAS from the exact `rocm-X.Y.Z` tag
  that was the *last officially gfx803-supported release* (find it via
  `DEFAULT_AMDGPU_TARGETS`/`TARGET_LIST` changes in rocBLAS's CMake across
  tags), using that release's own base image (`rocm/dev-ubuntu-*:X.Y-complete`)
  so its own bundled compiler builds it. If the bug reproduces there too
  (ours did, worse even), it's not a newer-LLVM regression -- it predates
  official support entirely and was never caught.
- **Source vs. assembly kernels**: Tensile ships both hand-tuned assembly
  (`ISA803`) and portable-HIP-source (`ISA000`/`KLS`) kernels as fallback for
  every solvable problem. Force-select fallback solutions specifically (low
  solution-index numbers in the "fallback" `.dat` file, distinct from the
  large per-arch assembly index ranges) and test them the same way. If both
  kernel classes fail, the bug isn't in hand-written assembly specifically.

If all of the above come back "still broken," the defect is in Tensile's own
generated logic/dispatch, not any single tunable parameter -- which is
exactly the situation this GEMM bug turned out to be.

### 5. When you can't isolate one broken parameter: write a replacement kernel

If literally every candidate solution is unreliable (as was the case here --
55/55, both officially and unofficially supported compiler eras, both kernel
classes), there is no "good" kernel to select and no single yaml field to
patch. The fix is to stop asking Tensile for a kernel at all, for the
affected op:

1. Write a plain, unoptimized, correctness-first replacement using the SAME
   primitives already proven reliable in step 4's hardware tests (LDS tiling,
   FMA accumulation, no cleverness). See
   [`patches/rocblas/sgemm-shim/gfx803_sgemm.h`](patches/rocblas/sgemm-shim/gfx803_sgemm.h)
   for the SGEMM version -- 16x16 tiles, no prefetch, no vectorized loads,
   nothing for a miscompile to hide in.
2. Verify it exhaustively: aligned shapes, misaligned shapes, tiny shapes,
   large shapes, hundreds of reps each, against the same double-precision
   reference and combined absolute+relative tolerance from step 1.
3. Wire it in via `LD_PRELOAD` symbol interposition rather than patching
   rocBLAS's source -- intercept the specific entry point(s)
   (`rocblas_sgemm`, `rocblas_gemm_ex`, etc.), route the verified-safe subset
   of calls to the replacement, and `dlsym(RTLD_NEXT, ...)` everything else
   through to the real symbol unchanged. See
   [`patches/rocblas/sgemm-shim/sgemm_shim.cpp`](patches/rocblas/sgemm-shim/sgemm_shim.cpp).
   This keeps the blast radius explicit and auditable, and makes it trivial
   to A/B against stock rocBLAS (`GFX803_SGEMM_SHIM_DISABLE=1`) without
   rebuilding anything.
4. **Build with an explicit `--offload-arch=gfx803`.** hipcc's default target
   auto-detection depends on a GPU being visible on the machine doing the
   compile; Docker builds run on a CPU-only host, so without this flag the
   shim silently compiles for the wrong (or no) architecture and fails with
   `invalid device function` at runtime, not at build time.
5. **Link the shim against the real library** (`-lrocblas` here). Symbols the
   shim calls directly (not through `dlsym`), like `rocblas_get_stream`, need
   to resolve at `LD_PRELOAD` load time -- which happens before the
   application has loaded anything else -- so an implicit "it'll resolve
   against whatever the process loads later" doesn't work.

### The shape-sweep tool, and why the shim is still blanket

[`tools/rocblas_sweep.cpp`](tools/rocblas_sweep.cpp) generalizes step 2 above
into a matrix sweep: 21 shapes (tiny/aligned/prime/skinny/large/degenerate)
x every candidate solution x 30 reps, run overnight. Result: every one of the
55 candidate solutions was broken for every tested shape with
`min(M,K,N) < 256`, and every one was **100% clean** for every tested shape
with `min(M,K,N) >= 256` (256^3, 512^3, 513^3, 768x768x3072, 1024^3 all
0/55 broken).

That looked like grounds to narrow the shim's scope -- pass large shapes
through to real rocBLAS, only intercept small ones. **Direct testing
disproved it.** Re-running `rocblas_repro` at 768x768x3072 through the
*default* dispatch path (`algo=standard`, `solution_index=0` -- what every
real caller actually uses, not the explicit `rocblas_gemm_algo_solution_index`
override the sweep used) still corrupted 5/500 iterations, error magnitude up
to 1.4. **rocBLAS's automatic solution-selection heuristic for the default
path does not necessarily choose one of the solutions the sweep enumerated
and tested as clean.** The two dispatch paths (explicit solution index vs.
automatic default) can behave differently on identical hardware and shape --
a sweep over one doesn't characterize the other.

Practical upshot: the shim stays blanket for now. **Do not narrow it based on
explicit-solution-index sweep data** -- it doesn't predict the default path's
behavior. Narrowing it correctly would require sweeping the *default* path
itself (varying shape only, always `solution_index=0`), which is a different,
not-yet-built tool. This is a real performance cost for fp32 GEMM
specifically (our replacement is unoptimized), traded for correctness.

### The actual root cause, and a real (failed) attempt to fix it at the source

Chasing the "first call correct, every call after wrong" signature further
(rather than accepting the shim as final) found the real mechanism, not just
another correlation:

**gfx803 (Polaris/GCN3) has no native float atomic-add instruction.**
Tensile emulates GlobalSplitU (split-K reduction across multiple workgroups)
on this architecture via a software compare-and-swap retry loop --
confirmed by disassembling an actual installed kernel binary (solution
537657391, `..._GSU32_...`): the accumulation phase is `buffer_atomic_cmpswap`
in a load/add/CAS/retry-on-contention loop, not a hardware atomic. Newer
architectures (gfx9+) have real hardware float atomics and never generate
this code path at all -- a clean explanation for why AMD's own testing would
never have caught this.

This CAS loop has **no zero-init logic of its own** (confirmed by reading
the disassembly directly, not inferred) -- it goes straight from computing
its partial sum to reading whatever is currently at the target address and
accumulating on top via CAS. With N independent, concurrently-running GSU
groups all doing this, there is no "first" group that could self-initialize
even if the codegen wanted one to; zero-init is necessarily the caller's
responsibility, done once, before the kernel launches.

rocBLAS's own device memory pool (`library/src/include/handle.hpp`,
`gsu_malloc_by_size()`) hands out this workspace as reused memory across
successive GEMM calls **without re-zeroing it**. Confirmed via direct
`hipMalloc`/`hipFree` tracing (an `LD_PRELOAD` interposer logging every real
allocation) on real hardware: rocBLAS allocates its internal pool **once**,
not per call -- there is no repeated alloc/free between GEMM calls, the same
physical device memory backs every GSU-using call in a process. A controlled
dose-response test nailed the mechanism precisely: giving a broken solution
its own externally-managed workspace via `rocblas_set_workspace`, zeroed
before every call, eliminated corruption completely (0/50); the same buffer
set once and never re-zeroed only partially helped (15/50, down from
baseline's 20/50) -- proving it's specifically the missing re-zero, not some
other side effect of user-managed memory mode.

The workspace-management fix works when applied at the *shim* layer too, not
just source-patched into rocBLAS: see
[`patches/rocblas/sgemm-shim/sgemm_shim_privatehandle_reference.cpp`](patches/rocblas/sgemm-shim/sgemm_shim_privatehandle_reference.cpp)
for a compiling, correctness-verified (0/N mismatches on every previously-broken
shape) reference implementation -- kept as documentation of "freeing/zeroing
workspace memory correctly fixes this," not shipped, because it benchmarked
slower than the plain hand-written kernel in every case tested (see the
file's own header for the numbers).

**Patch attempt** (`patches/rocblas/gsu-workspace-not-zeroed.sh`): insert one
`hipMemsetAsync` of the GSU workspace, sized to Tensile's own
`solution->requiredWorkspaceSize()`, in `tensile_host.cpp` right after
`gsu_malloc_by_size()` hands back the pointer and before the kernel launches
that follows -- on the same stream, so no extra synchronization needed.
Rebuilt rocBLAS from source with this patch (full Tensile kernel generation
+ build, ~20-30 min on the test box) and tested against real hardware:

| Shape | Zero-init patch alone |
|---|---|
| 40,72,800 (small) | **worse**: 499/500 broken, max error magnitude ~2.2 (vs ~1.4-2.2 unpatched) |
| 768,768,3072 (large) | **fixed**: 0/50 clean |
| 129,129,129 | **worse**: 99/100 broken, max error magnitude ~4.5 |

Large shape genuinely fixed; small shapes got *worse*, with *larger* error
magnitudes than the original bug ever produced -- a real regression, not
noise. Reading `gsu_malloc_by_size()` again turned up a second, independent,
pre-existing rocBLAS bug that would explain it:

```cpp
auto gsu_malloc_by_size(size_t requested_Workspace_Size)
{
    if(this->gsu_workspace) // Added to accomodate quant, remove comment after testing
        return _gsu_malloc_by_size(this);
    return _gsu_malloc_by_size(this, requested_Workspace_Size);
};
```

The fast/reuse path hands back whatever workspace the handle already has
**without checking it's actually big enough** for the current request. If an
earlier call on the same handle needed less GSU workspace than the current
one does, this returns an undersized buffer -- and our zero-init memset,
sized to the *current* call's requirement, would then write past the end of
it. That's consistent with the evidence: bigger error magnitudes than
before, looks like real memory corruption rather than stale data.

**Fixed this too** (also in `gsu-workspace-not-zeroed.sh`): guard the reuse
path with `this->gsu_workspace_size >= requested_Workspace_Size`, falling
through to a fresh correctly-sized allocation otherwise. Rebuilt and
retested:

| Shape | Zero-init + size-guard patch |
|---|---|
| 40,72,800 | still broken: 499/500, max error ~2.4 -- no real change |
| 768,768,3072 | still fixed: 0/50 clean |
| 129,129,129 | still broken: 99/100, max error ~3.9 -- no real change |

**The size-guard fix made essentially no difference.** That disproves the
theory of what was causing the small-shape regression -- and since neither
patch touches anything for shapes that don't dispatch through a GSU-based
solution in the first place, the most likely explanation is that the
*default* solution-selection heuristic (`algo=standard, solution_index=0`,
what every real caller uses) picks a **different kernel/dispatch path
entirely** for small shapes than the GSU-workspace mechanism this patch
fixes -- and that different path has its own, still-unidentified defect,
never touched by either patch. The large shape happening to only ever
select GSU-based solutions is presumably why it improved while small shapes
didn't. This is the same "default dispatch doesn't match explicit solution
enumeration" trap documented above for the sweep tool, encountered again
here from a different angle.

**Conclusion: reverted.** The source patch is real, mechanistically
justified, and demonstrably fixes what it targets (confirmed clean on the
one shape that exercises only that mechanism) -- but does not make rocBLAS
correct overall, and applying it changes rather than removes the shim's
necessity. Given no regression suite for this architecture and a live
demonstration that a "small, well-understood" patch attempt still produced
a *worse* failure mode on other shapes before the second bug was found (and
no real improvement after), further blind iteration here is not a good use
of time versus the already-verified, structurally-immune (no GSU/atomic
reduction at all) hand-written kernel in
[`patches/rocblas/sgemm-shim/`](patches/rocblas/sgemm-shim/). The patch
script and this writeup are kept for future reference -- the zero-init fix
in particular may still be a genuinely correct, worthwhile fix for AMD to
apply upstream even though it isn't sufficient on its own for gfx803's
default dispatch path.

Diagnostic tools built during this investigation, kept in
[`tools/`](tools/) for reference: `malloc_tracer.cpp` (LD_PRELOAD
hipMalloc/hipFree logger), `rocblas_workspace_test.cpp` and
`rocblas_privatehandle_test.cpp` (workspace-management isolation tests),
`rocblas_repro_abscheck.cpp` (combined abs+rel tolerance checker --
supersedes the older `rocblas_repro.cpp`, which uses relative-only tolerance
and produces false positives), `rocblas_timing.cpp` (pure GPU-side timing,
no host round-trips).

This same caution applies to any future MIOpen conv-solver investigation:
MIOpen's own solver auto-selection may not match whatever gets tested by
enumerating individual solvers explicitly, for the same reason.

Not covered by this fix at all:

- **int8/int32 GEMM** (used by CT2's `int8_float32` compute type): different
  rocBLAS dtype path, confirmed still broken (garbled repeating output, not a
  crash), not yet investigated with this methodology.

MIOpen's own conv solvers (separate from rocBLAS/Tensile's GEMM library
entirely, not reached by this shim) were also confirmed to have their own,
distinct bug -- see the next section. That one is now fixed too.

## MIOpen ConvActivationFusion investigation

The fp32 GEMM fix above measurably delayed the ConvActivationFusion crash
(single-digit sessions to survive -> 11 sessions before a GPU memory access
fault) without eliminating it, confirming it as a genuinely separate defect.
Chasing it down required extending the methodology above in one significant
way: **the fault this bug produces is an actual out-of-bounds memory
access** (a hard page fault), not silent wrong-but-readable data like the
GEMM bug -- and HIP's async kernel dispatch means the *reported* fault can
lag well behind the kernel that actually caused it, so naive symptom-first
debugging reliably misattributes the crash to whatever kernel happened to
be in flight when the fault surfaced, not the real culprit.

### Isolating a synthetic MIOpen-only repro: four clean negative results

Following the same instinct as the GEMM investigation (write a
framework-independent minimal repro), four different isolation attempts
using bare MIOpen (no ORT, no real model) all came back completely clean,
each ruling out one plausible hypothesis:

1. **Plain MIOpen handle create/destroy churn** (100 iterations, a synthetic
   8-channel conv, sequential `miopenConvolutionForward` +
   `miopenActivationForward`): 100/100 clean. Rules out generic MIOpen
   handle lifecycle.
2. **Fusion-plan churn** (same synthetic shape, but through the actual
   `miopenCreateFusionPlan`/`miopenCompileFusionPlan`/`miopenExecuteFusionPlan`
   API `ConvActivationFusion` really uses): 50/50 clean. Rules out generic
   fusion-plan lifecycle.
3. **The real crashing shape, single-shot** (`hi=16 wi=50 n=1
   k_per_group=240 c_per_group=40` -- decoded from the crashing kernel's raw
   arg dump via `AMD_LOG_LEVEL=3` tracing, a 1x1 pointwise 40->240-channel
   conv from CLAP's audio backbone), non-fused, forced through
   `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1`: 5/5 clean, confirmed via
   shader-name trace that the `naive_conv_ab_nonpacked_fwd_nchw` kernel
   really did launch (multiple times -- `miopenFindConvolutionForwardAlgorithm`
   benchmarks every applicable solver).
4. **Same real shape, 200x churn**: 200/200 clean.

None of the "obvious" variables (handle lifecycle, fusion-plan lifecycle,
exact real shape alone, exact real shape + churn) reproduced it standalone.
The defect needed something about the *actual* ORT/model integration.

### Confirming genuine memory corruption, not stale data

Before going further, `tools/malloc_tracer.cpp` (the same `LD_PRELOAD`
`hipMalloc`/`hipFree` interposer that found the rocBLAS bug) was run against
the real crashing `clap_churn_repro.py` session. The fault address
(`0x911c12000`) landed **73728 bytes past the end of a legitimately-sized
4MB allocation** visible in the trace -- a real out-of-bounds access, not a
reused-but-unzeroed buffer like the GEMM bug. Different bug class entirely;
the GEMM investigation's workspace-zeroing lessons didn't transfer directly.

### `AMD_SERIALIZE_KERNEL=3`: precise fault attribution, and a correction

The raw crash trace (no serialization) pointed at
`naive_conv_ab_nonpacked_fwd_nchw_float_double_float` -- MIOpen's generic
reference/fallback kernel, used when no tuned solver covers a shape (gfx803
has no tuning-database entries at all, so this triggers constantly). This
looked like the culprit and led to some real but ultimately misleading
analysis time.

Setting `AMD_SERIALIZE_KERNEL=3` (forces every kernel to complete
synchronously before the next launches, so a fault attributes to the exact
kernel causing it instead of whichever one happened to be mid-flight)
reran the same crash and told a different story: the actual kernel at fault
was **`MIOpenConvUniBatchNormActiv`** -- a fused conv+batchnorm+activation
kernel, not `naive_conv` at all. Re-running the *original* (unfixed) crash
a second time under serialization confirmed the same kernel again, with
weight pointer `0x92c9c5900` inside a 4MB `obj:[0x92c600000-0x92ca00000]`
buffer, faulting at `0x92ca11000` -- again just past the buffer's end.

**The earlier `naive_conv` attribution was a misattribution**, an artifact
of async fault-reporting lag, not a second independent bug. Both serialized
captures point at the same single root cause. Worth internalizing for any
future crash-not-miscompute investigation: don't trust which kernel a raw
(non-serialized) trace blames for a page fault -- confirm with
`AMD_SERIALIZE_KERNEL=3` before spending time on it.

### Finding the real shape, and the applicability bug

`tools/miopen_stride_tracer.cpp` (an `LD_PRELOAD` interceptor for
`miopenSetTensorDescriptor`/`miopenConvolutionForward`/`miopenExecuteFusionPlan`,
reading real dims/strides at the API level since the raw kernel-arg trace
couldn't render struct-typed arguments) traced the actual CLAP model's
layer sequence and found the crash landing right after setting up
descriptors for a classic MobileNet-style inverted-residual block --
expand 1x1, **depthwise 3x3**, project 1x1 -- with no
`miopenConvolutionForward` call logged before the fault, meaning it was
going through the fused path (`miopenExecuteFusionPlan`) the plain-conv
interceptor never saw.

Reading `ConvOclDirectFwd::IsApplicable`
(`src/solver/conv/conv_ocl_dir2Dfwd.cpp`) found the actual gap: a
filter-size restriction gate that only applies `if
(problem.GetGroupCount() == 1)`. Grouped/depthwise convolutions skip that
gate entirely and fall through with no group-specific size or shape
restriction at all, reaching `MIOpenConvUniBatchNormActiv` (via
`ConvOclDirectFwdFused`) without the kernel's own group-aware correctness
ever having been established for gfx803.

### Minimal, deterministic, standalone reproduction

`tools/miopen_depthwise_fusion_repro.cpp`: the exact depthwise shape
(`K=240, C=1 per group, groups=240, R=S=3`) through the fusion-plan API,
nothing else involved. **Reliably faults on the very first execute** --
the first fully isolated, deterministic repro of this bug, no ORT, no
churn, no real model required.

### Two candidate fixes, and why the scoped one shipped

**Considered first: `MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD=0`** (an existing
MIOpen debug env var matching the solver's name). Confirmed effective:
40/40 session churns clean (vs. crashing by session ~4 without it) --
disabling the solver makes its fusion-plan `compile` step fail instead of
crashing, and ONNX Runtime's `ROCMExecutionProvider` handles that
gracefully by falling back to unfused conv+bias+activation. But this
disables the solver for **every** shape, not just grouped ones -- a real,
concrete risk: some other non-grouped shape (in this model or another)
that currently relies on `ConvOclDirectFwd` because no faster solver
applies could get pushed onto a *different* MIOpen fallback with its own
unverified correctness (`naive_conv` was briefly, incorrectly suspected of
exactly this during the investigation above -- turned out not to be
independently broken, but the *category* of risk is real and would apply
to whatever the next-best solver is for some untested shape).

**Shipped: `patches/miopen/conv-direct-fwd-grouped-oob.sh`.** Rejects
`GroupCount() != 1` directly in `IsApplicable`, so only the genuinely broken
case is excluded; non-grouped shapes that legitimately want this solver are
unaffected. Required standing up a from-source MIOpen build (`miopen-builder`
stage in the Dockerfile) where none existed before -- MIOpen's own
`install_deps.cmake` unconditionally pulls in `composable_kernel` and
`rocMLIR`, neither of which has ever supported gfx8 (same as this repo's
main Dockerfile already excludes both); both were filtered out of
`requirements.txt` before running it, which turned a build that was
heading toward compiling tens of thousands of irrelevant CK instantiation
files into one that completed normally.

Verified on real hardware after the source rebuild:
- The standalone depthwise repro: no crash, `compile` fails gracefully
  (status 8) as expected -- same safe outcome as the env var, but reached
  through the applicability check instead of a blanket solver disable.
- Non-grouped fusion churn (the earlier synthetic 8-channel repro): still
  20/20 clean -- the patch doesn't affect legitimate non-grouped dispatch.
- The real production workload (`clap_churn_repro.py`, 40 session churns,
  `ConvActivationFusion` enabled, no other workarounds): **40/40 clean**.

Diagnostic tools from this investigation, kept in [`tools/`](tools/):
`miopen_handle_churn.cpp`, `miopen_fusion_churn.cpp`,
`miopen_naive_shape_repro.cpp` (isolation attempts 1-4 above),
`miopen_stride_tracer.cpp` (API-level dims/strides interceptor),
`miopen_depthwise_fusion_repro.cpp` (the minimal deterministic repro),
`clap_churn_repro.py` (the real-workload regression test these were all
validated against).
