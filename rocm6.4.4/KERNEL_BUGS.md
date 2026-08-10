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
| Small-shape Tensile assembly GEMM miscompute | rocBLAS/Tensile fp32 GEMM, gfx803 assembly (`ISA803`/`KLA`) kernels, whenever *every* dimension is small (correct from ~48x48x48 up; thin shapes fine if one dimension is large) | Silent wrong results, `rocblas_status_success`. Reproduced down to a **1x1x1** problem: a cross-workgroup race in the GSU CAS-accumulate path, winner decided by GPU queue occupancy, so it only manifests under back-to-back dispatch. Surfaced as 84 of `onnxruntime_test_all`'s 96 failures (Einsum, Attention, DecoderAttention, MultiHeadAttention, FusedMatMul) | Worked around, not root-caused: [`patches/rocblas/small-gemm-assembly-miscompute.sh`](patches/rocblas/small-gemm-assembly-miscompute.sh) makes Tensile dispatch prefer a HIP source kernel for small fp32 problems (guarded to `prob.strided_batch` -- the array-of-pointers batched GEMM API isn't supported by that source solution and must fall through to the tuned assembly). 96 -> 12 failures. The race *inside* the assembly kernels is understood but not fixable via patch -- see "The small-shape assembly GEMM miscompute" below |
| int8_float32 GEMM (CT2) | rocBLAS int8_r/int32_r GEMM path | Garbled/repeating output, not a crash | **Confirmed fixed as of the MIOpen grouped-conv fix**, though not independently re-isolated. Retested end-to-end via the real audiomuse-rocm-plugin image (JFK sample, `faster_whisper`, `int8_float32`) after rebuilding the base image with the MIOpen fix: correct transcription, whereas this was previously garbled. Whisper's encoder has real Conv1D layers, so this was plausibly the same `ConvOclDirectFwd` grouped-conv defect corrupting the int8 path's output, not a separate rocBLAS int8 bug -- but that's an inference from the retest, not a confirmed root cause the way the fp32 GEMM and MIOpen bugs above were. Worth a real isolation pass if it regresses. |
| MultiHeadAttention "MHA basic" mode rejected on ROCm | ORT's own `GemmSoftmaxGemmPermuteGenericPipeline` (contrib op C++, not rocBLAS/MIOpen) -- the only path available once Composable Kernel is compiled out | `Could not find viable op` for any plain 3-tensor MultiHeadAttention call with query sequence_length > 1 (an ordinary shape, not an edge case); rocBLAS is never even reached | Fixed in ORT source: [`patches/onnxruntime/mha-basic-mode-no-viable-op.sh`](patches/onnxruntime/mha-basic-mode-no-viable-op.sh) adds the missing Q/K/V transpose (via ORT's own existing `LaunchTransQkv` utility) before the GEMMs that need it. 12 -> 3 failures, no regressions. |
| Generic TopK RadixTopK tie-break nondeterminism | ORT's own generic TopK kernel (`topk_impl.cuh`, contrib op / core op C++, not rocBLAS/MIOpen) -- hipified verbatim from CUDA, no ROCm-specific source | Picks a different winner among exactly-tied candidates from run to run on bit-identical input (hipCUB `BlockScan`/`BlockReduce`-based tie allocation). Surfaced as `BeamSearchTest.GptBeamSearchFp32_DisableFastTopK` decoding a different token sequence each run, ~50% of isolated runs | Fixed in ORT source: [`patches/onnxruntime/topk-radix-tiebreak-nondeterministic.sh`](patches/onnxruntime/topk-radix-tiebreak-nondeterministic.sh) replaces the affected dispatch path with a self-contained, hipCUB-free kernel on ROCm only. 18 isolated runs: 17 passed (was ~50%). Not gfx803-specific in principle. |
| MIOpen `miopenReduceTensor` dynamic-reduction intermittent wrong sums | Root cause: below MIOpen. The HIP allocator's **device-address recycling** (hipFree->hipMalloc returning the same input VA) interacts with gfx803's GPU memory-system state (stale cache/TLB coherence) so the **specific CK reduction kernels mis-read freshly-written input** at the reused address. Proven via the committed `gfx803/tools/reduce-harness/` (real CK kernels, plain hipModule + hipMalloc/hipFree reuse, no MIOpen host code: reuse=10/52, fresh-distinct-input=0/52, dominated by the INPUT buffer). Rock-solid: seq5-via-API reuse=7/52 vs fresh=0/52; observer effect ruled out (copy to a *recycled* internal buffer = 21/52). | **Not fixed -- open, below MIOpen.** Fix belongs in the HIP runtime stack (ROCR-Runtime / libhsakmt re-map + invalidation on VA reuse), not MIOpen (confirmed unable: caller owns the recycled d_x). Full root-cause record + ROCm7 repro guidance: see "The ReduceSum kernel-cache mystery" below AND `docs/AGENT_HANDOFF.md` §18. Prior eviction workarounds (always-fresh ~70x slow; LRU-bound 91% not 100%) are NOT shippable. MIOpen-fallback (reduce from a fresh >=16-buffer ring-address internal copy) is proven correct (0/52) at +9-13% perf but is only a fallback. |

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

## The small-shape assembly GEMM miscompute

gfx803's tuned Tensile *assembly* GEMM kernels return wrong results, silently
and with `rocblas_status_success`, whenever every dimension of the problem is
small. Fixing this took `onnxruntime_test_all` from 96 failures to 12.

The shipped fix ([`patches/rocblas/small-gemm-assembly-miscompute.sh`](patches/rocblas/small-gemm-assembly-miscompute.sh))
routes small fp32 problems to a HIP source kernel inside rocBLAS's Tensile
dispatch. It is a workaround at the right layer, **not** a root-cause fix: what
is actually wrong inside those assembly kernels is still unknown.

### The boundary

| problem | assembly kernels |
|---|---|
| all of m, n, k <= ~32 | **wrong** |
| 48x48x48 and larger | correct |
| thin shapes (m, n or k = 1/4/16/32 with another dimension at 256) | correct |

The corruption needs *all three* dimensions to be small. A GEMV-shaped problem
with one large dimension is fine. Miscomputing problems captured from ORT were
as small as `1x1x1`, `2x2x2`, `3x3x2` and `2x4x2`.

### What it is not

Four structural hypotheses were tested and each was refuted with direct
evidence, which is worth recording so nobody re-runs them:

* **GlobalSplitU / CAS accumulation.** GCN3 has no float atomic-add, so GSU>1
  solutions CAS-accumulate into C. Rewriting `GlobalSplitU` to 1 across the
  gfx803 logic fixed only *half* the Einsum failures (96 -> 72) and none of the
  Attention ones. Not the cause, and redundant once the dispatch fix is in.
* **LocalSplitU LDS reduction / missing barrier.** The failing calls dispatch
  `..._MT16x16x16_..._WG8_8_4`, i.e. LocalSplitU=4, which reduces through LDS.
  Disassembling the built kernel shows the correct
  `ds_write / s_waitcnt lgkmcnt(0) / s_barrier / ds_read` sequence, and the
  reduction fits inside the `m0` LDS bound (4096 B used, `m0 = 0x1880`).
* **Edge handling.** `EdgeType: ShiftPtr` -> `Branch` across the gfx803 logic
  builds clean, deploys, and changes nothing: same 16 mismatching GEMMs, one
  bad-element count moving 16 -> 15.
* **Any of the above, at all.** A `1x1x1` GEMM miscomputes. One output element,
  one multiply -- no split-K, no tiling, no reduction, no edge shift.

### How to reproduce it

The only trustworthy oracle is an **in-process** verifier inside ORT. Standalone
rocBLAS harnesses do not reproduce this reliably; see the traps below.

Apply the verifier to
`onnxruntime/core/providers/rocm/tunable/gemm_rocblas.h` -- snapshot A, B and C
with device-to-device copies **on the GEMM's own stream** (a host `hipMemcpy`
capture lies, see the note in the WGM section), recompute on the host in double,
and print any mismatch. Then:

```
GEMM_VERIFY=1 ./onnxruntime_test_all \
    --gtest_filter='Einsum.*:EinsumTransposeMatMulThreeInputsTests/*'
```

That reproduces the whole failure set in about 10 seconds instead of the ~4
minutes a full suite run takes, and prints the exact shapes.

### Traps that produce false results

Every one of these cost real time here:

* **Never force a solution index globally.** Assembly solutions are
  shape-restricted; forcing one for every GEMM makes rocBLAS reject most calls,
  `onnxruntime_test_all` aborts, and each abort dumps core. A 68-solution sweep
  produced 4416 core files / 4.1 GB and took the machine down hard. Enumerate
  applicable solutions per shape with
  `rocblas_gemm_strided_batched_ex_get_solutions` first, and always
  `ulimit -c 0` before running the test binary.
* **A fixed solution index is transpose-specific.** Solution 369 is
  `Cijk_Ailk_Bljk` -- NN only. Forcing it on Attention's transposed GEMMs
  returns `rocblas_status_invalid_value` and fails the node, which looks exactly
  like ~14 "Attention regressions" caused by source kernels. They were not
  regressions; they were rejected calls.
* **`librocblas.so.4` does not point at `librocblas.so.4.4`.** It resolves to
  `librocblas.so.4.4.60404`. Copying a freshly built library over
  `librocblas.so.4.4` changes nothing at all and every measurement silently
  describes the stock library. Check the symlink target, and replace the file
  with `mv` (atomic rename) rather than `cp` -- overwriting the mapped, in-use
  `.so` in place segfaults `cp` and can leave a truncated library.
* **Tensile logic (YAML) changes and C++ changes deploy differently.** The
  generated kernels and selection logic live in
  `/opt/rocm/lib/rocblas/library/*.dat`/`.co`; dispatch code lives in the `.so`.
  Replacing only one of the two tests only half of what you think it does.
* **Observing the bug hides it.** `TENSILE_DB=0x8000`, and even an extra
  `fprintf` to stderr between launches, is enough to make the failures
  disappear. Kernel *selection* is deterministic and can safely be inspected
  that way; kernel *correctness* cannot.
* **Standalone per-solution scans are not conclusive here.** A padded,
  warmed-up standalone harness reports every solution clean at these shapes,
  and a harsher one condemns the source solutions that ORT proves correct.
  Snug allocations also let one solution corrupt the next. Trust the in-process
  verifier.

### The 1x1x1 case: root cause narrowed to a cross-workgroup race

Extended the in-process verifier to dump every operand and result for any
mismatch small enough to read by eye (`patch_dumpsmall.py`, applies on top of
`patch_snapshot.py`), restored the *stock* rocBLAS library so the bug would
reproduce, and re-ran the full `Einsum.*` suite. This produced two `1x1x1`
mismatches -- one multiply, one output element, `alpha=1 beta=0`, expected
`A[0]*B[0]` exactly:

```
call=31  A=2  B=10  Cpre=10  Cpost=10   expect=20   (Cpost == Cpre: no write at all)
call=36  A=1  B=10  Cpre=10  Cpost=20   expect=10   (Cpost == Cpre + expect: stale value added in, not zeroed)
```

Same output address both times (`ptrC=0x90a800200`), same shape, same
transpose, same `alpha`/`beta` -- and two structurally opposite failures. That
rules out a static per-shape logic defect (a fixed wrong edge-shift or wrong
operand index would fail the same way every time) and points at a **race
between two per-element roles in the GSU (GlobalSplitU) accumulate path**: one
workgroup is responsible for the initial `beta*C` store, another does the
atomic-CAS accumulate of its partial sum (GCN3 has no native float
atomic-add, so this is a compare-and-swap retry loop -- see
`gsu-atomic-accumulate-miscompute` history above). Depending on which one wins:
the accumulator can fire before the init-store lands (reads/CAS-adds onto
stale pool memory -- call=36), or the init-store can land *after* the
accumulate and clobber it outright (call=31 looks like nothing happened only
because the "clean" write of `beta*C = 0` never actually ran either -- both
paths were skipped or overwritten).

A third, structurally distinct-looking case confirmed this isn't limited to
`1x1x1`: a `3x3x2` GEMM (`Einsum.ExplicitEinsumAsMatmulNhcwTransposeB`) came
back with its entire last output column (the `n=2` tail, since 3 doesn't tile
evenly) hard **zero** rather than stale-plus-partial. Zero looked at first like
a separate edge/tail-handling defect (`EdgeType: ShiftPtr`), but see below --
it isn't.

**The decisive test: isolation kills the bug.** Both the `1x1x1` cases and the
`3x3x2` case were re-run *alone* (just that one test, nothing before it) three
times each. All six runs passed -- verifier reports `ok` every time. The
failures only appear inside the full `Einsum.*` run, always on the same tests,
always preceded by other small GEMMs on the same stream. A pure logic bug
(wrong shift, wrong index, wrong identity value) reproduces regardless of
context; this does not. That makes it a **race whose winner is decided by GPU
queue occupancy at launch time** -- back-to-back dispatch leaves the previous
kernel's tail wavefronts still retiring when the next kernel's workgroups
launch, which changes hardware scheduling order deterministically for a given
call sequence but never shows up when a kernel launches onto an otherwise-idle
GPU. This also explains two earlier observations that looked unrelated at the
time: `TENSILE_DB=0x8000` (or any added `fprintf` between launches) hides the
bug because it changes launch timing, and the "3x3x2 zero tail" and "1x1x1
stale-add/dropped-write" are the *same* mechanism, not two bugs -- which
workgroup loses the race just determines what garbage (zero, stale pool
memory, a clobbered partial) ends up in the output.

**This is not something a Docker-image patch can fix.** The missing piece is
an inter-workgroup dependency/fence between the beta-init store and the GSU
CAS-accumulate that Tensile's code generator does not emit for gfx803. Fixing
it means regenerating or hand-patching the assembly kernels themselves, which
was already ruled out as unreliable (see "Patching the assembly kernels
directly" discussion, above). The shipped dispatch fix
(`small-gemm-assembly-miscompute.patch`) remains correct and sufficient: it
avoids this whole class of kernel below the 8x8x8 threshold rather than
trying to fix the race inside it.

### Resolved: the four FusedMatMulOpTest failures were the dispatch fix rejecting a GEMM variant it doesn't support, not a miscompute

The four `FusedMatMulOpTest` cases (`FloatTypeScale`, `FloatTypeTransposeA`,
`FloatTypeTransposeAB`, `FloatTypeTransposeBatch`) failed under the rocBLAS
fix but passed under an earlier ORT-side version of the same routing, and the
in-process verifier showed no GEMM mismatch for them -- which was genuinely
confusing, because "the verifier saw nothing wrong" was being read as "the
math is fine, something else is going on." It doesn't mean that: the verifier
instruments the *result* of a rocBLAS call, and these calls never got that
far.

Running the four alone showed the actual error every time:

```
ROCBLAS failure 11: rocblas_status_invalid_value ... expr=rocblasGemmBatchedHelper(...)
```

`rocblasGemmBatchedHelper` is the **array-of-device-pointers** batched GEMM
(`rocblas_gemm_batched_ex`) -- a different call shape from
`rocblas_gemm_strided_batched_ex`, which passes one base pointer plus a fixed
stride. rocBLAS's `RocblasContractionProblem` (`library/src/include/tensile_host.hpp`)
carries an explicit `strided_batch` bool for exactly this distinction: `true`
for plain/strided GEMM, `false` when built from `batch_A`/`batch_B`/`batch_C`
pointer arrays (`library/src/blas3/Tensile/gemm_tensile.hpp`, the two
`rocblas_call_tensile` overloads).

The small-gemm dispatch patch's gate never checked this flag. It called
`candidate->canSolve(tensile_prob, *hardware)` on HIP source (`_KLS_`)
solutions and trusted a `true` to mean "this call will work" -- but
`canSolve()` validates problem *shape* (size bounds, divisibility), not
pointer-array compatibility. For an array-of-pointers batched call, it
returned `true` for a source solution that rocBLAS's actual dispatch path
then rejected outright. Confirmed by running each of the four alone (nothing
before them): they failed every time, ruling out cache pollution from an
earlier strided-batched call with the same shape signature -- the very first
call for that shape already picks the incompatible solution.

Fix: added `&& prob.strided_batch` to the gate in
`small-gemm-assembly-miscompute.patch`. Array-of-pointers batched GEMMs now
fall straight through to `findBestSolution` and the tuned assembly path,
exactly as they did before this patch existed -- unaffected by either the fix
or, for whatever it's worth, the underlying race, since none of the four
tests exercise the tiny-and-racy shape region in the first place.

`onnxruntime_test_all`: **16 -> 12 failures**, all four `FusedMatMulOpTest`
cases pass, no regressions elsewhere.

### Not a kernel bug: StringNormalizer x7 was a missing locale, fixed in the image

Seven of the twelve remaining failures were never GPU/ROCm-related. ORT's
`StringNormalizer` op constructs a `std::locale` by name at runtime
(`en_US.UTF-8` in these tests); Ubuntu's base images ship no locale data at
all, so the constructor throws for every caller, GPU or not:

```
Failed to construct locale with name:en_US.UTF-8:locale::facet::_S_create_c_locale
name not valid:Please, install necessary language-pack-XX and configure locales
```

Fixed in the image itself (final Dockerfile stage): install `locales`,
`locale-gen en_US.UTF-8`, `update-locale`, and set `LANG`/`LANGUAGE`/`LC_ALL`.
This matters beyond the test suite -- any real caller using `StringNormalizer`
against this image hit the same failure. Verified directly (not just inferred
from the count): all 8 `StringNormalizer` tests pass after generating the
locale, re-run in isolation. Removes 7 of the 12 `onnxruntime_test_all`
failures; see the combined full-suite number after the MultiHeadAttention fix
below, since both were verified together in the same run.

### MultiHeadAttention "MHA basic" mode: an ORT bug, not rocBLAS/MIOpen

The remaining `MultiHeadAttentionTest.CrossAttention_Batch1_HeadSize8` failure
looked, at first pass, like a gfx803 hardware gap: `TunableOp::FindFastestImpl`
reported `Could not find viable op` for an fp16 fused attention op, and GCN3
genuinely lacks the matrix-core hardware Composable Kernel (CK) needs -- CK is
compiled out of this image entirely (`onnxruntime_USE_COMPOSABLE_KERNEL=OFF`,
see the ORT build stage). That looked like the end of the story.

It wasn't. `batched_gemm_softmax_gemm_permute_pipelines.cuh` implements the
attention computation two ways: CK, and a "Generic" fallback built from plain
rocBLAS GEMMs. With CK compiled out, Generic is the *only* path -- but
Generic's own `GetSupportedStatus()` rejects the failing test's exact mode,
`BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE`, labeled in this same file's own mode
table as **"MHA basic"**: the single most ordinary MultiHeadAttention call,
three separate un-preprocessed Q/K/V tensors, no fused QKV projection. It's
rejected unconditionally except when query `sequence_length == 1` (where a
`[1,H]` slice per head happens to be byte-identical whether read as BSNH or
BNSH). Any real multi-token attention call hits the rejection outright.

Confirmed by isolating the exact failing test and instrumenting
`GetSupportedStatus` to print `attn->mode`/`qkv_format` rather than guessing:
`mode=5` (`BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE`), `sequence_length=2`,
`kv_sequence_length=3` -- an entirely ordinary cross-attention shape. Tracing
`params.q_buffer`/`k_buffer`/`v_buffer` back through `multihead_attention.cu`
confirmed they are direct offset views into the raw input tensors -- nothing
upstream permutes them for this mode, unlike the fused QKV-projection path the
file's own top-of-file doc comment describes ("In QKV projection (prior to
this pipeline): Q [B,S,N*H] ->Reshape-> [B,S,N,H] ->Permute0213-> [B,N,S,H]").
`Gemm1`/`Gemm2` assume BNSH-strided input; for "MHA basic" that permute simply
never happens. **rocBLAS is never called for this case at all** -- the
rejection happens in ORT's own C++ dispatch, one layer above any GPU kernel
launch, which is why the fix belongs in ORT and not in a rocBLAS/MIOpen patch.

The fix ([`patches/onnxruntime/mha-basic-mode-no-viable-op.sh`](patches/onnxruntime/mha-basic-mode-no-viable-op.sh))
adds the missing permute using `LaunchTransQkv` -- a utility ORT already ships
and already uses for the equivalent step in the plain fused `Attention` op,
just never wired up for this MultiHeadAttention mode. Extra workspace is
reserved (three transposed BNSH copies of Q/K/V, appended after the existing
gemm1_out/softmax_out/gemm2_out region so it can't alias it), the transpose
runs once per call via a shallow copy of the params struct (safe: the struct
owns no resources), and `GetSupportedStatus` now accepts the mode instead of
rejecting it.

Verified: the target test passes with correct fp16 output. Full
`MultiHeadAttentionTest`/`AttentionTest`/`DecoderAttentionTest`: 53 passed, 0
failed, 13 pre-existing explicit skips, no regressions. Full
`onnxruntime_test_all` with both this fix and the locale fix above active:
**12 -> 3 failures** (`SamplingTest.Gpt2Sampling_GPU`,
`ReductionOpTest.ReduceSum_apex_matrix_large`,
`ResizeOpTest.NoAntialias_AlignCorners_Cubic_Floor_NHWC`).
`BeamSearchTest.GptBeamSearchFp32_DisableFastTopK` also didn't fail in this
run, but neither fix here addresses it -- follow-up work (below) found it's
genuinely nondeterministic even completely alone (not the same
context-dependent pattern as `ReductionOpTest.ReduceSum_apex_matrix_large`,
which is deterministic given the same preceding calls), so its absence here
was luck, not something to credit to either fix.

Since CK is what upstream/CUDA builds actually run, this gap is specific to
ROCm builds without CK -- not gfx803-specific in principle, though only
verified here.

### Generic TopK: RadixTopK's tie-breaking is genuinely nondeterministic on ROCm

`BeamSearchTest.GptBeamSearchFp32_DisableFastTopK` looked at first like a
gfx803 hardware gap too: it forces `ORT_BEAM_SEARCH_USE_FAST_TOPK=0`, which
routes through ORT's *generic* TopK op instead of the specialized beam-search
topk kernel. GCN3 has no matrix cores, and CK is compiled out of this image
-- easy to assume the same story as MultiHeadAttention above. It wasn't.

Isolating the test alone (nothing else running) showed it fails ~50% of the
time, in a **fresh process every run** -- not the "fails in the full suite,
passes alone" signature the GEMM race and `ReduceSum` show. This rules out
context/memory-pool state as the cause; whatever's wrong is inside the
kernel itself, on identical input, every time it's invoked.

`onnxruntime/core/providers/cuda/math/topk_impl.cuh` (hipified verbatim from
CUDA, no ROCm-specific source, like MHA's pipeline file) dispatches TopK
three ways depending on shape: `BitonicTopK` (fits one block, no external
dependency), `RadixTopK` (moderate K, built on hipCUB's
`BlockScan`/`BlockReduce`/`BlockRadixSort` -- rocPRIM-backed on ROCm, a
different implementation from NVIDIA's own cub), and a
`cub::DeviceRadixSort`-based fallback for large K. Instrumenting the dispatch
confirmed `RadixTopK` fires every call for this test (`K=8`,
`dimension=4000`). Microsoft's own code already flags this exact kernel as
non-deterministic when `use_deterministic_compute` is requested -- a
warning, not a different code path.

Dumping the actual selected scores/tokens took real care: an initial dump
that added a `hipStreamSynchronize` + device-to-host copy made the test pass
4/4 -- the same observer effect as `TENSILE_DB=0x8000` masking the gfx803 GEMM
race, just rediscovered in a different subsystem. Switching to a
non-invasive dump (printing `next_scores`/`next_tokens`/`next_indices`,
already host-visible at that point in the existing code, no new
synchronization) reproduced the failure normally. At `%.9g` precision, the
differing values were literal `0`, not near-zero floating-point noise --
this test's toy GPT2 model outputs exact-zero logits at several decode
steps, so the top-K selection is choosing among many exactly-tied
candidates, and `RadixTopK`'s tie handling (an `equal_quota` scheme built
from `BlockScan`/`BlockReduce` exclusive prefix sums, meant to allocate tied
slots deterministically by thread index) picks a different winner across
runs on the same input.

Two fixes that looked obvious were tried and rejected:

- Routing to `cub::DeviceRadixSort` (a different hipCUB code path --
  device-wide, not block-level primitives, so plausibly immune to the same
  defect) segfaults. Backtrace pinned it to
  `RocmKernel::GetScratchBuffer`: the beam-search call site
  (`generation_device_helper.cc`) deliberately passes `kernel=nullptr` to
  `TopKImpl`, with an inline comment explaining why -- `"We limit number of
  beams in BeamSearchParameters, so K <= 256 and use NULL here"` -- banking
  on always landing in `RadixTopK`, never the kernel-dependent
  `DeviceRadixSort` path.
- Extending `BitonicTopK` (no hipCUB dependency at all) past its
  `aligned_dimension <= blockDim` gate isn't a safe drop-in: its sort/merge
  phases index via `tid`/`tid<<1` pairs that only cover positions reachable
  within one block's thread count in a single pass. Making it handle this
  test's dimension (4000) needs rewriting the compare-exchange network for
  multiple elements per thread throughout.

Fix: [`patches/onnxruntime/topk-radix-tiebreak-nondeterministic.sh`](patches/onnxruntime/topk-radix-tiebreak-nondeterministic.sh)
adds `SafeSmallKTopK` -- K sequential block-wide argmax passes, hand-written,
no hipCUB block primitives, no `CudaKernel` dependency (works with
`kernel=nullptr`), and no O(dimension) shared memory (tracks the K selected
indices found so far, not a dimension-sized flag array, so it stays cheap
for real vocab sizes). Ties broken by lowest index, matching this file's own
`IS_SMALLER`/`BIGGER` macro convention. Scoped to ROCm only via `#ifdef
USE_ROCM`; CUDA is untouched.

Verified: 18 isolated runs of the target test, 17 passed (was failing ~50%
of the time). Full `TopKOperator` suite (53 tests, including
`Top3AllSame` -- an explicit tie-handling test -- and large-array cases)
plus `BeamSearchTest`/`SamplingTest`: 62 passed, 0 regressions.

**Not fully fixed.** The one residual failure (of 18 isolated runs) showed
the identical symptom -- a different decoded token sequence, not a crash or
new error. Since `SafeSmallKTopK` has no atomics, no hipCUB, and no
data-dependent control flow beyond straightforward comparisons, a lingering
~6% failure rate after removing the confirmed defect points at a second,
separate source -- most plausibly numeric variance from ORT's `TunableOp`
runtime kernel auto-tuning (which op wins its own timing-based benchmark can
vary run to run) interacting with this test's near-degenerate, heavily-tied
scores. Not chased further here -- it's a much broader mechanism (affects
any op with multiple tunable candidate implementations, not specific to
TopK or beam search) and de-risking it would mean a different investigation
entirely.

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

## The ReduceSum kernel-cache mystery

**This one is not fixed at the root.** Everything else in this document
identifies a specific defective instruction sequence and corrects it. This
section is different: a real, reproducible bug was found and worked around,
but *why* the workaround works was never established. Read this before
touching [`patches/miopen/reduce-kernel-cache-eviction.sh`](patches/miopen/reduce-kernel-cache-eviction.sh)
or reusing its pattern anywhere else.

**The bug.** A standalone, ORT-independent MIOpen-only repro -- one
`miopenHandle_t`, a sweep of `miopenReduceTensor` ADD calls over many
`[m,n]` shapes reducing axis 0, matching `ReductionOpTest.ReduceSum_apex_matrix_large`'s
exact call pattern -- shows about 15% of shapes in the sweep return wrong
sums. Every one of those same shapes passes 100% of the time when run
alone on a fresh handle. Reversing the shape iteration order changes
*which* shapes fail and makes it worse (16/52 wrong instead of 8/52) --
proof this is about accumulated state on the handle, not a property of any
individual shape.

**What it is not**, each ruled out with a targeted, isolated experiment:

- *A GPU dispatch-order race between the two kernels one `miopenReduceTensor`
  call launches* (a "prepare" kernel that writes a tensor descriptor into a
  scratch workspace, then a "main" kernel that reads it back).
  `AMD_SERIALIZE_KERNEL=3` (forces fully synchronous kernel execution) has
  zero effect -- same shapes fail, same mismatch counts. A patch inserting
  `handle.Finish()` between the two dispatches (confirmed active via HIP
  API tracing: an explicit `hipEventSynchronize` genuinely occurs) also had
  zero effect, and was reverted.
- *MIOpen's on-disk JIT kernel-binary cache.* Identical failures whether
  that cache starts cold or warm, and identical with `MIOPEN_DISABLE_CACHE=1`.
- *GPU device-memory address reuse.* Identical failures even with
  `hipMalloc`/`hipFree` per shape replaced by one static, never-freed
  buffer pool for the entire sweep (rules out stale-cache-on-reused-address
  theories).
- *Kernel launch occupancy.* Adding `__launch_bounds__(BlockSize, 1)` to
  both kernels (forcing `minBlocksPerMultiprocessor=1`) has zero effect.
- *Host-side memory corruption in `reducetensor.cpp`'s own C++ logic.*
  Running the repro under `valgrind --track-origins=yes` reproduces the
  exact same failures with **zero** memcheck errors reported -- and
  valgrind's massive host-timing slowdown didn't change the outcome
  either, so it isn't host-timing-sensitive.
- *Insufficiently-specific compile-time cache keys* (two different
  tunables colliding on the same compiled binary). Ruled out by
  inspection: `get_definition_string_from_tunable`/
  `get_network_config_string_from_tunable` both fully encode
  `BlockSize`/`GredThreadBufferLength`/etc, so different tunables never
  share a cache key.

**What reliably eliminates it**, confirmed repeatedly: a genuinely fresh
`miopenHandle_t` per call (nothing has ever been compiled/cached on it),
or adding *any* extra per-thread device memory write inside the kernel
(tried purely as a debugging aid, to dump internal state -- and this
masked the bug even once the instrumentation was moved to the file that's
*actually* live for this repro's shapes; an earlier round of debugging had
wasted a full rebuild cycle instrumenting `..._all_dims.cpp` when the real
dispatch for this shape family goes through `..._partial_dims.cpp`,
confirmed via `AMD_LOG_LEVEL=4` HIP API tracing showing the true kernel
argument count).

**A parallel escape route was tried and abandoned:** forcing MIOpen's
older, non-CK "static" reduction implementation
(`MIOPEN_DEBUG_DYNAMIC_REDUCTION=0`) to sidestep the dynamic path
entirely. That path has its own independent, pre-existing compile-time bug
-- an `Array<int,N>` size mismatch inside `CalculateLowerIndexDiff`/`to_array`
in the old `composable_kernel` template library -- and does not currently
build at all for this shape family. Fixing that would be its own
open-ended investigation into unmaintained template metaprogramming, with
no guarantee the result is even correct once it compiles.

**A workaround was built, validated extensively, and then abandoned --
NOT shipped.** After every `miopenReduceTensor()` dynamic-CK dispatch, evict
that call's own kernel-cache and program-cache entries
(`handle.ClearKernels`/`handle.ClearProgram`, both existing public `Handle`
APIs) so the handle never gets to accumulate multiple different compiled
reduction kernels -- reproducing the "nothing else cached" state that makes
isolated/fresh-handle calls reliable. This passed every standalone repro
thrown at it, repeatedly and thoroughly: 5 consecutive clean runs of the
full 52-shape sweep, the worst-case reversed-order sweep (16/52 without the
fix), a 200-iteration repeat of the identical shape on one handle, and a
repro mixing `MIOPEN_REDUCE_TENSOR_ADD`/`MAX`/`MIN` across many shapes on
one handle to mimic ORT's real usage pattern -- 0 failures in every case,
across two versions of the patch (the second adding `handle.Finish()`
before every eviction, to close a suspected use-after-eviction race).

It nonetheless failed ORT's actual test suite
(`onnxruntime_test_all --gtest_filter=*ReductionOpTest*`)
non-deterministically -- not on every run, and not reproducible by any
standalone repro built to chase it. When it failed,
`ReductionOpTest.ReduceSum_apex_matrix_large` returned small integer
sequences (`1, 2, 3, 4...`) instead of sums -- a worse, less-understood
failure signature (looks like execution from stale/wrong memory, not a
numerically-off reduction) than the original bug it was meant to fix. Since
there was no fast, reliable way left to iterate (every test cycle meant a
multi-minute, sometimes-inconclusive full ORT suite run, and the standalone
repros that previously caught every other variant of this bug all came back
clean), shipping this would have traded one characterized bug for a worse,
uncharacterized one. Abandoned; the patch and its `.sh` wrapper are kept in
`patches/miopen/` for reference (both are explicitly marked as not applied
by the Dockerfile), and ReduceSum is left on the unpatched baseline.

**How to pick this up and find the real fix (or make the workaround
actually reliable):**

1. The failure is a function of *which other* dynamic-CK reduction kernels
   were compiled and cached on the same `Handle` before the failing call,
   not a property of the failing shape alone -- `m=8,9,11,13` at `n=128`
   all share the identical compiled kernel (same `reduceImpl`/tunable/
   padding bucket), yet only `m=13` (the last of that group dispatched in
   default shape-sweep order) was wrong; `m=9` and `m=11` (also needing
   padding) were fine.
2. Failing output columns aren't scattered randomly -- one valgrind run of
   the `m=13,n=128` case reproduced the failure as a single **contiguous**
   run of wrong columns (`j=32..41` out of 128), suggestive of a specific
   wavefront/SIMD-lane-group boundary rather than generic corruption.
3. No experiment ever directly observed a wrong byte inside the workspace
   descriptor or the compiled kernel's machine code for a failing call --
   every attempt to look (buffer-writes stashing decoded descriptor
   fields, a sentinel proving the prepare kernel's writes are visible to
   the main kernel) itself masked the bug by adding a memory operation to
   the kernel.
4. Next tools to try, roughly in order of likely payoff: (a) disassemble
   the actual compiled code object (`roc-obj`/`roc-obj-ls`) for the same
   `kernel_name`+`params` compiled fresh vs. compiled as the Nth kernel in
   a long sweep, to check whether the generated machine code genuinely
   differs -- if so, this is COMGR/LLVM codegen state leaking between
   separate in-process JIT compiles; (b) if the machine code is identical,
   HSA-level tracing (rocprofiler's HSA API trace, not just the HIP-level
   trace used here) to check for GPU-visible code-object address overlap
   between different loaded modules across a session; (c) `rocgdb` wasn't
   usable in this container/ROCm-version combination -- worth retrying
   bare-metal or with newer tooling, since a hardware breakpoint at the
   point of divergence would settle this quickly if it can be made to
   work; (d) fixing the static-reduction compile error is a separate,
   self-contained yes/no question that doesn't require understanding the
   dynamic-path bug at all, and would unblock testing whether routing to
   static reduction (the clean, Winograd-precedent-style fix this
   investigation deliberately avoided) is actually viable.
5. The eviction workaround's ORT-only failure is itself a clue worth
   chasing before writing off that whole approach: whatever ORT's real test
   binary does differently from every standalone repro tried (thread pool
   behavior, its own allocator reusing memory across tests, running
   hundreds of unrelated ops before reaching ReduceSum, or something else
   entirely) is *also* part of triggering the original bug's real-world
   exposure, not just an artifact of the attempted fix. A working
   bisection strategy would need to run inside (or alongside) the actual
   ORT test binary rather than a fresh standalone MIOpen process --
   e.g. `--gtest_filter` bisection to find the minimal preceding-test set
   that makes `ReduceSum_apex_matrix_large` unreliable, since the standalone
repro's inability to reproduce it suggests the trigger condition isn't
    just "many different reduce ops on one handle."

---

## SOLVED (mechanism) -- 2026-08-08: root cause is below MIOpen, in HIP allocator address recycling

A fresh pass with the committed `gfx803/tools/reduce-harness/` pinned the
mechanism. THIS is the authoritative record; the older theories in the mystery
section above were superseded one by one.

### The trigger (confirmed, controlled, at BOTH MIOpen-API and harness level)
`miopenReduceTensor`'s CK reduce kernel intermittently mis-reads its INPUT when
the input lives at a **recycled device address** -- i.e. the HIP allocator
returns the same VA (the one a previous reduce's buffer used) for the new
call's `d_x`. Same data, same kernel, same grid: copy the input to a **fresh,
never-recycled address** and the identical kernel+data+grid is 100% correct.

Evidence (harness = real CK kernels via `hipModule`, no MIOpen host code;
`HAR_MODE` selects data-address behavior, `REDUCE_HSACO_DIR=/scratch/ckbuild`):

| mode | input addr | output addr | failed/52 | conclusion |
|---|---|---|---|---|
| reuse | recycled | recycled | 10 | baseline |
| reuse-in | recycled | fresh | 5/6 | INPUT recycling is the driver |
| reuse-out | fresh | recycled | 1 | output minor |
| fresh | fresh (never repeats) | fresh | **0** | clean |
| copyin | recycled, but reduce reads a FRESH internal D2D copy | recycled | **0 (x5)** | MIOpen escape proven |
| copyin-rec | recycled, reduce reads RECYCLED internal D2D copy | recycled | **21** | observer-effect ruled OUT: recycled dest is worse |

MIOpen-via-API (seq5 repro, allfixes lib): mode0 reuse=7/52, mode1/3
fresh-distinct=0/52. Simple pure-HIP kernels (seq7/8) never trigger even on
recycled addresses -- it needs these specific CK kernels.

### What this rules out (final)
- NOT stale *data bytes*: input is freshly `hipMemcpy`'d correct every call in
  every mode; only the ADDRESS differs.
- NOT codegen: the harness loads one static `.hsaco` set; a 10x-padded code
  object changes nothing.
- NOT buffer-holding per se: a buffer held at a constant address is maximal
  reuse and still fails; only *fresh distinct* addresses are clean.
- NOT an observer effect of the D2D copy (see copyin-rec above), NOT a
  cross-kernel dispatch race, NOT a workspace/descriptor issue.

### Mechanism statement
A recycled device VA carries stale GPU memory-system state (cache/TLB/PTE
coherence) that gfx8's memory system does not fully invalidate on hipFree ->
hipMalloc reuse, and the CK kernels' input-access pattern is the first thing to
expose it. Fix belongs in the runtime: **ROCR-Runtime / libhsakmt re-map+flush
on VA reuse**, or a HIP allocator that does not immediately recycle hot VAs.
Same-address `d_x=0x924800000` re-appearing for every shape in the harness is
the smoking gun; `fresh` mode (monotonic distinct VAs) is the clean control.

### MIOpen-level escape (proven, fallback ONLY)
Reduce from a **fresh, non-recycled** internal copy of the input (MIOpen owns
the internal buffer; caller's d_x/d_y stay recycled as in real ORT/PyTorch):
0/52 stable. REQUIREMENT: the internal buffer's address must stay fresh across
a ring of **>=16 distinct buffers** -- small rings (2/4) concentrate reuse and
are WORSE (20/52, 19/52); ring8=8/52; ring16=0/52. Cost: one extra input D2D
copy, measured **+9-13%** per reduce call (HOT m=64 n=128: 172.1->187.2 us/call;
HOT m=515 n=16: 170.5->186.7; SWITCH 3 shapes: 160.3->180.6). Never ship the
small-ring version.

### Where the runtime fix goes (6.4.4 source)
- `hipMalloc`(hipamd/src/hip_memory.cpp:325) -> `SvmBuffer::malloc`(rocclr)
  -> `hsa_amd_memory_pool_allocate`(rocr hsa_ext_amd.cpp:809) ->
  `Runtime::AllocateMemory`(rocr runtime.cpp:315) -> `region->Allocate` ->
  `hsakmt_*` libhsakmt (`libhsakmt/src/memory.c`,`fmm.c`) -> KFD ioctl.
- `hipFree`(hip_memory.cpp:99) -> `FreeMemory` -> `hsakmt_fmm_release`
  (VA + physical pages return to the FastMemoryManager free-lists; next same-size
  malloc reissues the same VA).
- `clr/hipamd` has NO cache/TLB invalidation primitive (grep clean); no such
  fix exists upstream 6.4.4->7.2.4 either (only ExtendedCoherent/virtio churn).
- Primarily needed components to rebuild: `ROCm/clr@rocm-6.4.4` (e15540a9) and
  `ROCm/ROCR-Runtime@rocm-6.4.4` (37d84dc). NOTE: AMD's HIP runtime is in
  `ROCm/clr` (HIPAMD, `hipamd/`), never `hip`/`hip-runtime-amd` (headers only /
  non-existent). Monorepos (rocm-libraries/rocm-systems) have no 6.4.x tags.

### How to replicate on ROCm 7 (when the port lands)
1. Rebuild 11 code objects: `gfx803/tools/reduce-harness/build_variants.sh`
   against the ROCm 7 CK sources (adjust CK_SRC/CK_INC at top).
2. `hipcc -O1 ... miopen_reduce_kernel_harness.cpp -o reduce_harness`,
   `REDUCE_HSACO_DIR=<dir> ./reduce_harness` -> record baseline failed-count.
3. The stable signal is the FAILING COUNT (~10/52 here) + groups (n=128 mid-m
   threadwise/warpwise; n=16 m>=512 blockwise), NOT exact shapes (layout shifts
   them). A fixed ROCm 7 must show 0.
4. Same logic via MIOpen-API: `miopen_seq5_repro 0` should be >0 and
   `miopen_seq5_repro 1` should be 0 (recycled vs fresh-distinct).
5. If a runtime fix is applied, ALSO pass real ORT `ReductionOpTest` -- the two
   gates are independent (standalone has passed while ORT failed before).

### Experiment files (on the box)
- `harness_exp.cpp` (env: `HAR_MODE=reuse|fresh|reuse-in|reuse-out|copyin|copyin-rec|ring`,
  `HAR_RING=N`, `HAR_LOGADDR=0|1`, `HAR_BENCH="m,n[,m,n]"`, `HAR_BENCH_REPS=N`,
  `HAR_PRELOAD=1`) + `run_copyin.sh`, `run_bench.sh`, `/scratch/ckbuild*/`.
- Local copies: `/tmp/opencode/harness_exp.cpp` and the padded-variant
  experiment files (tmpfs -- may not persist across reboot; the important
  knowledge is recorded here and in `docs/AGENT_HANDOFF.md` §18).

## FIXED (runtime) -- 2026-08-09: ROCR va-reuse-defer patch, both gates green

The runtime fix called for above is implemented and validated:
`gfx803/patches/rocr/va-reuse-defer.patch` (applied by the sibling `.sh`,
wired into `gfx803/Dockerfile` as the `rocr-builder` stage and into CI as the
`rocr` component). It patches **ROCR-Runtime @ rocm-6.4.4** only -- libhsakmt
is statically linked into libhsa-runtime64, so no clr/HIP rebuild is needed.

### What the fix is
Two layers recycle freed GPU VAs; both are blocked:

1. **libhsakmt fmm aperture free-list** (`libhsakmt/src/fmm.c`,
   `__fmm_release`): the KFD buffer object and physical pages are still freed
   IMMEDIATELY (no memory-footprint change), but the VA stays reserved -- the
   vm_object remains in the aperture tree on a per-aperture FIFO and is only
   returned to the free-list once the FIFO exceeds 64 entries
   (`FMM_REUSE_QUEUE_MAX`). A freed address therefore can't be re-issued until
   >=64 intervening frees -- comfortably above the proven ring-16 freshness
   window -- while the queue is length-bounded and O(1), so nothing leaks
   (contrast: the earlier "never reuse VAs" leak experiment passed the harness
   but made ORT time out at 900s with 400+ failures -- VA space exhaustion).
2. **libhsa SimpleHeap fragment sub-allocator** (`core/util/flag.h`): default
   of `HSA_DISABLE_FRAGMENT_ALLOCATOR` is INVERTED (now off unless explicitly
   set to "0"). This layer caches freed 2MB blocks above libhsakmt, so the
   fmm defer cannot see its recycles: with it re-enabled (`=0`) the harness
   still fails **10/52 (3/3 runs)** even with the fmm defer active. The flip
   is load-bearing, not optional. Measured cost on the real gates: zero
   (ORT ReductionOpTest wall time unchanged, 28.4s vs 28.5s stock); it would
   only show in alloc/free-hot microbenchmarks.

### The one trap in the fix: scratch teardown
Scratch-backing apertures must bypass the defer FIFO
(`fmm_is_scratch_aperture() == aperture` -> immediate release). They are torn
down only at process exit by `fmm_release_scratch()`, whose
`while (rbtree_node_any(...))` loop requires each released object to leave the
tree immediately -- a deferred (still-in-tree) object makes that loop spin
forever: 100% user-CPU after main() completes, stack = `hsa_shut_down ->
~GpuAgent -> hsakmt_fmm_release -> fmm_release_scratch`. First patch version
shipped without the bypass and exhibited exactly that.

### Gates (gfx803 box, ROCm 6.4.4, patched lib)
- `harness_exp`, `HAR_MODE=reuse`, 52-shape sweep, 5 consecutive runs:
  **0/52 failed, ~0.7s wall each** (stock baseline: 6-10/52).
- ORT `onnxruntime_test_all --gtest_filter=*ReductionOpTest*` (317 tests),
  3 consecutive runs: **315 OK / 0 FAILED / 2 pre-existing SKIPs, ~28.5s**
  (stock: 314 OK / 1 FAILED `ReduceSum_apex_matrix_large`, 28.5s).
- Clean process exit (no teardown spin).

### Harness / test assets (canonical locations)
- Repo: `gfx803/tools/reduce-harness/harness_exp.cpp` (experiment variant with
  `HAR_MODE` etc.; builds like the committed harness:
  `hipcc -O2 harness_exp.cpp -o harness_exp -L<rocm>/lib -Wl,-rpath,<rocm>/lib`,
  run with `REDUCE_HSACO_DIR=gfx803/tools/reduce-harness/codeobj`).
- gfx803 box (persistent ext4): `/data/rocr644-ring16/` -- harness binary +
  source, `libs/` (libhsa variants: `.ba` baseline, `.r16final` shipped fix,
  `.vanoreuse`/`.leak` rejected experiments, `.orig` stock), `run_harness.sh`
  and `run_ort_reduce.sh` (exact invocations, incl. LD_PRELOAD sgemm shim,
  HSA_OVERRIDE_GFX_VERSION=8.0.3). Working copy in the `ortrun` container at
  `/scratch` (bind of `/data/claude-scratch`).

### Debugging lessons (cost us hours; don't relearn)
- **Check the soname file, not the versioned file.** libhsa loads via the
  `libhsa-runtime64.so.1` soname. In the test container that was once
  accidentally a REAL FILE (copy, not symlink), so replacing
  `libhsa-runtime64.so.1.15.0` changed nothing and we "tested" stale code for
  several rounds -- including a round where the actual fix was already built.
  Always `sha256sum $(readlink -f .../libhsa-runtime64.so.1)`.
- `docker exec` does NOT forward SIGINT into the container; `timeout -s INT`
  around a docker-exec'd process is useless. Use `timeout` INSIDE the
  container, or `docker exec ... kill -INT <pid>`.
- gdb user-CPU-spin backtrace: run the harness under `gdb -batch -ex run -ex
  "bt 30"`, then `kill -INT` from a second exec once main() output is done.
- The gfx803 image ENTRYPOINT is the audiomuse supervisord -- for any plain
  exploration use `--entrypoint sh`, and to run a idle test container use
  `--entrypoint /bin/sleep ... infinity`, or you boot a flask app that crashes
  looking for postgres.

### ROCm 7 port note
Same validation applies there: rebuild `codeobj/` for ROCm 7, run
`harness_exp` (expect 0/52 with an equivalent fix, ~10/52 without), then the
ORT ReductionOpTest gate. Whether ROCm 7's fmm/SimpleHeap still recycle
immediately has NOT been checked -- re-derive, don't assume.

### Post-ship hardening: intermittent GPU page faults (stale-handle replay)
Soak testing the shipped fix (serial ORT ReductionOpTest batches) exposed a
NEW, rarer failure: ~3/40 runs aborted mid-suite (exit 134, "Memory access
fault ... Page not present", around test ~130, `ReduceSum_batch_by_seq_by_128`).
Stock lib: 0/10. Root cause was in the defer mechanism itself, NOT the reduce
bug: a deferred release keeps the vm_object in the aperture tree **with its
KFD handles intact** (stock removes the object immediately). KFD recycles
handle ids, so a second `hsakmt_fmm_release()` of the same address -- legal in
caller-tolerance terms, and a no-op under stock ("object not found") -- would
re-issue `FREE_MEMORY_OF_GPU` on a stale handle id that KFD may already have
handed to a different, LIVE allocation: that allocation's pages get unmapped,
and its next kernel access faults "Page not present".

Fix (now part of `va-reuse-defer.patch`): when parking a release on the defer
FIFO, scrub `object->handles[]` to 0 and set `object->deferred`;
`__fmm_release()` no-ops on an already-deferred object. Any stale
map/unmap/release through the parked object now fails harmlessly instead of
cross-wiring into a live BO. Validation: harness 0/52 (unchanged), ORT
ReductionOpTest **10/10 runs: 315 OK / 0 FAILED / 0 GPU faults** (previously
~7.5% fault rate; 0/10 is consistent with fixed but not yet statistically
conclusive -- rerun the soak after any further fmm change).

Tooling for this: `run_ort_faultprobe.sh` on the gfx803 box
(`/data/rocr644-ring16/`), counts per-run `Memory access fault` occurrences;
lib variants `.r16final` (unhardened, faults) and `.r16v4` (hardened, shipped).

NOTE also a build trap hit here: iterating on `/tmp/opencode/Dockerfile.
rocr644-ring16` locally, `docker build` served a STALE `ring16.patch` from
cache despite the file changing (produced a byte-identical lib, hash checked).
Use `--no-cache` (or verify the extracted lib's sha256 against expectations)
when rebuilding the local quick-iteration image.
