# vLLM on gfx803 (Polaris) — findings

Findings from porting and tuning vLLM on an RX 470 (gfx803, Polaris, RX
470/8GiB), targeting `Qwen2.5-1.5B-Instruct` (fp16, safetensors) as the
primary test model, later extended to `Qwen3.5-2B` (hybrid mamba/linear-
attention architecture). Everything here is hardware-measured on a real
Polaris card, not guessed from reading code.

The single most useful number: **decode throughput went from ~6.5 tok/s
(~10% of llama.cpp-Vulkan's 63.39 tok/s ceiling) to 64.13-64.27 tok/s
(~101% — surpassing it) via three stacked kernel fixes, no algorithmic
change to the model.** Prefill reached ~96-101% of llama.cpp's ceiling
depending on prompt length. See `llama-cpp-gfx803/POSSIBLE_LLAMA_CPP_
IMPROVEMENTS.md` for the sibling llama.cpp-side investigation this work
grew out of.

**This vLLM path is not currently usable end-to-end.** A real hardware
erratum (gfx7/8 EOP-completion-interrupt loss — see "The blocking issue"
below) intermittently and unrecoverably wedges HIP kernel-module upload,
independent of everything documented here. All the performance work
below is real and hardware-verified in isolation; it was never able to
run for long, uninterrupted stretches because of that separate bug. See
`LAST_REMAINING_PROBLEMS.md` problem 1 and `MIGRATION_NOTES.md` for the
authoritative, most current status of that erratum — this file only
covers the parts of the investigation unique to reaching it (the
process-cascade behavior, and the relaunch-supervisor workaround), not
the full up-to-date story.

## What ships and works

### Structure: vLLM is vendored, not patched

Unlike every other component in this repo (`patches/rocblas/`,
`patches/rocm-systems/`, etc.), vLLM's gfx803 port is **not** a patch
against pristine upstream. It's vendored: a `vllm/` folder at repo root
with its own `.git`, based on `ai-infos/vllm-gfx906-mobydick @ ff063e4`
(the current tip of that fork's `main` at time of vendoring — not
`eb450bd`, a stale preset pin that's been force-pushed away on origin),
with the gfx803 port as one squashed commit on top
(`0b5207679`, "Vendor gfx803 port on top of ML-gfx906's vllm-v2 base").

Why vendored instead of patched: the port touches `CMakeLists.txt`
(`HIP_SUPPORTED_ARCHS`), 3 `csrc/` files (arch `#if` gates —
`quickreduce/base.h`, `quick_reduce_impl.cuh`, `attention/
dtype_float16.cuh`), `vllm/platforms/rocm.py` (capability/arch-detection
wiring), the GEMM/GEMV dispatch in `vllm/model_executor/layers/utils.py`,
two full custom kernels (below), and 5 attention-kernel files — real
diverging code, not a small deltas-against-upstream diff that would
survive an upstream bump cleanly.

Triton stays a small (~7-line) patch against `ai-infos/triton-gfx906`
instead (that fork itself is just upstream Triton + ~2 commits, so a
patch against it is the right convention there — unlike vLLM's much
larger fork distance from upstream).

**Not yet done**: wiring the vendored `vllm/` tree into this repo's
actual Dockerfile build. It currently only exists as a manual editable
install on the box (`/data/vllm-mobydick`, `pip3 install --no-build-
isolation --no-deps -e .`) — the vendored tree captures the *source*,
not yet the build pipeline.

### Triton on gfx803: works, 6-line ISA wiring

Triton's AMD backend deduces hardware capability from an `ISAFamily`
enum, not a hardcoded arch whitelist. gfx803 needed:
- `ISAFamily::GCN3` added to the enum (`TargetUtils.h`)
- `deduceISAFamily()` maps `GK_GFX801..GK_GFX810` → `GCN3`
  (`TargetUtils.cpp`)
- `getWarpSize()` returns 64 for GCN3 — **critical**, gfx803 is wave64;
  the default of 32 would silently produce wrong results, not an error
  (`TargetInfo.cpp`)
- `warpReduce()`'s DPP-broadcast branch extended to `isCDNA(...) ||
  ISAFamily::GCN3` — gfx803 has DPP broadcast (`FeatureDPP`,
  `hasDPPBroadcasts() = HasDPP && gen<GFX10`) but not `v_permlanex16`
  (GFX10+ only), and the un-patched else-branch assumed RDNA and emitted
  the GFX10-only instruction, producing `LLVM ERROR: Cannot select:
  %llvm.amdgcn.permlanex16` at JIT time for any kernel using a warp
  reduction (first hit via vLLM's `_topk_topp_kernel`)
- Removed two hard `"unsupported target: Unknown"` gates in
  `TritonGPUToLLVM.cpp`/`ConvertWarpPipeline.cpp`
- `supportsVDot()`/`isCDNA()`/`isRDNA()` all correctly `false` already —
  gfx803 has no matrix cores and no `v_dot4`, so Triton's existing
  scalar-fallback path is exactly right, nothing to add there

Verified on real hardware: JIT'd add-kernel `allclose` pass; `tl.dot`
GEMMs exact match (`maxerr=0.0000`) at 3 shapes; naive `BLOCK=64` GEMM at
0.589 TFLOPS fp32 vs rocBLAS `torch.matmul`'s 0.518 (~12% of the card's
~5 TFLOPS peak — real headroom for tuning, not a ceiling).

Build gotchas (LLVM side, needed to get `libtriton.so` to link at all):
`LLVM_ENABLE_RTTI=ON` (Triton's pybind module isn't built with
`-fno-rtti`; without RTTI, `libLLVMSupport.a` lacks typeinfo for
`llvm::cl::GenericOptionValue`), `LLVM_ABI_BREAKING_CHECKS=FORCE_OFF`,
`LLVM_TARGETS_TO_BUILD="X86;AMDGPU"` (Triton references X86 register
classes even for AMD-only builds), `LLVM_ENABLE_ASSERTIONS=ON` (matches
Triton's own `RelBuildWithAsserts` config), `FileCheck` copied into
`install/bin` manually (Triton's CMake requires it). Runtime needs
`TRITON_BACKENDS_IN_TREE=1` (no installed entry points from a hand-copied
install).

### Custom kernels: what routes where, and why

vLLM's default dispatch on gfx803 falls through every existing
`on_gfx906()`/GFX9+-only fast path straight to `torch.nn.functional.
linear` → rocBLAS/Tensile. **rocBLAS ships zero tuned fp16 kernels for
gfx803** — every fp16 GEMM/GEMV variant in the deployed library is named
`..._fallback_gfx803.{dat,hsaco}`, Tensile's generic portability-only
codegen, never benchmark-tuned for this arch (confirmed both from the
deployed library naming and from Tensile's own source: `AsmCaps.py`
marks every packed/HPA-capable MAC instruction — `v_dot2_f32_f16`,
`v_pk_fma_f16`, `v_mad_mix_f32`/`v_fma_mix_f32` — `False` for gfx803;
these are real Vega+ (gfx900+) silicon features Polaris never had, not a
missing config). This is *why* every fix below works — there's a real,
large gap between what's shipped and what the hardware can do, not a
tuning-nuance difference.

**Decode path (`n==1`, one token at a time)** — `gfx803_skinny_linear()`
in `vllm/model_executor/layers/gfx803_gemv.py`:
1. `ops.LLMM1` (vLLM's own existing skinny-GEMM kernel, upstream-gated to
   `on_gfx9()`/`on_gfx1x()`/`on_gfx906()` but built from portable HIP
   intrinsics — `__hmul2`/`__hfma2`/`__shfl_xor`, not GFX9+-only packed
   ASM like its sibling `wvSplitK_*` kernels in the same file — so it
   already ran on gfx803's compiled code objects with zero rebuild),
   `rows_per_block=2` (re-measured properly with 3x200-iteration median
   trials; a hastily-drawn "4 wins for o_proj" conclusion from a single
   trial was noise). Used for `K<=4096` and `M%4==0`.
2. `rocblas_hssgemv_strided_batched` (a hand-written HIP-source rocBLAS
   GEMV, not Tensile-generated) for `K>4096` (only `down_proj`, K=8960,
   in the 1.5B test model) — LLMM1 has a real, unfixed correctness bug at
   large K (see "Root-caused bugs" below).
3. `gfx803_triton_gemv` (a from-scratch, hand-tuned Triton GEMV,
   `BLOCK_M=16, BLOCK_K=256, num_warps=2` — a 48-point sweep winner at
   down_proj's exact shape) replaces (2) specifically for K>4096, ~9%
   faster than the rocBLAS GEMV there — this is the kernel that finally
   pushed decode past llama.cpp's ceiling.

Called via `ctypes` (PyTorch doesn't expose its internal rocBLAS/kernel-
launch handles to Python) — own `rocblas_handle` +
`rocblas_set_stream()` matched to Torch's current stream, confirmed
CUDA-graph-capture-safe. First real gotcha: `rocblas_operation_transpose`
is `112`, not `1` — rocBLAS's enums start at 111 to avoid collisions with
other libraries' 0/1-based ones.

**Decode attention** — `gfx803_split_decode_attention` in `vllm/v1/
attention/ops/gfx803_split_attn.py`, wired into `chunked_prefill_
paged_decode.py`, `NUM_KV_SPLITS=2` (swept 1/2/4/8 on real hardware).
Split-KV decode attention, gated to pure-decode calls only (every
sequence in the batch has `query_len==1`, no ALIBI/sliding-window/sinks/
fp8) — falls through to the unmodified original kernel otherwise.

**Prefill GEMM (`n>1`)** — `gfx803_prefill_gemm.py` +
`gfx803_gemm_lib.hip`, a genuinely hand-written HIP C++ GEMM (`hipcc
--offload-arch=gfx803 -O3`, no Tensile ASM-DSL, no Triton), double-
buffered, `__launch_bounds__(256, 1)`-pinned. Routes only `m >= 8192 and
n <= 384` (gate_up_proj-shaped weights, `M=17920` in this model, at
small prefill batch sizes) — everywhere else falls through to rocBLAS/
Tensile, which won 7 of 8 head-to-head shape comparisons once its own
fp16 codegen gap was independently fixed by a parallel effort on this
box (`patches/rocblas/tensile-gfx803-fp16-nond16.patch`). Loaded via a
transposed-weight cache (weights transposed once at first use, capped at
64MB/weight to bound VRAM cost — ~1.5GB total across this model's
layers) since the kernel wants `[K,N]` and vLLM stores `[N,K]`; a
native-`[N,K]`-layout rewrite was tried (would cover every shape, no
cache) and reverted — the transposed-cache version won every real
end-to-end benchmark despite the extra VRAM, because the native layout's
non-coalesced weight reads cost more per-call than the cache saves in
coverage.

Compiler-level lesson that unlocked all of this (**the single highest-
value finding of the whole investigation**): a "safe-looking" 4x4
register tile GEMM kernel was silently spilling 35 VGPRs/thread to slow
memory on every iteration — invisible from source, only visible via
`llvm-readobj --elf-output-style=GNU -S --notes` on the unbundled
`gfx803` code object (`.vgpr_spill_count`). Adding
`__launch_bounds__(256, 1)` (tells the compiler to target 1 threadblock/
CU instead of a higher default occupancy, trading concurrency for
register budget) eliminated the spill entirely (`.vgpr_count: 110,
.vgpr_spill_count: 0`) and turned a kernel that lost to rocBLAS by
2-4x into one that *beats* it by 3-26% at every shape it was measured
against, with zero other change. It also reversed two earlier "bigger
tile/double buffering hurt" conclusions (§29 in the archived session
log) — both had been measured against a *spilling* baseline; against a
non-spilling one, double buffering was a real additional 10-25% win on
top.

### Final numbers (Qwen2.5-1.5B-Instruct, fp16, RX 470)

| | llama.cpp-Vulkan (ceiling) | vLLM, final state |
|---|---:|---:|
| Decode (steady-state) | 63.39 tok/s | **64.13-64.27 tok/s (~101%)** |
| Prefill @ 128 tok | 607.52 tok/s | 583.77-584.78 tok/s (~96%) |
| Prefill @ 512 tok | ~607.52 tok/s (extrapolated) | 613.41 tok/s (~101%) |

Decode's climb, in order (each step hardware-verified, correctness
re-checked against `torch.nn.functional.linear` at every step): 6.5 →
13.1 (LLMM1's bias case fixed) → 21.5 (`m` cutoff relaxed) → 28.0 (`m`
cutoff removed entirely, lm_head included) → ~40 (first LLMM1 pass) →
45.08 → 58.83 (LLMM1 added on top of GEMV) → 61.82 (`rows_per_block=2`
re-tuned, fp32 round-trip removed from bias-add) → 61.98 (attention
`waves_per_eu=0`) → 62.76 (split-KV decode attention, use-after-free
race fixed — see below) → **64.13-64.27** (down_proj's LLMM1 large-K bug
worked around with a dedicated Triton GEMV).

## Root-caused bugs and their fixes

**PCIe ASPM (host firmware, not gfx803-specific)**: this board's ACPI
FADT never hands Linux OS-level ASPM ownership
(`pcie_aspm=off` alone doesn't retroactively clear firmware-set hardware
Link Control register bits — confirmed by clearing them live with
`setpci` on both the GPU endpoint and the CPU-side root port). Already
documented as a general host caveat in the main `README.md` and fixed
persistently via `tools/host-setup/gfx803-aspm-disable.service`.

**gfx7/8 missed-interrupt-wakeup kernel bug**: `kfd_events.c`'s fast
completion-lookup path trusts the interrupt payload's `context_id` as an
exact event ID whenever `signal_mailbox_updated` is true — but gfx7/8's
own interrupt handler hardcodes that flag `true` even though this
generation has no real firmware mailbox confirmation, letting a
coincidentally-signaled-looking garbage ID silently swallow (or
misdirect) the real completion. Root-caused and fixed in this repo
already (`patches/kernel/REFERENCE-amdkfd-gfx7-8-missed-interrupt-
wakeup.patch`) — **but this was found chasing a *different*, now-fixed
symptom** (a multi-hour engine-startup hang during compiled-kernel
loading); it reduced but did not eliminate the deeper, still-open EOP-
interrupt-loss erratum documented in `LAST_REMAINING_PROBLEMS.md`
problem 1 and below. Don't confuse the two — this fix is real and
should stay, but it is not the fix for "the blocking issue" below.

**`gfx803_triton_gemv` producing wrong values only under live dispatch**
(not synthetic data): root-caused via careful bisection with a live
per-call correctness shadow-check against `F.linear` — the kernel passes
every synthetic/gaussian-random test at every real shape in the model,
including down_proj's exact shape, but produces the `notably!!!!!!!!!`
degenerate-repetition failure mode under real generation. Never fully
root-caused (leading theory: CUDA-graph-capture interaction, since the
kernel allocates its output tensor fresh via `torch.empty()` on every
call inside a captured decode graph — not confirmed). **Fixed by not
using it there**: `gfx803_skinny_linear` never dispatches to
`gfx803_triton_gemv` for K>4096 in the decode path (only for
down_proj-shaped *prefill*-adjacent large-K work, a separate call
site/context where it tests correct) — `rocblas_hssgemv_strided_batched`
remains the K>4096 decode fallback. Diagnostic env-var toggles used to
bisect this were all removed from the shipped code.

**`ops.LLMM1`'s real correctness bug at K∈[8064,8192]**: the host
launcher computes `NUM_THREADS ≈ ceil_to_64(K/8)` with no upper-bound
check against gfx803's 1024-thread/block hardware limit. At K=8064/8192
(right at/near that 1024 boundary) the kernel launches without error but
returns catastrophically wrong values (`max_abs_diff` ~2-4.5, not fp16
noise); at K=8960 it fails outright with `hipErrorInvalidConfiguration`.
Leading theory (not confirmed by disassembly): register-spilling/
occupancy collapse at that width, and/or a warp-shuffle reduction stage
ported from CUDA's 32-lane warps without adjusting for AMD's 64-lane
wavefronts. Standalone repro saved: `tools/vllm-diagnostics/
repro_llmm1_bug.py`. **Fixed by gating around it, not patching
`skinny_gemms.cu`** (that would need rebuilding vLLM's C++/HIP extension
— slower and riskier than writing a kernel that doesn't share the bug):
LLMM1 used only for `K<=4096`.

**Split-KV decode attention: a real use-after-free race, not a math
bug.** The split kernel pair was silently corrupting *unrelated* GPU
memory (caught via a canary-tensor bracketing test, not a numerical
diff — both kernels matched a pure-PyTorch reference to ordinary fp16
noise on cases that "looked correct"). Root cause:
`partial_out`/`partial_m`/`partial_l`, allocated fresh via `torch.
empty()` inside the wrapper function, get Python-refcount-freed at
function return while the reduce kernel launched just before may *still*
be asynchronously reading them — this hardware's caching allocator does
not reliably defer reuse of a buffer a still-running async kernel is
using. Fix, two parts, both required: (1) cache the intermediate buffers
in a module-level dict keyed by shape, never free them; (2) a same-
stream `torch.cuda.Event()` record/`wait_event()` barrier between the
two kernel launches (a host `cuda.synchronize()` works too but is
illegal during CUDA-graph capture — confirmed directly,
`hipErrorStreamCaptureUnsupported`). 0/16 corrupted after, vs.
consistently 7-16/16 before.

## Dead ends — don't re-attempt these

- **Naive `tl.dot`-loop Triton GEMM for prefill** (no pipelining, no
  tiling engineering): loses to Tensile's *untuned fallback* by 3.5-5x.
  Unlike decode's GEMV case, prefill's M=128-512 shapes are ones
  Tensile's fallback already handles competently — there's no free lunch
  the way there was at M=1.
- **Multi-stage software pipelining** (`num_stages=2/3` in Triton, or
  explicit double-buffering in hand-written HIP *before* the register-
  spilling fix) — hurts, consistently, across three independent
  implementations. (Caveat: double buffering *after* the spilling fix
  is a real win — see above. The pipelining-specific `num_stages` result
  in Triton was never re-tested against a non-spilling baseline.)
- **Bigger register tiles** (8x8 vs 4x4 in hand-written HIP): ~5x
  regression pre-spill-fix, still a real (if smaller, 0.93x) loss even
  after fixing spilling — the one technique that stayed bad both times.
- **`BLOCK_K` 16→32** in the native-layout prefill GEMM rewrite: verified
  non-spilling (176 VGPRs, 0 spills) but regressed every shape anyway —
  larger register footprint likely hurt ILP more than fewer sync
  barriers helped.
- **Tensile fp16-HPA (`HighPrecisionAccumulate: True`) for gfx803**:
  architecturally impossible with Tensile's current kernel library, not
  a config gap. Every HPA-capable MAC component requires
  `v_dot2_f32_f16`/`v_pk_fma_f16`/`v_mad_mix_f32` — real Vega+ (gfx900+)
  silicon Polaris never had (confirmed in Tensile's own `AsmCaps.py` and
  independently in a 2020 upstream commit's own comment: `# No HPA on
  803, every other combination should work though.`). Plain (non-HPA)
  fp16 gets further (real assembly generated, reaches the actual
  assembler) but then fails on `d16_hi`/packed-store instructions with
  *no* existing capability gate or fallback path in Tensile's kernel
  writer — genuine, unscoped assembly-codegen engineering, not a config
  flag. Scoped in detail for whoever picks it up:
  `patches/rocblas/TENSILE_GFX803_FP16_TODO.md`.
- **`rocprofv3 --kernel-trace`**: unreliable on this stack, hung 3+ times
  under different trigger conditions (concurrent `hsa_executable_freeze`
  calls from parallel kernel-loader threads; a separate hang recurring
  even with loading forced single-threaded). Real per-kernel GPU timing
  when needed came from `torch.profiler`'s engine-side hook instead
  (`llm.start_profile()`/`stop_profile()` + `profiler_config={"profiler":
  "torch", ...}` passed to the `LLM()` constructor — the env var alone,
  `VLLM_TORCH_PROFILER_DIR`, is not suffient on this vLLM version). Note:
  this profiler's CUDA-activity trace never populated device-kernel
  events on this ROCm/PyTorch build either (CPU-side `cuda_runtime`
  categories only) — there is currently no reliable way to get real
  per-kernel *GPU* execution time on this stack; `hipEventSynchronize`
  totals (CPU time blocked waiting on the GPU) were used as the best
  available proxy throughout.
- **HIP graph capture for decode**: works (after
  `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` — see the llama.cpp findings
  doc for the segmented-path signal-pool bug this works around) but
  gives negligible standalone speedup once GEMM/attention are already
  fast; the real decode win came entirely from the kernel fixes above,
  not from graph capture itself.

## The blocking issue: gfx7/8 EOP-interrupt loss (open, unresolved)

This is the same erratum documented in `LAST_REMAINING_PROBLEMS.md`
problem 1 and `MIGRATION_NOTES.md` — **read those for the current,
authoritative status**, including a much sharper live PM4-trace/fence-
state localization done in a later session, and confirmation that
enabling the existing mitigation (`ROCR_GFX8_EOP_MITIGATION=1`, or the
untracked `ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US` variant) can cause
outcomes *worse* than the hang itself (a full unrecoverable GPU bus
death, and separately a full system hard lockup with no network
response). This section only adds the parts of the investigation unique
to reaching that bug from the vLLM side, not already in those files:

**Deep kernel-level tracing (five independent layers, all confirmed
clean) ruled out every software-side explanation for the interrupt
loss** — IH ring wptr/rptr genuinely advance together at the exact
instant the process is hung; every kernel-side accept/reject path
(`kfd_events.c`'s exhaustive scan, `cik_event_interrupt_isr`'s
vmid/pasid guards, `amdgpu_irq_dispatch`'s validation, both IP-block
interrupt handlers, the KFD node-loop race) traced clean via
`pr_debug_ratelimited` instrumentation and ftrace kretprobes; userspace
ROCr's own AQL packet construction (`BlitKernel::SubmitLinearCopyCommand`
→ `PopulateQueue()`) confirmed structurally correct and unremarkable
(if this were a packet-construction bug, every dispatch through this
path would fail deterministically, not intermittently). The conclusion:
a genuine gfx7/8 firmware erratum in EOP-interrupt generation, not
reachable from any software layer between kernel driver source and
userspace HSA packet construction.

**Confirmed: the wedge is per-*process*, not per-queue or per-dispatch.**
Once one `BlitKernel::SubmitLinearCopyCommand` (nearly always the
one-time `hipModuleLoadData` code-object upload — identified via a live
backtrace, and consistent with the failure always being exactly 12288
bytes) fails in a process, *every* later dispatch in that process fails
too — on any queue, freshly created or not. Three independent recovery
strategies were tried and measured on real hardware, all refuted:
- **Same-queue signal retry**: 0/9 recovered.
- **Queue recreation** (fresh queue via a new `GpuAgent::
  CreateBlitRecoveryQueue()` hook): 8/12 still dead; `RecreateQueue()`
  itself never failed, so the wedge isn't tied to one physical queue
  either.
- **Queue warmup** (absorb the hazard with a throwaway dispatch before
  real data rides on a queue): ~37% clean, no better than baseline; two
  sampled failing runs directly refuted the "first dispatch only" theory
  — the real copy was already that queue's *second* dispatch and still
  failed.
- **Submission-rate throttling** (a delay between consecutive dispatches,
  20us through 10ms): also refuted with a properly-sized sample (n=10 at
  1ms and 10ms converge on the same ~20% baseline rate; 10ms is 500x the
  one delay that looked promising at n=5-6, and shows *zero* improvement
  over it — not what a real race-window fix would look like).

**Practical workaround shipped, explicitly a workaround not a fix (per
this repo's own "fix at source" standing rule, which the maintainers
agreed doesn't apply here since the root cause is unreachable from any
software layer)**: `tools/host-setup/vllm-relaunch-supervisor.sh`.
Wraps a vLLM launch command; watches only for the specific `giving up`
failure marker during an early detection window (default 180s —
anything else, including an unrelated process exit, passes straight
through unretried); on the known wedge, kills the process (including
the grandchild `EngineCore`, not just the direct child), waits for VRAM
to actually drain (`rocm-smi`-polled — a too-fast relaunch into still-
held VRAM produces a spurious, different-looking "insufficient GPU
memory" failure), and relaunches up to 8 times. Validated against a
real Qwen3.5-2B `bench latency` workload: **6/6 clean** end-to-end
(individual launches only clear ~30-35% on their own; the supervisor's
relaunch compounds that to the target reliability). This script exists
and works; whether it's still the right approach given later sessions'
findings (documented in `LAST_REMAINING_PROBLEMS.md`) is a decision for
whoever picks this back up.

## How to reproduce

All commands assume the box (`192.168.1.214`, `user`/`user`) with either
the `ct-old` container or the native (no-container) install set up per
`tools/host-setup/native-rocm-vllm-setup.sh` and its README section.

**Sanity generation** (coherence check, always run after any kernel
change before trusting a speed number):
```
cd /root && /opt/venv/bin/python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/data/qwen2.5-1.5b', gpu_memory_utilization=0.7,
          max_model_len=1024, dtype='float16',
          compilation_config={'cudagraph_capture_sizes': [1]})
for p in ['The capital of France is', '2 + 2 =']:
    out = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=32))
    print(repr(out[0].outputs[0].text))
"
```
Never `dtype='bfloat16'` on gfx803 — bf16 is not native here and hangs
the very first bf16 kernel. Always `float16`.

**Isolated decode-only throughput** (minimizes prefill contamination —
single short prompt, forced decode length, wall-clock timed):
```
# pattern used throughout this investigation; see the archived
# session log for the exact decode_only_bench.py this repo's
# session history refers to (not currently checked in — recreate
# from this pattern if needed: LLM(...), one short prompt, N forced
# decode tokens via SamplingParams(min_tokens=N, max_tokens=N), time
# the generate() call, divide by N).
```

**Real end-to-end latency** (the number quoted throughout this doc):
```
/opt/venv/bin/python3 -m vllm.entrypoints.cli.main bench latency \
  --model /data/qwen2.5-1.5b \
  --input-len 128 --output-len 64 --batch-size 1 \
  --num-iters-warmup 2 --num-iters 3 --dtype float16 \
  --gpu-memory-utilization 0.7 --max-model-len 1024 \
  --cudagraph-capture-sizes 1
```
`--cudagraph-capture-sizes 1` is required on an 8GiB card — the bench
harness's 51-size default OOMs regardless of any kernel change here.

**Wrapped with the crash-recovery supervisor** (for a workload expected
to hit the EOP hang):
```
tools/host-setup/vllm-relaunch-supervisor.sh \
  /opt/venv/bin/python3 -m vllm.entrypoints.cli.main bench latency \
  --model /data/qwen2.5-1.5b ...  # same args as above
```

**Verifying a kernel change didn't regress correctness**, the pattern
used for every fix in this doc:
```python
import torch
ref = torch.nn.functional.linear(x, weight, bias)
out = gfx803_skinny_linear(x, weight, bias)  # or whichever kernel
print((ref - out).abs().max().item())  # expect ~1e-4 to ~1e-5, fp16-noise order
```

**Extracting and inspecting a hand-written kernel's real register usage**
(the technique that found the register-spilling breakthrough — use this
*before* concluding a hand-written HIP kernel "should" be fast enough):
```
hipcc --offload-arch=gfx803 -O3 -o mybench mybench.hip
objcopy -O binary --only-section=.hip_fatbin mybench mybench.fatbin
/opt/rocm/core-7.14/lib/llvm/bin/clang-offload-bundler -unbundle \
  -type=o -targets=hipv4-amdgcn-amd-amdhsa--gfx803 \
  -input=mybench.fatbin -output=mybench-gfx803.o
/opt/rocm/llvm/bin/llvm-readobj --elf-output-style=GNU -S --notes \
  mybench-gfx803.o | grep -E 'vgpr|sgpr|group_segment'
```
Any `.vgpr_spill_count` above 0 means try `__launch_bounds__(threads,
minBlocksPerMultiprocessor)` before anything else.

## Open items (as of the last session that touched this work)

1. **The EOP-interrupt-loss hang is the actual blocker** — see
   `LAST_REMAINING_PROBLEMS.md` problem 1 for the current state; nothing
   in this document changes that status.
2. `qkv_proj` bias support for the prefill GEMM kernel — currently
   excluded entirely (`bias is None` gate), meaning one of the four
   linear layers gets none of the prefill speedup.
3. Whether `__launch_bounds__`/occupancy tuning helps the *Triton*
   kernels (`gfx803_split_attn.py`, `gfx803_triton_gemv`) the same way
   it transformed the hand-written HIP GEMM — flagged repeatedly across
   sessions, never actually checked.
4. Root-cause (not just gate around) `ops.LLMM1`'s large-K correctness
   bug, and `gfx803_triton_gemv`'s live-vs-synthetic discrepancy — both
   currently worked around, neither understood.
5. Real concurrent-request throughput (`vllm bench throughput`) never
   measured — everything in this document is single-request (batch=1)
   latency. At batch=1, decode is fundamentally memory-bandwidth-bound
   (confirmed: `lm_head` alone sits close to the card's ~211GB/s
   bandwidth floor) — batching is the standard, not-yet-measured lever
   for pushing past this ceiling in aggregate serving capacity.
6. Whether `Qwen3_5ForConditionalGeneration`'s hybrid mamba/linear-
   attention/vision architecture fully works on this stack is still not
   completely confirmed — a first successful end-to-end run happened,
   but was interrupted by the EOP hang before extensive validation, and
   an output-quality question on that model was open at the time (see
   `LAST_REMAINING_PROBLEMS.md`/`MIGRATION_NOTES.md` for whether this
   was later resolved).
7. The user's longer-term goal (4-6x RX 470/580 8GB cards, multi-GPU,
   larger models) is entirely unstarted — every number in this document
   is single-GPU.
8. Wiring the vendored `vllm/` tree into this repo's actual Dockerfile
   build (currently a manual box-only editable install).
