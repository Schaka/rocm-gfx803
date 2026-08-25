NOT an auto-applied patch (vLLM isn't built from source in this repo's
Dockerfile -- it's a box-only editable install per SESSION_HANDOFF.md §7,
"vendored fork" is still a pending decision). This file documents the
exact change, verified working on real hardware, to be turned into a real
`git apply`-able patch once the vendored-fork step happens (no pristine
upstream baseline was kept on the box to diff against -- reconstruct
against whatever vLLM commit that fork pins).

WHY -- see `gfx803_gemv.py` (shipped alongside this file, copy exact) for
the full investigation. Short version: rocBLAS/Tensile ships zero tuned
fp16 GEMM kernels for gfx803 -- every fp16 GEMM call falls through to
Tensile's generic, never-benchmarked `_fallback_` library, which
dispatches a 128-row-tile kernel even for decode's actual M=1 shape.
Fixed with two combined kernels, chosen per-shape:

1. `rocblas_hssgemv_strided_batched` -- rocBLAS's hand-written (non-
   Tensile) GEMV kernel, already shipped in `librocblas.so`. 3-32x faster
   than the untuned-GEMM baseline on every measured shape.
2. `ops.LLMM1` -- vLLM's own existing skinny-GEMM kernel
   (`csrc/rocm/skinny_gemms.cu`), upstream-gated to gfx9/gfx11/gfx906
   only but built from portable HIP intrinsics with no GFX9+-only
   instructions, and already compiled into this stack's `_rocm_C.abi3.so`
   for gfx803 (no rebuild needed). 1.6-2x faster again than option 1, but
   has a **real correctness bug at larger K** (verified: K=8192/8064 --
   both within vLLM's own stated `k<=8192` safety gate -- launch cleanly
   but return catastrophically wrong results, `max_abs_diff` ~4.2-4.5;
   K=1536 verified correct up to M=151936). Gated to `K<=4096` here,
   safely below the failure point.

Routing decode through the combined path took real `vllm bench latency`
total latency from 9.96s to 1.4634s (6.8x) on a 128-token-prefill +
64-token-decode run. Isolated decode-only measurement: 58.83 tok/s (from
a ~6.5 tok/s baseline), ~92.8% of llama.cpp-Vulkan's 63.39 tok/s ceiling
on the same hardware (up from ~10% at session start). Correctness
verified against `torch.nn.functional.linear` for every shape actually
used by this model, and against real generation output (coherent,
correct completions, bit-for-bit identical across kernel-choice changes
for the same sanity prompts).

Two follow-up fixes, folded into `gfx803_gemv.py` (this directory's copy
is current -- diff it against the box if this file's numbers look stale):
`rows_per_block=2` (not 4) for `ops.LLMM1` -- a single-trial sweep first
suggested 4 was faster for o_proj specifically, but that was measurement
noise; median-of-3x200-iteration trials showed 2 winning uniformly, 1-4%
real. And the bias-add after LLMM1 no longer round-trips through fp32 --
LLMM1 already returns fp16 matching the bias tensor's dtype, so the add
happens directly. Combined with the attention-kernel tuning below, decode
reached 61.98 tok/s, ~97.8% of llama.cpp's ceiling.

Separately, `chunked_prefill_paged_decode.py`'s decode-path Triton launch
kwargs (`triton_launch_kwargs`, gated `on_gfx906() or on_gfx803()`):
`waves_per_eu` changed from `1` to `0` (let the compiler pick occupancy
instead of pinning it) -- 0.2% faster decode throughput, median of
repeated real `vllm bench latency` runs (`Avg latency` 1.4138s->1.4076s).
Every other launch-param combination tried (`num_warps=2/8`,
`waves_per_eu=2`, `num_stages=2`) measured worse and was reverted -- see
SESSION_HANDOFF.md for the full sweep. This is the one line actually
changed in that file; no diff shown here since the surrounding function is
long and unrelated to the gfx803_gemv.py change above.

A deeper attention-kernel investigation (split-KV decode attention, to
raise the decode-attention grid from `(num_seqs, num_kv_heads)` --
2 threadblocks on this 32-CU GPU -- to `(num_seqs, num_kv_heads,
NUM_KV_SPLITS)`) was attempted, hit a real correctness bug, root-caused
(a genuine use-after-free race in the intermediate buffers, not a logic
bug), fixed, and shipped -- see `gfx803-split-attention.patch.md` and
SESSION_HANDOFF.md §23-24 for the full investigation.

**Third fix, `gfx803_triton_gemv`**: down_proj (K=8960) exceeds LLMM1's
verified-safe K<=4096 ceiling, so it was still routing through the slower
`rocblas_hssgemv_strided_batched` path. Root-caused LLMM1's large-K bug
far enough to be confident it's a real bug in `skinny_gemms.cu`'s host
launcher (`NUM_THREADS` computed with no upper-bound check against the
hardware's max threads/block -- see `repro_llmm1_bug.py`, shipped
alongside this file, for a minimal standalone repro and the full
writeup), but fixing it means rebuilding vLLM's C++/HIP extension --
bigger and riskier than writing a kernel that avoids the bug entirely.
`gfx803_gemv.py` gained `gfx803_triton_gemv`, a plain Triton GEMV with no
warp-shuffle assumptions, now used for K>4096 in place of
`rocblas_hssgemv_strided_batched` -- ~9% faster at down_proj's exact
shape (tuned `BLOCK_M=16, BLOCK_K=256, num_warps=2` via a 48-point
sweep). Combined with the split-attention fix, decode reached
**64.13-64.27 tok/s, surpassing llama.cpp-Vulkan's 63.39 tok/s ceiling**
for the first time this session (SESSION_HANDOFF.md §25).

WHAT
----
1. New file: `vllm/model_executor/layers/gfx803_gemv.py` (copied
   verbatim into this directory -- see that file for the full module,
   including the LLMM1 correctness-bug writeup and the exact K/M gating
   logic in `gfx803_skinny_linear()`).

2. `vllm/model_executor/layers/utils.py`, `rocm_unquantized_gemm_impl()`:
   added an early gfx803-specific branch, before the existing
   `on_gfx906()`/`use_skinny` logic (which doesn't gate gfx803 at all --
   this arch fell through to plain `torch.nn.functional.linear`, i.e.
   the untuned Tensile GEMM path, for every shape):

```python
def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx1x, on_gfx803, on_gfx9, on_gfx906, on_gfx950

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    # gfx803: Tensile ships no tuned fp16 GEMM for this arch at all (every
    # fp16 shape falls through to Tensile's untuned generic `_fallback_`
    # library -- see gfx803_gemv.py's module docstring for the full
    # investigation). Route decode's actual M=1 shape through gfx803's
    # combined LLMM1/rocBLAS-GEMV fast path instead -- see that module for
    # per-kernel gating (LLMM1 has a real correctness bug at K>4096,
    # gated around there, not here).
    if (
        on_gfx803()
        and n == 1
        and (bias is None or bias.dtype == torch.float16)
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and weight.is_contiguous()
    ):
        from vllm.model_executor.layers.gfx803_gemv import gfx803_skinny_linear

        out = gfx803_skinny_linear(x.reshape(-1), weight, bias).to(x.dtype)
        return out.reshape(*x.shape[:-1], m)

    if not on_gfx906():
        # ... rest of function unchanged ...
```

Scoped to `n == 1` (decode) only -- prefill/chunked-prefill (`n > 1`)
keeps using the existing GEMM path unchanged. Not yet separately measured
whether prefill would also benefit from a similar fix (prefill was
already found close to llama.cpp's ceiling, ~8% behind, so likely lower
priority) -- see SESSION_HANDOFF.md's NEXT list.

STATUS: verified working end-to-end on real gfx803 hardware (real vLLM
`bench latency` run, real text generation, correctness-checked for both
kernels). Not yet promoted to an auto-applied Dockerfile patch -- gated
on the "vendored vLLM fork" decision (SESSION_HANDOFF.md §7).
