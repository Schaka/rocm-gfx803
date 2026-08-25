NOT an auto-applied patch (vLLM isn't built from source in this repo's
Dockerfile -- it's a box-only editable install per SESSION_HANDOFF.md §7,
"vendored fork" is still a pending decision). This file documents the
exact change, verified working on real hardware, to be turned into a real
`git apply`-able patch once the vendored-fork step happens.

WHY -- decode attention's Triton fallback kernel (`kernel_paged_attention_2d`
in `chunked_prefill_paged_decode.py` -- the only decode-attention path on
gfx803, since `use_rocm_custom_paged_attention()` MFMA-gates the ROCm
custom kernel to gfx90a/942/950/gfx11x) launches grid `(num_seqs,
num_kv_heads)`. For Qwen2.5-1.5B-Instruct that's `(1, 2)` -- 2
threadblocks on a 32-CU RX 470, each looping sequentially over every KV
block in the sequence. Every other kernel fixed this session (see
`gfx803-skinny-gemv.patch.md`) was bandwidth/launch-bound at M=1; this is
the one kernel whose own parallelism is structurally capped far below the
hardware's width, independent of how fast each block's inner loop runs.

WHAT -- `gfx803_split_attn.py` (shipped alongside this file, copy exact):
a second Triton kernel pair raising the grid to `(num_seqs, num_kv_heads,
NUM_KV_SPLITS)`. `kernel_paged_attention_2d_split` copies
`kernel_paged_attention_2d`'s K/V paged-cache addressing verbatim
(already proven correct on this hardware) but restricts each program to a
`[split_start, split_start+blocks_per_split)` slice of KV blocks, writing
partial online-softmax state `(acc, M, L)` to an intermediate buffer.
`kernel_paged_attention_2d_reduce` does a standard log-sum-exp merge
across splits to the final output.

REAL BUGS FOUND AND FIXED during development (see SESSION_HANDOFF.md §23
for the full investigation writeup) -- none of these were guesswork,
each was caught by a rigorous correctness/memory-safety test before being
accepted as fixed:

1. A Triton compile error from scalar-indexing a tensor
   (`alpha[s]`) inside a Python `for s in range(NUM_KV_SPLITS)` loop in
   the reduce kernel -- fixed by loading the whole `[NUM_KV_SPLITS,
   HEAD_SIZE_PADDED]` tile at once and reducing via
   `tl.sum(part * alpha[:, None], axis=0)`.
2. An intermediate-buffer sizing bug: the address arithmetic for the
   masked-off (never-written) padding lanes of the last kv head can
   still reach indices past `num_query_heads`, so sizing the buffer by
   `num_query_heads` risked address computation (not the store itself,
   which is masked) exceeding the buffer, and for `num_seqs>1` could
   alias a different sequence's region. Fixed with a `HEAD_SLOTS =
   num_kv_heads * num_queries_per_kv_padded`-sized addressing scheme
   used consistently in both kernels.
3. Two failed loop-restructuring attempts (a dynamic-`start` range
   `for j in range(split_start, split_end)`, and a fixed `range(0,
   num_blocks)` with an `if (j >= split_start) & (j < split_end):`
   guard around the loop-carried `acc`/`M`/`L` updates) both produced
   wrong results on this backend even at `NUM_KV_SPLITS=1` (which should
   be mathematically identical to the unsplit reference). Fixed by
   matching the *exact* pattern already proven correct in the original
   kernel: a dynamic-*end*, fixed-*start-at-0* loop
   (`for jj in range(0, blocks_per_split): j = split_start + jj`), no
   `if` at all -- letting the existing `seq_mask` (already how the
   original kernel handles its own final partial block) naturally zero
   out any tail overrun, with a `j_safe = tl.minimum(j, num_blocks - 1)`
   clamp on the block-table read for safety.
4. A real, confirmed, reproduced **use-after-free race**: `partial_out`/
   `partial_m`/`partial_l` allocated via `torch.empty()` local to the
   wrapper function were being reused by the caching allocator for
   unrelated tensors (confirmed via a canary-tensor test: a tensor that
   neither kernel ever touches got clobbered) while the reduce kernel was
   still asynchronously reading them -- reproducible specifically
   through a Python function call boundary (inline top-level-script code
   was consistently clean; the identical code wrapped in a function was
   consistently ~44% corrupted across repeated runs). This hardware's
   caching allocator does not reliably defer reuse of a just-freed buffer
   until the GPU has actually finished an async kernel still reading it.
   Fixed two ways, combined: (a) the buffers are now cached and reused
   forever (keyed by shape) rather than freed and reallocated every call
   -- see `_get_partial_buffers()` -- which also gives CUDA-graph
   capture/replay the static addresses it needs; (b) a `torch.cuda.Event`
   record+wait barrier between the two kernel launches (confirmed
   necessary even with (a) in some configurations). A host
   `torch.cuda.synchronize()` also fixes it but is illegal during CUDA
   graph capture (`hipErrorStreamCaptureUnsupported`, confirmed directly
   on this stack) -- the event-based barrier is capture-legal (confirmed
   directly) and was used instead.

Verified via: a standalone correctness harness comparing against
`kernel_paged_attention_2d` directly (not through the engine) across
seq_len 17/63/129/257/400 and `NUM_KV_SPLITS` 1/2/4/8 -- all differences
now bounded to ordinary fp16/fp32 accumulation-order noise (max ~4.5e-2,
typically ~1e-5, deterministic/reproducible, not the inf/nan/thousands
seen before the use-after-free fix); a pure-PyTorch (no Triton) reference
implementation, to settle which kernel was actually right during
debugging (both were, once fixed); a canary-tensor memory-safety test
across the same sweep, 0/16 corrupted across multiple repeated runs
post-fix (vs. consistently 7-16/16 corrupted before); real end-to-end
generation via `sanity_gen.py` (coherent, correct, bit-for-bit identical
completions across every change); and `vllm bench latency`/isolated
decode-throughput benchmarks on real hardware.

`chunked_prefill_paged_decode.py` changes:

```python
# near the top, alongside the other same-directory imports:
from .gfx803_split_attn import gfx803_split_decode_attention

# inside chunked_prefill_paged_decode(), immediately after:
#     block_size = value_cache.shape[3]
#     num_seqs = len(seq_lens)
# and before the rest of the existing decode-dispatch logic:
    if (
        on_gfx803()
        and query.shape[0] == num_seqs
        and alibi_slopes is None
        and sliding_window == 0
        and sinks is None
        and output_scale is None
        and "fp8" not in kv_cache_dtype
    ):
        gfx803_split_decode_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=sm_scale,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_size=min(block_size, 128) if (block_size & (block_size - 1) == 0) else 32,
            num_kv_splits=2,
        )
        return
```

Gated to a pure-decode call (`query.shape[0] == num_seqs`, i.e. every
sequence in this call contributes exactly one token) with none of
ALIBI/sliding-window/sinks/fp8-output active -- `gfx803_split_decode_attention`
doesn't implement per-sequence query-length filtering the way the
original kernel does (`filter_by_query_len`), so a batch mixing
chunked-prefill continuations with decode requests falls through to the
unmodified original path, unchanged and untouched by this patch.

`NUM_KV_SPLITS=2` was chosen from a direct sweep on real hardware
(isolated decode-throughput benchmark, median of repeated trials):

| NUM_KV_SPLITS | tok/s |
|---|---|
| (baseline, no split kernel) | 61.98 |
| 1 | 61.72 |
| **2** | **62.76** |
| 4 | 62.56 |
| 8 | 62.21 |

`NUM_KV_SPLITS=1` is slightly *worse* than baseline (extra kernel-launch
and reduce-merge overhead with zero added parallelism, since 1 split
doesn't raise the threadblock count above the original's). 2 wins; higher
split counts start losing to per-split fixed overhead outweighing the
added parallelism at this model's short (~192-token) decode context.

STATUS: 62.76 tok/s decode-only on real gfx803 hardware, up from 61.98
tok/s without this kernel -- **~99% of llama.cpp-Vulkan's 63.39 tok/s
ceiling**, and real `vllm bench latency` total latency (128-token prefill
+ 64-token decode) 1.4076s -> 1.3039s (~7.4% faster). Verified working
end-to-end on real hardware. Not yet promoted to an auto-applied
Dockerfile patch -- gated on the "vendored vLLM fork" decision
(SESSION_HANDOFF.md §7), same as `gfx803-skinny-gemv.patch.md`.
