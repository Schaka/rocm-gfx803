"""gfx803 split-KV decode-attention: raises threadblock count from
(num_seqs, num_kv_heads) to (num_seqs, num_kv_heads, NUM_KV_SPLITS).

kernel_paged_attention_2d (chunked_prefill_paged_decode.py) launches only
`num_seqs * num_kv_heads` programs -- for this model (num_seqs=1,
num_kv_heads=2) that's 2 threadblocks on a 32-CU gfx803 GPU, each looping
sequentially over every KV block in the sequence. Every other kernel in the
decode step is bandwidth/launch-bound at M=1; this one is the sole kernel
whose own work item count could trivially scale with hardware width instead
of being fixed at num_seqs*num_kv_heads. Splits the per-program KV-block
loop range across a 3rd grid dimension (online-softmax partial per split),
then merges partials with a standard log-sum-exp combine -- the K/V
addressing math is copied verbatim from kernel_paged_attention_2d (already
verified correct on this hardware), the only new logic is the loop-range
slice and the merge.

Decode-only real-hardware result (Qwen2.5-1.5B-Instruct, ~192-token
context): NUM_KV_SPLITS=2 measured fastest of {1,2,4,8} (median-of-N
`vllm bench latency`/isolated decode-throughput trials) -- 62.76 tok/s vs
61.98 tok/s without this kernel, ~99% of llama.cpp-Vulkan's 63.39 tok/s
ceiling on the same hardware. See SESSION_HANDOFF.md for the full
investigation, including a real, confirmed, and fixed use-after-free race
this kernel pair hit during development (not a logic bug -- see the long
comment at the `_split_reduce_barrier` call site below).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d_split(
    partial_out_ptr,  # [num_seqs, num_query_heads, NUM_KV_SPLITS, HEAD_SIZE_PADDED]
    partial_m_ptr,  # [num_seqs, num_query_heads, NUM_KV_SPLITS]
    partial_l_ptr,  # [num_seqs, num_query_heads, NUM_KV_SPLITS]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_SLOTS: tl.constexpr,  # num_kv_heads * num_queries_per_kv_padded
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_id = tl.program_id(2)

    cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    # slot_idx (unlike query_head_idx) stays < HEAD_SLOTS even for the
    # masked-off padding lanes of the last kv head -- used for the
    # partial-buffer address so a masked (unwritten) lane's pointer
    # arithmetic never aliases the next seq_idx's region.
    slot_idx = kv_head_idx * num_queries_per_kv_padded + tl.arange(
        0, num_queries_per_kv_padded
    )
    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
    blocks_per_split = cdiv_fn(num_blocks, NUM_KV_SPLITS)
    split_start = split_id * blocks_per_split

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    # Dynamic-end (start=0) for-loop -- the one pattern already proven
    # correct on this backend (kernel_paged_attention_2d's own `for j in
    # range(0, num_blocks)`). A dynamic-START range (`range(split_start,
    # split_end)`) and an `if`-guard wrapping loop-carried acc/M/L updates
    # inside a fixed range(0, num_blocks) both produced wrong results on
    # this ROCm/Triton backend even at NUM_KV_SPLITS=1 (see
    # SESSION_HANDOFF.md) -- likely a codegen issue with either dynamic
    # loop start or with predicated loop-carried state, not the split math
    # itself. Iterate 0..blocks_per_split unconditionally, compute the real
    # block index by offsetting inside the body instead. j can run past
    # num_blocks-1 only on the last split (when blocks_per_split doesn't
    # evenly divide num_blocks); j_safe clamps the block-table read so that
    # tail overrun never reads out of the table, while seq_mask (computed
    # from the unclamped j) is already what makes such tail blocks
    # contribute nothing -- exactly the same guarantee the original,
    # proven-correct kernel relies on for its own final partial block.
    for jj in range(0, blocks_per_split):
        j = split_start + jj
        j_safe = tl.minimum(j, num_blocks - 1)
        start_n = j_safe * BLOCK_SIZE
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        K = tl.load(
            key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0,
            eviction_policy="evict_last",
        )
        V = tl.load(
            value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0,
            eviction_policy="evict_last",
        )

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    out_base = (
        seq_idx * HEAD_SLOTS * NUM_KV_SPLITS
        + slot_idx * NUM_KV_SPLITS
        + split_id
    )
    tl.store(
        partial_out_ptr
        + out_base[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )
    tl.store(partial_m_ptr + out_base, M, mask=head_mask)
    tl.store(partial_l_ptr + out_base, L, mask=head_mask)


@triton.jit
def kernel_paged_attention_2d_reduce(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_SLOTS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # must match kernel_paged_attention_2d_split's slot_idx mapping exactly:
    # slot = kv_head_idx * num_queries_per_kv_padded + local_lane, where
    # kv_head_idx = head_idx // num_queries_per_kv, local_lane = head_idx %
    # num_queries_per_kv.
    kv_head_idx = head_idx // num_queries_per_kv
    local_lane = head_idx % num_queries_per_kv
    slot_idx = kv_head_idx * num_queries_per_kv_padded + local_lane

    base = seq_idx * HEAD_SLOTS * NUM_KV_SPLITS + slot_idx * NUM_KV_SPLITS
    splits = tl.arange(0, NUM_KV_SPLITS)
    m_vals = tl.load(partial_m_ptr + base + splits)
    l_vals = tl.load(partial_l_ptr + base + splits)

    m_final = tl.max(m_vals, axis=0)
    alpha = tl.exp(m_vals - m_final)
    alpha = tl.where(m_vals == float("-inf"), 0.0, alpha)
    l_final = tl.sum(l_vals * alpha, axis=0)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    part = tl.load(
        partial_out_ptr
        + (base + splits)[:, None] * HEAD_SIZE_PADDED
        + offs_d[None, :],
        mask=dim_mask[None, :],
        other=0.0,
    )
    acc = tl.sum(part * alpha[:, None], axis=0)

    acc = acc / (l_final + 1e-10)

    output_offset = seq_idx * output_stride_0 + head_idx * output_stride_1
    tl.store(
        output_ptr + output_offset + tl.arange(0, HEAD_SIZE_PADDED),
        acc,
        mask=dim_mask,
    )


# partial_out/m/l are never freed once allocated -- see the long comment at
# the call site for why: this hardware's caching allocator does not reliably
# defer reuse of a just-freed buffer until the GPU has actually finished an
# async kernel still reading it, so a function-local torch.empty() here is a
# real (confirmed, reproduced) use-after-free race. Keyed by the exact shape
# tuple so a change in NUM_KV_SPLITS/model shape gets a fresh buffer instead
# of reusing a wrongly-sized one. Also gives every call after the first a
# static buffer address, which CUDA-graph capture/replay needs anyway.
_PARTIAL_BUFFERS: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_partial_buffers(num_seqs, head_slots, num_kv_splits, head_size_padded, device):
    key = (num_seqs, head_slots, num_kv_splits, head_size_padded)
    bufs = _PARTIAL_BUFFERS.get(key)
    if bufs is None:
        n = num_seqs * head_slots * num_kv_splits
        bufs = (
            torch.empty(n, head_size_padded, dtype=torch.float32, device=device),
            torch.empty(n, dtype=torch.float32, device=device),
            torch.empty(n, dtype=torch.float32, device=device),
        )
        _PARTIAL_BUFFERS[key] = bufs
    return bufs


def gfx803_split_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_query_heads: int,
    num_queries_per_kv: int,
    block_size: int,
    num_kv_splits: int,
) -> None:
    """Decode-only (query_len==1 per seq), no ALIBI/sliding-window/sinks/fp8 --
    the subset actually exercised by this model. Caller must fall back to
    kernel_paged_attention_2d for anything outside that (prefill, chunked
    prefill, or a model/config using those features).
    """
    assert num_kv_splits & (num_kv_splits - 1) == 0, "NUM_KV_SPLITS must be a power of 2 (tl.arange constraint)"
    num_seqs = seq_lens.shape[0]
    num_kv_heads = key_cache.shape[1]
    head_size = key_cache.shape[2] * key_cache.shape[4]
    head_size_padded = triton.next_power_of_2(head_size)
    x = key_cache.shape[4]
    real_block_size = value_cache.shape[3]
    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    # query_head_idx = kv_head_idx * num_queries_per_kv + arange(0, padded) can
    # reach (num_kv_heads-1)*num_queries_per_kv + padded - 1 for the masked-off
    # (unwritten) lanes of the last kv head -- out_base's address arithmetic
    # still forms a pointer for those lanes even though the store is masked,
    # so the buffer must cover that range, not just num_query_heads.
    head_slots = num_kv_heads * num_queries_per_kv_padded
    partial_out, partial_m, partial_l = _get_partial_buffers(
        num_seqs, head_slots, num_kv_splits, head_size_padded, query.device
    )

    kernel_paged_attention_2d_split[(num_seqs, num_kv_heads, num_kv_splits)](
        partial_out_ptr=partial_out,
        partial_m_ptr=partial_m,
        partial_l_ptr=partial_l,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        scale=scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        BLOCK_SIZE=block_size,
        PHYSICAL_BLOCK_SIZE=real_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        x=x,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_SLOTS=head_slots,
        num_warps=4,
        num_stages=1,
    )

    # Without this, the reduce kernel intermittently reads corrupted
    # partial_out/m/l -- and, worse, the corruption reaches *outside* those
    # buffers (a canary tensor allocated purely to bracket the call and
    # never touched by either kernel gets clobbered too). Reproduces even
    # though both kernels are individually clean in isolation and nominally
    # run stream-ordered back-to-back -- a real launch-sequencing hazard on
    # this ROCm/gfx803 backend, not a logic bug in either kernel (verified:
    # a host `torch.cuda.synchronize()` between the two launches eliminates
    # it completely, but that op is illegal during CUDA graph capture --
    # `hipErrorStreamCaptureUnsupported`, confirmed directly on this stack.
    # A same-stream event record+wait eliminates it too and IS legal during
    # capture, confirmed directly -- use that instead.
    _split_reduce_barrier = torch.cuda.Event()
    _split_reduce_barrier.record()
    torch.cuda.current_stream().wait_event(_split_reduce_barrier)

    kernel_paged_attention_2d_reduce[(num_seqs, num_query_heads)](
        output_ptr=output,
        partial_out_ptr=partial_out,
        partial_m_ptr=partial_m,
        partial_l_ptr=partial_l,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_SLOTS=head_slots,
        num_warps=4,
        num_stages=1,
    )
