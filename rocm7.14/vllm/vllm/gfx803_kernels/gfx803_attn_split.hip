// gfx803 split-KV decode-attention, hand-written HIP replacement for
// kernel_paged_attention_2d_split (gfx803_split_attn.py). That Triton
// kernel measured 256/256 VGPR + 637 spilled registers at this model's
// HEAD_SIZE_PADDED=256 (pulled straight from ~/.triton/cache's .amdgcn) --
// Triton materializes a full [num_queries_per_kv_padded, HEAD_SIZE_PADDED]
// K/V tile and accumulator per iteration with no way to chunk it (no list
// comprehensions, no tensor slicing, no loop-carried tensor collections in
// Triton's frontend -- all three tried and rejected before writing this).
//
// Design: one wavefront (64 lanes) per query row, one row per
// num_queries_per_kv (no padding to 16 the way Triton's tl.dot needed --
// this kernel has no such minimum-tile constraint, so it only launches as
// many wavefronts as there are real query heads per kv head). Each lane
// owns head_size/64 elements of the row; K*Q dot product per KV token is
// a wavefront-local __shfl_xor butterfly reduction, done one token at a
// time so no full-width K/V tile or accumulator is ever materialized --
// per-thread state is just a few floats (acc[ELEMS], M, L), independent
// of BLOCK_SIZE or NUM_KV_SPLITS. Different wavefronts (rows) never
// share state, so there is no shared memory and no __syncthreads anywhere
// in this kernel.
//
// Output buffer layout (partial_out/m/l) and the slot_idx addressing
// formula are kept byte-identical to the Triton kernel's so the existing,
// already-verified kernel_paged_attention_2d_reduce can be reused
// unchanged -- only the split kernel itself is replaced.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define WARP_SIZE 64
#define MAX_ELEMS_PER_THREAD 4  // this model: head_size_padded=256, 256/64=4

__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }

extern "C" __global__ void __launch_bounds__(256, 4) gfx803_attn_split_kernel(
    float* __restrict__ partial_out,        // [n, head_size_padded]
    float* __restrict__ partial_m,          // [n]
    float* __restrict__ partial_l,          // [n]
    const __half* __restrict__ query,       // [num_tokens, num_query_heads, head_size]
    const __half* __restrict__ key_cache,   // [num_blks, num_kv_heads, head_size/x, blk_size, x]
    const __half* __restrict__ value_cache, // [num_blks, num_kv_heads, head_size, blk_size]
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,       // [num_seqs]
    float scale,
    int num_query_heads,
    int num_queries_per_kv,
    int num_queries_per_kv_padded,
    int block_table_stride,
    long long query_stride_0,
    long long query_stride_1,
    int physical_block_size,
    int head_size,
    int head_size_padded,
    int x,
    long long stride_k0, long long stride_k1, long long stride_k2,
    long long stride_k3, long long stride_k4,
    long long stride_v0, long long stride_v1, long long stride_v2,
    long long stride_v3,
    int num_kv_splits,
    int head_slots
) {
    const int seq_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int split_id = blockIdx.z;
    const int row = threadIdx.y;   // 0 .. num_queries_per_kv-1 (real rows only)
    const int lane = threadIdx.x;  // 0 .. 63

    const int query_head_idx = kv_head_idx * num_queries_per_kv + row;
    const bool head_valid = query_head_idx < num_query_heads;
    const int slot_idx = kv_head_idx * num_queries_per_kv_padded + row;

    const int seq_len = seq_lens[seq_idx];
    const int tokens_per_split = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int tok_start = split_id * tokens_per_split;
    const int tok_end = min(tok_start + tokens_per_split, seq_len);

    const int elems = (head_size_padded + WARP_SIZE - 1) / WARP_SIZE;

    float acc[MAX_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; e++) acc[e] = 0.0f;

    float M = -INFINITY;
    float L = 0.0f;

    const long long query_base = (long long)seq_idx * query_stride_0
                                + (long long)query_head_idx * query_stride_1;
    const int block_table_base = seq_idx * block_table_stride;

    for (int tok = tok_start; tok < tok_end; tok++) {
        const int p_block_idx = block_tables[block_table_base + tok / physical_block_size];
        const int internal_offset = tok % physical_block_size;

        float partial = 0.0f;
        #pragma unroll
        for (int e = 0; e < MAX_ELEMS_PER_THREAD; e++) {
            if (e >= elems) break;
            const int d = lane + e * WARP_SIZE;
            if (d < head_size) {
                const float qv = head_valid ? h2f(query[query_base + d]) : 0.0f;
                const long long k_off = (long long)p_block_idx * stride_k0
                    + (long long)kv_head_idx * stride_k1
                    + (long long)(d / x) * stride_k2
                    + (long long)internal_offset * stride_k3
                    + (long long)(d % x) * stride_k4;
                partial += qv * h2f(key_cache[k_off]);
            }
        }
        #pragma unroll
        for (int offset = 32; offset >= 1; offset >>= 1) {
            partial += __shfl_xor(partial, offset);
        }
        const float score = head_valid ? partial * scale : -INFINITY;

        const float m_new = fmaxf(M, score);
        const float p = (m_new == -INFINITY) ? 0.0f : expf(score - m_new);
        const float alpha = (M == -INFINITY) ? 0.0f : expf(M - m_new);
        L = L * alpha + p;
        M = m_new;

        #pragma unroll
        for (int e = 0; e < MAX_ELEMS_PER_THREAD; e++) {
            if (e >= elems) break;
            const int d = lane + e * WARP_SIZE;
            if (d < head_size) {
                const long long v_off = (long long)p_block_idx * stride_v0
                    + (long long)kv_head_idx * stride_v1
                    + (long long)d * stride_v2
                    + (long long)internal_offset * stride_v3;
                acc[e] = acc[e] * alpha + p * h2f(value_cache[v_off]);
            }
        }
    }

    if (!head_valid) return;

    const long long out_base = (long long)seq_idx * head_slots * num_kv_splits
                              + (long long)slot_idx * num_kv_splits
                              + split_id;
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; e++) {
        if (e >= elems) break;
        const int d = lane + e * WARP_SIZE;
        if (d < head_size) {
            partial_out[out_base * head_size_padded + d] = acc[e];
        }
    }
    if (lane == 0) {
        partial_m[out_base] = M;
        partial_l[out_base] = L;
    }
}

extern "C" {

void gfx803_attn_split_launch(
    void* partial_out, void* partial_m, void* partial_l,
    const void* query, const void* key_cache, const void* value_cache,
    const void* block_tables, const void* seq_lens,
    float scale,
    int num_seqs, int num_kv_heads, int num_query_heads,
    int num_queries_per_kv, int num_queries_per_kv_padded,
    int block_table_stride,
    long long query_stride_0, long long query_stride_1,
    int physical_block_size,
    int head_size, int head_size_padded, int x,
    long long stride_k0, long long stride_k1, long long stride_k2,
    long long stride_k3, long long stride_k4,
    long long stride_v0, long long stride_v1, long long stride_v2,
    long long stride_v3,
    int num_kv_splits, int head_slots,
    void* stream
) {
    dim3 grid(num_seqs, num_kv_heads, num_kv_splits);
    dim3 block(WARP_SIZE, num_queries_per_kv);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);

    hipLaunchKernelGGL(
        gfx803_attn_split_kernel, grid, block, 0, s,
        reinterpret_cast<float*>(partial_out),
        reinterpret_cast<float*>(partial_m),
        reinterpret_cast<float*>(partial_l),
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const __half*>(key_cache),
        reinterpret_cast<const __half*>(value_cache),
        reinterpret_cast<const int*>(block_tables),
        reinterpret_cast<const int*>(seq_lens),
        scale,
        num_query_heads, num_queries_per_kv, num_queries_per_kv_padded,
        block_table_stride,
        query_stride_0, query_stride_1,
        physical_block_size,
        head_size, head_size_padded, x,
        stride_k0, stride_k1, stride_k2, stride_k3, stride_k4,
        stride_v0, stride_v1, stride_v2, stride_v3,
        num_kv_splits, head_slots
    );
}

} // extern "C"
