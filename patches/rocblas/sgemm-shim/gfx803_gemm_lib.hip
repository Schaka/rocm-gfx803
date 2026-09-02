// Shared-library entry point for gfx803_handwritten_gemm_v2_doublebuffer's
// kernel -- same kernel, no benchmark/main() harness, exported as a plain
// C function for ctypes. See SESSION_HANDOFF.md sections 27-31 and
// patches/rocblas/gfx803_handwritten_gemm_v2_doublebuffer.hip for the full
// investigation and derivation of every design choice below.
//
// This file is shared with the vLLM fork (vllm/vllm/gfx803_kernels/
// gfx803_gemm_lib.hip) and embedded in the rocBLAS fp16 takeover in
// sgemm_shim.cpp; keep the two copies identical.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16
#define THREAD_M 4
#define THREAD_N 4

__device__ __forceinline__ float2 load_a_tile(const half* __restrict__ A, int M, int K, int block_row, int k0, int a_row, int a_chunk) {
    int gr = block_row + a_row;
    int gc = k0 + a_chunk * 4;
    if (gr < M && gc + 3 < K) {
        return *reinterpret_cast<const float2*>(&A[(size_t)gr * K + gc]);
    }
    half zeros[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        int kk = gc + e;
        if (gr < M && kk < K) zeros[e] = A[(size_t)gr * K + kk];
    }
    return *reinterpret_cast<float2*>(zeros);
}

__device__ __forceinline__ float2 load_b_tile(const half* __restrict__ B, int K, int N, int block_col, int k0, int b_row, int b_chunk) {
    int gr = k0 + b_row;
    int gc = block_col + b_chunk * 4;
    if (gr < K && gc + 3 < N) {
        return *reinterpret_cast<const float2*>(&B[(size_t)gr * N + gc]);
    }
    half zeros[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        int nn = gc + e;
        if (gr < K && nn < N) zeros[e] = B[(size_t)gr * N + nn];
    }
    return *reinterpret_cast<float2*>(zeros);
}

__global__ __launch_bounds__(256, 1) void gfx803_gemm_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[2][BLOCK_K][BLOCK_M];
    __shared__ half Bs[2][BLOCK_K][BLOCK_N];

    int tx = threadIdx.x % (BLOCK_N / THREAD_N);
    int ty = threadIdx.x / (BLOCK_N / THREAD_N);
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    int tid = threadIdx.x;
    int a_row = tid / (BLOCK_K / 4);
    int a_chunk = tid % (BLOCK_K / 4);
    int b_row = tid / (BLOCK_N / 4);
    int b_chunk = tid % (BLOCK_N / 4);

    {
        float2 av = load_a_tile(A, M, K, block_row, 0, a_row, a_chunk);
        half* avh = reinterpret_cast<half*>(&av);
        #pragma unroll
        for (int e = 0; e < 4; e++) As[0][a_chunk * 4 + e][a_row] = avh[e];

        float2 bv = load_b_tile(B, K, N, block_col, 0, b_row, b_chunk);
        half* bvh = reinterpret_cast<half*>(&bv);
        #pragma unroll
        for (int e = 0; e < 4; e++) Bs[0][b_row][b_chunk * 4 + e] = bvh[e];
    }
    __syncthreads();

    int cur = 0;
    int num_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int t = 0; t < num_tiles; t++) {
        int next = 1 - cur;
        bool has_next = (t + 1 < num_tiles);
        float2 av_next, bv_next;
        if (has_next) {
            int k0_next = (t + 1) * BLOCK_K;
            av_next = load_a_tile(A, M, K, block_row, k0_next, a_row, a_chunk);
            bv_next = load_b_tile(B, K, N, block_col, k0_next, b_row, b_chunk);
        }

        // Full unroll of this loop (the default for a bare '#pragma unroll')
        // inflates live-range count enough that the compiler spills past the
        // launch_bounds(256,1) budget's comfort zone: 114 VGPR/thread, capping
        // real occupancy at 2 waves/SIMD (8 wavefronts/CU out of a 40 max --
        // this GEMM has a VGPR floor from the K-tile+register-blocked design,
        // not an LDS one: 8KB/block used out of 64KB LDS/CU). Capping the
        // unroll factor at 8 measured lowest on real hardware across
        // unroll={1,2,4,8,16(full)} -- 52-60 VGPR/thread depending on factor,
        // consistently ~4 waves/SIMD, and unroll=8 specifically beat unroll=4
        // (same 57 VGPR) on wall-clock despite identical occupancy, so this
        // isn't purely register-count-driven -- some benefit from the
        // shorter, more schedulable instruction sequence too. Head-to-head at
        // M in {128..2101}, K=2048, N=12288 (this model's gate_up_proj
        // shape): 2.79 -> 3.07-3.14 TFLOPS, ~10-13% over the unroll=4 result
        // and ~30% over the original full-unroll kernel, correctness
        // unchanged (max abs err 0.0000-0.0001 either way).
        #pragma unroll 8
        for (int kk = 0; kk < BLOCK_K; kk++) {
            half a_reg[THREAD_M];
            half b_reg[THREAD_N];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                a_reg[i] = As[cur][kk][ty * THREAD_M + i];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++)
                b_reg[j] = Bs[cur][kk][tx * THREAD_N + j];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                float af = __half2float(a_reg[i]);
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += af * __half2float(b_reg[j]);
            }
        }

        if (has_next) {
            half* avh = reinterpret_cast<half*>(&av_next);
            #pragma unroll
            for (int e = 0; e < 4; e++) As[next][a_chunk * 4 + e][a_row] = avh[e];
            half* bvh = reinterpret_cast<half*>(&bv_next);
            #pragma unroll
            for (int e = 0; e < 4; e++) Bs[next][b_row][b_chunk * 4 + e] = bvh[e];
        }
        __syncthreads();
        cur = next;
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int gr = block_row + ty * THREAD_M + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int gc = block_col + tx * THREAD_N + j;
            if (gc >= N) continue;
            C[(size_t)gr * N + gc] = __float2half(acc[i][j]);
        }
    }
}

extern "C" {

// C[M,N] = A[M,K] @ B[K,N], all row-major fp16. B must already be in
// [K,N] layout (i.e. the *transpose* of vLLM's standard [N,K] weight
// storage) -- transposing per call would defeat the purpose; callers are
// expected to cache a transposed copy of each weight once, the same
// pattern already established for rocblas_hssgemv_strided_batched in
// gfx803_gemv.py.
void gfx803_gemm_launch(const void* A, const void* B, void* C, int M, int N, int K, void* stream) {
    dim3 block(256);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    gfx803_gemm_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const half*>(A),
        reinterpret_cast<const half*>(B),
        reinterpret_cast<half*>(C),
        M, N, K);
}

} // extern "C"
