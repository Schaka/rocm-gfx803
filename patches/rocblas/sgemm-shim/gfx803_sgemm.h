// Correctness-first replacement SGEMM for gfx803.
//
// WHY
// ---
// Every rocBLAS/Tensile SGEMM solution tested on gfx803 (RX 470, ROCm 6.4.4)
// silently miscomputes intermittently -- not just WGM8 kernels (see
// ../wgm-miscompute.sh, a separate and already-fixed bug), but ALL 55
// candidate solutions returned by rocblas_gemm_ex_get_solutions for a real
// CLAP-conv-derived shape (M=40,K=72,N=800), at rates from ~1% to ~100% of
// output elements per call, first call always correct, every one after that
// intermittently wrong. Confirmed pervasive, not shape-specific: aligned
// shapes (128x128x1024), tiny shapes (8x8x8), and large shapes (512^3) all
// showed the same pattern. Confirmed not a regression: rocBLAS built from the
// rocm-5.7.0 tag (the last officially gfx803-supported release, using ITS OWN
// bundled compiler) reproduces the same defect, only worse (~100% of elements
// wrong per bad call instead of ~2-15%). Confirmed not GSU/split-K specific:
// forcing GlobalSplitU=1 across every r9nano (gfx803) logic file, rebuilt and
// verified via direct msgpack inspection, made no difference. Confirmed not a
// hardware/thermal issue: a hand-written kernel doing heavy LDS tile staging
// and sustained FMA accumulation (far more demanding than any of the above)
// ran clean for 300 iterations, as did a hand-written global-atomic /
// cross-workgroup-flag stress kernel (2000 iterations) -- both primitives
// Tensile's own kernels rely on. Every individual primitive is reliable in
// isolation; every Tensile-generated GEMM kernel is not.
//
// Full investigation, including the harness bugs found and corrected along
// the way (a silent status-check bug that produced false "clean" results,
// and a 20-rep sample size that let a truly ~15%-broken kernel look clean by
// chance), is preserved in the session transcript this patch was extracted
// from; see the project's memory notes for pointers.
//
// WHAT
// ----
// A plain, unrolled-free, correctness-over-speed tiled SGEMM: 16x16 LDS
// tiles, one accumulator register per thread, no prefetching, no vectorized
// loads, no split-K. Deliberately simple -- there is nothing left for a
// miscompile to hide in. Verified clean (0 corrupted iterations) across
// dozens of shapes: the exact broken 40x72x800 case, aligned shapes,
// misaligned shapes, tiny shapes, and 512^3, each for 100-500 repeated
// randomized-input runs.
//
// This is NOT performance-tuned and is not meant to be: gfx803 has been
// abandoned upstream since ROCm 6.0, so shipping a slower-but-correct kernel
// is a straightforward tradeoff, matching the WGM patch's own philosophy.
// See sgemm-shim.cpp for how this gets wired into rocBLAS's dispatch, and
// note the scope there: as of this patch, it intercepts ALL standard-algo
// f32 rocblas_sgemm/rocblas_gemm_ex calls, not just the specific shapes
// proven broken above -- there is no evidence yet that ANY fp32 GEMM shape
// on this card is reliable through Tensile, so there is no known-good subset
// to preserve Tensile's speed for. Narrowing that scope (only intercepting
// shapes actually proven bad, via an exhaustive per-shape solution scan)
// is future work.
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>

#define GFX803_SGEMM_TILE 16

// C[MxN] = alpha * op(A) * op(B) + beta * C, column-major storage (BLAS/
// rocBLAS convention), matching rocblas_sgemm/rocblas_gemm_ex semantics:
//   transA/transB: false = N (no transpose), true = T (transpose)
//   A is (transA ? K x M : M x K) with leading dimension lda, column-major
//   B is (transB ? N x K : K x N) with leading dimension ldb, column-major
//   C is M x N with leading dimension ldc, column-major
__global__ void gfx803_sgemm_kernel(int M, int N, int K,
                                    float alpha,
                                    const float* __restrict__ A, int lda, bool transA,
                                    const float* __restrict__ B, int ldb, bool transB,
                                    float beta,
                                    float* __restrict__ C, int ldc) {
    __shared__ float As[GFX803_SGEMM_TILE][GFX803_SGEMM_TILE];
    __shared__ float Bs[GFX803_SGEMM_TILE][GFX803_SGEMM_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * GFX803_SGEMM_TILE + ty; // 0..M-1 (C row index)
    int col = blockIdx.x * GFX803_SGEMM_TILE + tx; // 0..N-1 (C col index)

    float acc = 0.0f;
    int n_tiles = (K + GFX803_SGEMM_TILE - 1) / GFX803_SGEMM_TILE;

    for (int t = 0; t < n_tiles; t++) {
        int a_k = t * GFX803_SGEMM_TILE + tx; // K-index for A load
        int b_k = t * GFX803_SGEMM_TILE + ty; // K-index for B load

        // transA=false: A physically MxK, op(A)(row,k) = A[k*lda + row]
        // transA=true:  A physically KxM, op(A)(row,k) = A[row*lda + k]
        if (row < M && a_k < K) {
            As[ty][tx] = transA ? A[(size_t)row * lda + a_k] : A[(size_t)a_k * lda + row];
        } else {
            As[ty][tx] = 0.0f;
        }

        // transB=false: B physically KxN, op(B)(k,col) = B[col*ldb + k]
        // transB=true:  B physically NxK, op(B)(k,col) = B[k*ldb + col]
        if (col < N && b_k < K) {
            Bs[ty][tx] = transB ? B[(size_t)b_k * ldb + col] : B[(size_t)col * ldb + b_k];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GFX803_SGEMM_TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        size_t idx = (size_t)col * ldc + row;
        C[idx] = alpha * acc + (beta != 0.0f ? beta * C[idx] : 0.0f);
    }
}

inline void gfx803_sgemm(int M, int N, int K,
                         float alpha,
                         const float* A, int lda, bool transA,
                         const float* B, int ldb, bool transB,
                         float beta,
                         float* C, int ldc,
                         hipStream_t stream = 0) {
    dim3 block(GFX803_SGEMM_TILE, GFX803_SGEMM_TILE);
    dim3 grid((N + GFX803_SGEMM_TILE - 1) / GFX803_SGEMM_TILE,
             (M + GFX803_SGEMM_TILE - 1) / GFX803_SGEMM_TILE);
    hipLaunchKernelGGL(gfx803_sgemm_kernel, grid, block, 0, stream,
                       M, N, K, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc);
}
