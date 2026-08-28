// Hand-tiled fp16-in/fp32-accumulate GEMM for gfx803, compiled directly via
// hipcc (no Tensile ASM generator, no Triton) -- classic shared-memory
// tiled + register-blocked design (BLOCK_M x BLOCK_N output tile per
// threadblock, THREAD_M x THREAD_N register sub-tile per thread),
// following the same style as vLLM's own csrc/rocm/skinny_gemms.cu
// (portable __hmul2/__hfma2 intrinsics, no packed-hardware assumptions
// beyond what those intrinsics already compile to correctly on this arch).
//
// C[M,N] = A[M,K] @ B[K,N], A/B/C row-major, fp16 storage, fp32 accumulate.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %d at %d\n", e, __LINE__); exit(1); } } while(0)
#define CHECK_RB(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { printf("rocBLAS error %d at %d\n", s, __LINE__); exit(1); } } while(0)

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16
#define THREAD_M 4
#define THREAD_N 4
// threadblock: (BLOCK_M/THREAD_M) x (BLOCK_N/THREAD_N) = 16 x 16 = 256 threads

// __launch_bounds__(256, 1): without this, the compiler targets an
// occupancy level that caps this kernel at 64 VGPRs/thread and spills 35
// registers to memory every iteration (confirmed via llvm-readobj on the
// compiled gfx803 code object -- .vgpr_count=64, .vgpr_spill_count=35).
// Telling it 1 block/CU is enough (trading occupancy for register budget)
// lets it use 110 VGPRs/thread with .vgpr_spill_count=0 -- this alone is
// the difference between losing to rocBLAS by 2-4x and beating it by
// 1.03-1.26x at every shape/M this model uses. See vllm-gfx803/NOTES.md
// section 29 for the full investigation.
__global__ __launch_bounds__(256, 1) void gemm_tiled_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[BLOCK_K][BLOCK_M];
    __shared__ half Bs[BLOCK_K][BLOCK_N];

    int tx = threadIdx.x % (BLOCK_N / THREAD_N); // 0..15
    int ty = threadIdx.x / (BLOCK_N / THREAD_N); // 0..15
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // Vectorized tile loads: 256 threads, 1024 half-elements per tile for
    // both A (BLOCK_M x BLOCK_K) and B (BLOCK_K x BLOCK_N) -- each thread
    // loads one contiguous float2 (4 fp16 values, 8 bytes) instead of 4
    // scattered scalar 16-bit loads. gfx803 has no packed-fp16 ALU
    // (v_pk_fma_f16/v_mad_mix_f32/v_fma_mix_f32 all False -- confirmed via
    // Tensile's AsmCaps.py this same session), so packed *compute*
    // (__hmul2/__hfma2) buys nothing here; the real win on this hardware is
    // fewer, wider *memory* instructions, the same reason
    // csrc/rocm/skinny_gemms.cu's LLGemm1_kernel uses float4 loads.
    int tid = threadIdx.x;

    // A tile: BLOCK_M rows x BLOCK_K cols, 4 threads per row, each loads 4
    // contiguous K-elements (a fixed row is contiguous in global memory).
    int a_row = tid / (BLOCK_K / 4);
    int a_chunk = tid % (BLOCK_K / 4);

    // B tile: BLOCK_K rows x BLOCK_N cols, (BLOCK_N/4) threads per row, each
    // loads 4 contiguous N-elements (a fixed row is contiguous in global
    // memory).
    int b_row = tid / (BLOCK_N / 4);
    int b_chunk = tid % (BLOCK_N / 4);

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        {
            int gr = block_row + a_row;
            int gc = k0 + a_chunk * 4;
            float2 v4;
            if (gr < M && gc + 3 < K) {
                v4 = *reinterpret_cast<const float2*>(&A[(size_t)gr * K + gc]);
            } else {
                half zeros[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int kk = gc + e;
                    if (gr < M && kk < K) zeros[e] = A[(size_t)gr * K + kk];
                }
                v4 = *reinterpret_cast<float2*>(zeros);
            }
            half* vh = reinterpret_cast<half*>(&v4);
            #pragma unroll
            for (int e = 0; e < 4; e++)
                As[a_chunk * 4 + e][a_row] = vh[e];
        }
        {
            int gr = k0 + b_row;
            int gc = block_col + b_chunk * 4;
            float2 v4;
            if (gr < K && gc + 3 < N) {
                v4 = *reinterpret_cast<const float2*>(&B[(size_t)gr * N + gc]);
            } else {
                half zeros[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int nn = gc + e;
                    if (gr < K && nn < N) zeros[e] = B[(size_t)gr * N + nn];
                }
                v4 = *reinterpret_cast<float2*>(zeros);
            }
            half* vh = reinterpret_cast<half*>(&v4);
            #pragma unroll
            for (int e = 0; e < 4; e++)
                Bs[b_row][b_chunk * 4 + e] = vh[e];
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++) {
            half a_reg[THREAD_M];
            half b_reg[THREAD_N];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                a_reg[i] = As[kk][ty * THREAD_M + i];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++)
                b_reg[j] = Bs[kk][tx * THREAD_N + j];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += __half2float(a_reg[i]) * __half2float(b_reg[j]);
        }
        __syncthreads();
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

void run_gemm(const half* dA, const half* dB, half* dC, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    gemm_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

double bench_hand(const half* dA, const half* dB, half* dC, int M, int N, int K, int iters) {
    for (int i = 0; i < 5; i++) run_gemm(dA, dB, dC, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run_gemm(dA, dB, dC, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_rocblas(rocblas_handle handle, const half* dA, const half* dB, half* dC,
                      int M, int N, int K, int iters) {
    float alpha = 1.0f, beta = 0.0f;
    // C[M,N] = A[M,K] @ B[K,N] row-major == C^T[N,M] = B^T[N,K] @ A^T[K,M] col-major
    auto call = [&]() {
        CHECK_RB(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
            N, M, K, &alpha,
            dB, rocblas_datatype_f16_r, N,
            dA, rocblas_datatype_f16_r, K,
            &beta,
            dC, rocblas_datatype_f16_r, N,
            dC, rocblas_datatype_f16_r, N,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    };
    for (int i = 0; i < 5; i++) call();
    CHECK_HIP(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) call();
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main() {
    rocblas_handle handle;
    CHECK_RB(rocblas_create_handle(&handle));

    struct Shape { const char* name; int K, N; };
    Shape shapes[] = {
        {"qkv_proj", 1536, 2048},
        {"o_proj", 1536, 1536},
        {"gate_up_proj", 1536, 17920},
        {"down_proj", 8960, 1536},
    };

    for (auto& sh : shapes) {
        for (int M : {128, 512}) {
            int K = sh.K, N = sh.N;
            std::vector<half> hA((size_t)M * K), hB((size_t)K * N);
            srand(0);
            for (auto& v : hA) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.04f);
            for (auto& v : hB) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.04f);

            half *dA, *dB, *dC, *dC_ref;
            CHECK_HIP(hipMalloc(&dA, (size_t)M * K * 2));
            CHECK_HIP(hipMalloc(&dB, (size_t)K * N * 2));
            CHECK_HIP(hipMalloc(&dC, (size_t)M * N * 2));
            CHECK_HIP(hipMalloc(&dC_ref, (size_t)M * N * 2));
            CHECK_HIP(hipMemcpy(dA, hA.data(), (size_t)M * K * 2, hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(dB, hB.data(), (size_t)K * N * 2, hipMemcpyHostToDevice));
            CHECK_HIP(hipMemset(dC, 0, (size_t)M * N * 2));
            CHECK_HIP(hipMemset(dC_ref, 0, (size_t)M * N * 2));

            run_gemm(dA, dB, dC, M, N, K);
            CHECK_HIP(hipDeviceSynchronize());

            float alpha = 1.0f, beta = 0.0f;
            CHECK_RB(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                N, M, K, &alpha,
                dB, rocblas_datatype_f16_r, N,
                dA, rocblas_datatype_f16_r, K,
                &beta,
                dC_ref, rocblas_datatype_f16_r, N,
                dC_ref, rocblas_datatype_f16_r, N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            CHECK_HIP(hipDeviceSynchronize());

            std::vector<half> hC(M * N), hC_ref(M * N);
            CHECK_HIP(hipMemcpy(hC.data(), dC, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            CHECK_HIP(hipMemcpy(hC_ref.data(), dC_ref, (size_t)M * N * 2, hipMemcpyDeviceToHost));

            double max_diff = 0.0;
            for (size_t i = 0; i < (size_t)M * N; i++) {
                double d = fabs((double)__half2float(hC[i]) - (double)__half2float(hC_ref[i]));
                if (d > max_diff) max_diff = d;
            }

            double t_hand = bench_hand(dA, dB, dC, M, N, K, 50);
            double t_rocblas = bench_rocblas(handle, dA, dB, dC_ref, M, N, K, 50);

            printf("%-14s M=%-4d K=%-6d N=%-6d  hand=%.1fus  rocblas=%.1fus  speedup=%.2fx  max_diff=%.4e\n",
                sh.name, M, K, N, t_hand, t_rocblas, t_rocblas / t_hand, max_diff);
            fflush(stdout);

            // NOTE: running multiple shapes/M-values back-to-back in one
            // process has a real, unresolved host-side bug (not a kernel
            // bug -- every shape's correctness was independently verified
            // against rocblas_gemm_ex when run in isolation) that
            // eventually crashes with a GPU memory access fault and no
            // further output. Workaround used throughout this
            // investigation: one shape+M combination per process. See
            // vllm-gfx803/NOTES.md before trusting a multi-shape
            // run of this file end-to-end.
            hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dC_ref);
        }
    }

    rocblas_destroy_handle(handle);
    return 0;
}
