// v3: v1's proven tile size (4x4 register blocking, no spilling) + real
// double buffering -- prefetch next K-tile's global data into registers
// while computing on the current shared-memory tile, hiding global memory
// latency behind compute instead of stalling on __syncthreads() every
// iteration.
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

// __launch_bounds__(256, 1): without this, the compiler caps this kernel
// at a lower VGPR budget and forces spilling (confirmed via llvm-readobj
// on the compiled gfx803 code object -- see vllm-gfx803/NOTES.md
// for the full investigation, including how to extract and inspect the
// actual compiled ISA). 1 block/CU is enough register budget for zero
// spilling here (.vgpr_count=114, .vgpr_spill_count=0) -- combined with
// double buffering, this beats rocblas_gemm_ex by 18-43% at every real
// shape/M this model uses (see the table in section 30). Without this
// line, double buffering measured as a clear REGRESSION vs the
// non-double-buffered kernel (section 29) -- that earlier result was
// confounded by uncontrolled register-spilling differences between the
// two kernels, not a real property of double buffering on this hardware.
__global__ __launch_bounds__(256, 1) void gemm_tiled_kernel_v3(
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

    // prologue: load tile 0 into buffer 0
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

        #pragma unroll
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

void run_gemm(const half* dA, const half* dB, half* dC, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    gemm_tiled_kernel_v3<<<grid, block>>>(dA, dB, dC, M, N, K);
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

int main(int argc, char** argv) {
    rocblas_handle handle;
    CHECK_RB(rocblas_create_handle(&handle));

    struct Shape { const char* name; int K, N; };
    Shape shapes[] = {
        {"SHAPE_NAME", SHAPE_K, SHAPE_N},
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

            float alpha2 = 1.0f, beta2 = 0.0f;
            auto call = [&]() {
                rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                    N, M, K, &alpha2,
                    dB, rocblas_datatype_f16_r, N,
                    dA, rocblas_datatype_f16_r, K,
                    &beta2,
                    dC_ref, rocblas_datatype_f16_r, N,
                    dC_ref, rocblas_datatype_f16_r, N,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            };
            for (int i = 0; i < 5; i++) call();
            CHECK_HIP(hipDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 50; i++) call();
            CHECK_HIP(hipDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            double t_rocblas = std::chrono::duration<double, std::micro>(t1 - t0).count() / 50;

            printf("%-14s M=%-4d K=%-6d N=%-6d  hand=%.1fus  rocblas=%.1fus  speedup=%.2fx  max_diff=%.4e\n",
                sh.name, M, K, N, t_hand, t_rocblas, t_rocblas / t_hand, max_diff);
            fflush(stdout);

            // NOTE: running multiple shapes and/or multiple M values
            // back-to-back in one process has a real, unresolved
            // host-side bug (not a kernel bug -- every configuration's
            // correctness was independently verified against
            // rocblas_gemm_ex when run in isolation) that eventually
            // crashes with a GPU memory access fault and no further
            // output. Workaround used throughout this investigation: one
            // shape+M combination per process. See vllm-gfx803/NOTES.md
            // sections 29-30 before trusting a multi-config run of this
            // file end-to-end.
            hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dC_ref);
        }
    }

    rocblas_destroy_handle(handle);
    return 0;
}
