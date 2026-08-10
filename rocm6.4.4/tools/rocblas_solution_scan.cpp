// Enumerate every candidate rocBLAS/Tensile solution for the M=40,N=800,K=72
// FP32 NN GEMM (same shape as MIOpen's node_Conv_1215 1x1 conv-as-matmul),
// and run each one individually, several times, checking correctness. Same
// methodology that found the WGM8 bug: isolate which specific solutions are
// broken so the shared bad parameter can be identified from their names.
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#define CHECK_HIP(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e), __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

int main(void) {
    const int M = 40, K = 72, N = 800;
    const int REPEATS = 100; // per-solution repeat count to catch intermittent failures

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    const size_t a_elems = (size_t)M * K;
    const size_t b_elems = (size_t)K * N;
    const size_t c_elems = (size_t)M * N;

    float *h_a = (float*)malloc(a_elems * sizeof(float));
    float *h_b = (float*)malloc(b_elems * sizeof(float));
    float *h_c = (float*)malloc(c_elems * sizeof(float));

    void *d_a, *d_b, *d_c;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(float)));

    const float alpha = 1.0f, beta = 0.0f;

    // get solution count
    rocblas_int list_size = 0;
    rocblas_gemm_ex_get_solutions(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  M, N, K, &alpha,
                                  d_a, rocblas_datatype_f32_r, M,
                                  d_b, rocblas_datatype_f32_r, K,
                                  &beta,
                                  d_c, rocblas_datatype_f32_r, M,
                                  d_c, rocblas_datatype_f32_r, M,
                                  rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_solution_index,
                                  0 /*flags*/,
                                  nullptr, &list_size);
    fprintf(stderr, "Found %d candidate solutions\n", list_size);

    std::vector<rocblas_int> solutions(list_size);
    rocblas_gemm_ex_get_solutions(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  M, N, K, &alpha,
                                  d_a, rocblas_datatype_f32_r, M,
                                  d_b, rocblas_datatype_f32_r, K,
                                  &beta,
                                  d_c, rocblas_datatype_f32_r, M,
                                  d_c, rocblas_datatype_f32_r, M,
                                  rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_solution_index,
                                  0,
                                  solutions.data(), &list_size);

    int total_broken = 0;
    for (int si = 0; si < list_size; si++) {
        rocblas_int sol_idx = solutions[si];
        int bad_runs = 0;
        for (int rep = 0; rep < REPEATS; rep++) {
            for (size_t i = 0; i < a_elems; i++) h_a[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
            for (size_t i = 0; i < b_elems; i++) h_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
            CHECK_HIP(hipMemcpy(d_a, h_a, a_elems * sizeof(float), hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(d_b, h_b, b_elems * sizeof(float), hipMemcpyHostToDevice));
            CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(float)));

            rocblas_status st = rocblas_gemm_ex(handle,
                                                rocblas_operation_none, rocblas_operation_none,
                                                M, N, K, &alpha,
                                                d_a, rocblas_datatype_f32_r, M,
                                                d_b, rocblas_datatype_f32_r, K,
                                                &beta,
                                                d_c, rocblas_datatype_f32_r, M,
                                                d_c, rocblas_datatype_f32_r, M,
                                                rocblas_datatype_f32_r,
                                                rocblas_gemm_algo_solution_index,
                                                sol_idx,
                                                0);
            if (st != rocblas_status_success) {
                fprintf(stderr, "SOLUTION %d: INAPPLICABLE (status=%d on rep %d) -- excluded from clean/broken counts\n", sol_idx, (int)st, rep);
                bad_runs = -1;
                break;
            }
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipMemcpy(h_c, d_c, c_elems * sizeof(float), hipMemcpyDeviceToHost));

            int bad = 0;
            for (int n = 0; n < N && bad == 0; n++) {
                for (int m = 0; m < M; m++) {
                    double ref = 0.0;
                    for (int k = 0; k < K; k++) ref += (double)h_a[k * M + m] * (double)h_b[n * K + k];
                    double got = h_c[n * M + m];
                    if (fabs(got - ref) / (fabs(ref) + 1e-6) > 1e-2) { bad = 1; break; }
                }
            }
            if (bad) bad_runs++;
        }
        if (bad_runs > 0) {
            total_broken++;
            fprintf(stderr, "SOLUTION %d: BROKEN (%d/%d reps had mismatches)\n", sol_idx, bad_runs, REPEATS);
        } else if (bad_runs == 0) {
            printf("SOLUTION %d: clean (%d/%d reps OK)\n", sol_idx, REPEATS, REPEATS);
        }
        fflush(stdout);
    }

    fprintf(stderr, "TOTAL: %d/%d solutions broken\n", total_broken, list_size);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    rocblas_destroy_handle(handle);
    printf("SCAN COMPLETE\n");
    return 0;
}
