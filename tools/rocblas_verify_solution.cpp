// Run ONE specific rocBLAS/Tensile solution_index many times (for
// statistical confidence beyond the 20-rep scan) against the M=40,N=800,K=72
// FP32 NN shape. Usage: rocblas_verify_solution <solution_index> <repeats>
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_HIP(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e), __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <solution_index> <repeats>\n", argv[0]); return 1; }
    int sol_idx = atoi(argv[1]);
    int repeats = atoi(argv[2]);

    const int M = 40, K = 72, N = 800;
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
    int bad_runs = 0;
    for (int rep = 0; rep < repeats; rep++) {
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
            fprintf(stderr, "SOLUTION %d: INAPPLICABLE (status=%d on rep %d, not a valid candidate for this shape)\n", sol_idx, (int)st, rep);
            hipFree(d_a); hipFree(d_b); hipFree(d_c);
            free(h_a); free(h_b); free(h_c);
            rocblas_destroy_handle(handle);
            return 2;
        }
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_c, d_c, c_elems * sizeof(float), hipMemcpyDeviceToHost));

        int bad = 0;
        int bad_count = 0;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double ref = 0.0;
                for (int k = 0; k < K; k++) ref += (double)h_a[k * M + m] * (double)h_b[n * K + k];
                double got = h_c[n * M + m];
                if (fabs(got - ref) / (fabs(ref) + 1e-6) > 1e-2) { bad = 1; bad_count++; }
            }
        }
        if (bad) { bad_runs++; fprintf(stderr, "rep %d: MISMATCH %d/%d elems\n", rep, bad_count, M*N); }
    }

    printf("SOLUTION %d: %d/%d reps had mismatches\n", sol_idx, bad_runs, repeats);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    rocblas_destroy_handle(handle);
    return 0;
}
