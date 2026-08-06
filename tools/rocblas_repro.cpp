// Standalone rocBLAS SGEMM repro, no MIOpen at all. Shape mirrors the 1x1
// conv-as-matmul MIOpen would dispatch for node_Conv_1215 (72->40 channels,
// spatial 16*50=800): C[M=40,N=800] = A[M=40,K=72] * B[K=72,N=800].
// Determines if this is a second, distinct Tensile miscompute bug (separate
// from the already-fixed WGM8 one) for this specific GEMM shape/tile on
// gfx803.
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

#define CHECK_ROCBLAS(x) do { \
    rocblas_status s = (x); \
    if (s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d: %s\n", (int)s, __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 40;
    const int K = argc > 2 ? atoi(argv[2]) : 72;
    const int N = argc > 3 ? atoi(argv[3]) : 800;
    const int ITERS = argc > 4 ? atoi(argv[4]) : 500;
    fprintf(stderr, "Testing M=%d K=%d N=%d, %d iters\n", M, K, N, ITERS);

    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    const size_t a_elems = (size_t)M * K; // weight, row-major MxK conceptually
    const size_t b_elems = (size_t)K * N; // input, KxN
    const size_t c_elems = (size_t)M * N; // output, MxN

    float *h_a = (float*)malloc(a_elems * sizeof(float));
    float *h_b = (float*)malloc(b_elems * sizeof(float));
    float *h_c = (float*)malloc(c_elems * sizeof(float));

    void *d_a, *d_b, *d_c;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(float)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int total_bad = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        for (size_t i = 0; i < a_elems; i++) h_a[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        for (size_t i = 0; i < b_elems; i++) h_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        CHECK_HIP(hipMemcpy(d_a, h_a, a_elems * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_b, h_b, b_elems * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(float)));

        // C(MxN, col-major) = A(MxK) * B(KxN), no transpose -- row-major
        // MxK * KxN is equivalent to column-major (K x M)^T style; use the
        // straightforward rocblas_sgemm with op_none, op_none and leading
        // dims matching row-major-as-column-major-transposed convention:
        // Compute C^T = B^T * A^T trick avoided -- just directly test with
        // standard column-major semantics: treat h_a as KxM col-major (i.e.
        // A^T), h_b as NxK... simplest: use op_transpose on A to get MxK from
        // a K-major storage. We just need SOME MxKxN gemm exercising the
        // same tile/shape Tensile would pick; exact transpose convention
        // doesn't matter for a correctness self-check since reference uses
        // the same indexing as what's passed to rocblas.
        CHECK_ROCBLAS(rocblas_sgemm(handle,
                                    rocblas_operation_none, rocblas_operation_none,
                                    M, N, K,
                                    &alpha,
                                    (const float*)d_a, M,
                                    (const float*)d_b, K,
                                    &beta,
                                    (float*)d_c, M));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_c, d_c, c_elems * sizeof(float), hipMemcpyDeviceToHost));

        // reference: column-major C[M,N] = A[M,K] * B[K,N], A col-major MxK, B col-major KxN
        int bad = 0;
        double max_err = 0;
        int max_err_idx = -1;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double ref = 0.0;
                for (int k = 0; k < K; k++) {
                    ref += (double)h_a[k * M + m] * (double)h_b[n * K + k];
                }
                int idx = n * M + m;
                double got = h_c[idx];
                double err = fabs(got - ref) / (fabs(ref) + 1e-6);
                if (err > 1e-2) {
                    bad++;
                    if (err > max_err) { max_err = err; max_err_idx = idx; }
                }
            }
        }
        if (bad) {
            total_bad++;
            fprintf(stderr, "iter %d: %d/%zu MISMATCH worst idx=%d max_err=%f\n", iter, bad, c_elems, max_err_idx, max_err);
        } else {
            printf("iter %d: OK c[0]=%f\n", iter, h_c[0]);
        }
        fflush(stdout);
    }

    fprintf(stderr, "TOTAL: %d/500 iterations had corruption\n", total_bad);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    rocblas_destroy_handle(handle);
    printf("ALL ITERATIONS COMPLETED\n");
    return 0;
}
