// Verifies the design intended for the real shim: route the GEMM through a
// SEPARATE, private rocblas_handle (never the caller's handle) with a
// self-managed, self-zeroed workspace sized via min(K, MAX_GSU), using
// rocBLAS's own default auto-selected solution (algo=standard,
// solution_index=0) -- i.e. AMD's real, fully-tuned Tensile kernel, just
// with the buggy pool-reuse-without-zeroing sidestepped.
//
// Usage: rocblas_privatehandle_test <repeats> <M> <K> <N>
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

#define CHECK_RB(x) do { \
    rocblas_status s = (x); \
    if (s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d: %s\n", (int)s, __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

#define MAX_GSU 32

int main(int argc, char** argv) {
    int repeats = (argc > 1) ? atoi(argv[1]) : 50;
    int M = (argc > 2) ? atoi(argv[2]) : 40;
    int K = (argc > 3) ? atoi(argv[3]) : 72;
    int N = (argc > 4) ? atoi(argv[4]) : 800;

    // "caller" handle -- represents the application's own handle, NEVER
    // touched by our workspace management
    rocblas_handle caller_handle;
    CHECK_RB(rocblas_create_handle(&caller_handle));

    // "shim" handle -- private, owned entirely by us
    rocblas_handle shim_handle;
    CHECK_RB(rocblas_create_handle(&shim_handle));

    size_t workspace_bytes = (size_t)M * N * sizeof(float) * (K < MAX_GSU ? K : MAX_GSU);
    workspace_bytes = ((workspace_bytes + 255) / 256) * 256; // round to 256B granularity
    if (workspace_bytes < 4096) workspace_bytes = 4096;
    fprintf(stderr, "workspace_bytes = %zu (%.2f MB)\n", workspace_bytes, workspace_bytes / (1024.0*1024.0));

    void* shim_workspace;
    CHECK_HIP(hipMalloc(&shim_workspace, workspace_bytes));
    CHECK_RB(rocblas_set_workspace(shim_handle, shim_workspace, workspace_bytes));

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

        // simulate: caller's handle would normally be used for the app's
        // own default-dispatch GEMM; we instead dispatch through our
        // private, workspace-managed handle
        CHECK_HIP(hipMemset(shim_workspace, 0, workspace_bytes));

        rocblas_status st = rocblas_gemm_ex(shim_handle,
                                            rocblas_operation_none, rocblas_operation_none,
                                            M, N, K, &alpha,
                                            d_a, rocblas_datatype_f32_r, M,
                                            d_b, rocblas_datatype_f32_r, K,
                                            &beta,
                                            d_c, rocblas_datatype_f32_r, M,
                                            d_c, rocblas_datatype_f32_r, M,
                                            rocblas_datatype_f32_r,
                                            rocblas_gemm_algo_standard, 0, 0);
        CHECK_RB(st);
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_c, d_c, c_elems * sizeof(float), hipMemcpyDeviceToHost));

        int bad = 0;
        for (int n = 0; n < N && !bad; n++) {
            for (int m = 0; m < M; m++) {
                double ref = 0.0;
                for (int k = 0; k < K; k++) ref += (double)h_a[(size_t)k * M + m] * (double)h_b[(size_t)n * K + k];
                double got = h_c[(size_t)n * M + m];
                double err_abs = fabs(got - ref);
                double err_rel = err_abs / (fabs(ref) + 1e-6);
                if (err_abs > 0.05 && err_rel > 1e-2) { bad = 1; break; }
            }
        }
        if (bad) bad_runs++;
    }

    printf("PRIVATE-HANDLE M=%d K=%d N=%d: %d/%d reps had mismatches\n", M, K, N, bad_runs, repeats);

    hipFree(shim_workspace);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    rocblas_destroy_handle(shim_handle);
    rocblas_destroy_handle(caller_handle);
    return 0;
}
