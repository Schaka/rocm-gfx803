// Tests whether rocBLAS's per-handle auto-managed device memory workspace
// (reused across calls, grown on demand, used as GSU/split-K atomic-
// reduction scratch space) is the actual root cause of the pervasive fp32
// GEMM corruption: first call in a process correct, every call after wrong.
// Hypothesis: the workspace isn't zeroed between calls, and a GSU>1
// solution's atomic-accumulation kernel assumes a zeroed workspace on entry
// -- so leftover partial sums from the previous call corrupt the next one.
//
// Modes (argv[3]):
//   baseline    - one handle reused for all reps (control; should reproduce
//                 corruption exactly like rocblas_verify_solution.cpp)
//   freshhandle - destroy+recreate the handle every rep (forces a fresh
//                 workspace allocation each time)
//   ownworkspace - one handle, but explicitly manage+zero our own workspace
//                 buffer via rocblas_set_workspace before every rep
//
// Usage: rocblas_workspace_test <solution_index> <repeats> <mode>
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <solution_index> <repeats> <baseline|freshhandle|ownworkspace> [M K N]\n", argv[0]); return 1; }
    int sol_idx = atoi(argv[1]);
    int repeats = atoi(argv[2]);
    const char* mode = argv[3];
    rocblas_gemm_algo algo = (sol_idx == 0) ? rocblas_gemm_algo_standard : rocblas_gemm_algo_solution_index;

    const int M = (argc > 6) ? atoi(argv[4]) : 40;
    const int K = (argc > 6) ? atoi(argv[5]) : 72;
    const int N = (argc > 6) ? atoi(argv[6]) : 800;

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

    rocblas_handle handle = nullptr;
    void* own_workspace = nullptr;
    size_t own_workspace_size = 64 * 1024 * 1024; // 64MB, generous for GSU scratch

    if (strcmp(mode, "baseline") == 0 || strcmp(mode, "ownworkspace") == 0 ||
        strcmp(mode, "ownworkspace_fixed") == 0 || strcmp(mode, "ownworkspace_fixed_nozero") == 0) {
        CHECK_RB(rocblas_create_handle(&handle));
    }
    if (strcmp(mode, "ownworkspace_fixed") == 0) {
        // bypass the size query entirely -- just a big fixed buffer, to
        // isolate whether "zeroing" matters or merely "having a
        // user-managed workspace set" changes rocBLAS's internal behavior
        own_workspace_size = 64 * 1024 * 1024;
        CHECK_HIP(hipMalloc(&own_workspace, own_workspace_size));
    }
    if (strcmp(mode, "ownworkspace_fixed_nozero") == 0) {
        // same fixed buffer, set ONCE, but never re-zeroed per call
        own_workspace_size = 64 * 1024 * 1024;
        CHECK_HIP(hipMalloc(&own_workspace, own_workspace_size));
        CHECK_HIP(hipMemset(own_workspace, 0, own_workspace_size));
        CHECK_RB(rocblas_set_workspace(handle, own_workspace, own_workspace_size));
    }
    if (strcmp(mode, "ownworkspace") == 0) {
        CHECK_RB(rocblas_start_device_memory_size_query(handle));
        rocblas_status qst = rocblas_gemm_ex(handle,
                                             rocblas_operation_none, rocblas_operation_none,
                                             M, N, K, &alpha,
                                             d_a, rocblas_datatype_f32_r, M,
                                             d_b, rocblas_datatype_f32_r, K,
                                             &beta,
                                             d_c, rocblas_datatype_f32_r, M,
                                             d_c, rocblas_datatype_f32_r, M,
                                             rocblas_datatype_f32_r,
                                             algo, sol_idx, 0);
        (void)qst;
        size_t queried_size = 0;
        CHECK_RB(rocblas_stop_device_memory_size_query(handle, &queried_size));
        fprintf(stderr, "queried real workspace size: %zu bytes\n", queried_size);
        own_workspace_size = queried_size > 0 ? queried_size : 4096;
        CHECK_HIP(hipMalloc(&own_workspace, own_workspace_size));
    }

    int bad_runs = 0;
    for (int rep = 0; rep < repeats; rep++) {
        for (size_t i = 0; i < a_elems; i++) h_a[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        for (size_t i = 0; i < b_elems; i++) h_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        CHECK_HIP(hipMemcpy(d_a, h_a, a_elems * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_b, h_b, b_elems * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(float)));

        if (strcmp(mode, "freshhandle") == 0) {
            CHECK_RB(rocblas_create_handle(&handle));
        }
        if (strcmp(mode, "ownworkspace") == 0 || strcmp(mode, "ownworkspace_fixed") == 0) {
            CHECK_HIP(hipMemset(own_workspace, 0, own_workspace_size));
            CHECK_RB(rocblas_set_workspace(handle, own_workspace, own_workspace_size));
        }
        // ownworkspace_fixed_nozero: workspace set ONCE before the loop, deliberately not touched here

        rocblas_status st = rocblas_gemm_ex(handle,
                                            rocblas_operation_none, rocblas_operation_none,
                                            M, N, K, &alpha,
                                            d_a, rocblas_datatype_f32_r, M,
                                            d_b, rocblas_datatype_f32_r, K,
                                            &beta,
                                            d_c, rocblas_datatype_f32_r, M,
                                            d_c, rocblas_datatype_f32_r, M,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            sol_idx, 0);
        if (st != rocblas_status_success) {
            fprintf(stderr, "SOLUTION %d: INAPPLICABLE (status=%d on rep %d)\n", sol_idx, (int)st, rep);
            return 2;
        }
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
        if (bad) {
            bad_runs++;
            fprintf(stderr, "  rep %d: MISMATCH\n", rep);
        } else {
            fprintf(stderr, "  rep %d: OK\n", rep);
        }

        if (strcmp(mode, "freshhandle") == 0) {
            rocblas_destroy_handle(handle);
            handle = nullptr;
        }
    }

    printf("MODE=%s SOLUTION=%d: %d/%d reps had mismatches\n", mode, sol_idx, bad_runs, repeats);

    if (handle) rocblas_destroy_handle(handle);
    if (own_workspace) hipFree(own_workspace);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
