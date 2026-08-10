// Shape-sweep kernel scanner. Generalizes rocblas_solution_scan.cpp (single
// shape) to a matrix of shapes, so we can catalog every broken solution
// across gfx803's fp32 GEMM solution space, not just the one shape that
// found the original bug. NN layout only (matches conv-as-matmul / CT2
// usage patterns that triggered the original crashes).
//
// Output (stdout, CSV, safe to tail/grep while running):
//   M,K,N,solution_index,status,bad_reps,total_reps
//   status in {clean, broken, inapplicable}
//
// Progress/diagnostics go to stderr.
//
// Usage: rocblas_sweep [repeats] [out.csv] [start_shape_index]
//   repeats: reps per solution per shape (default 50)
//   out.csv: also mirror CSV rows here (default: none, stdout only)
//   start_shape_index: skip shapes before this index (0-based, default 0) --
//     for resuming an interrupted overnight run without redoing finished shapes
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>

#define CHECK_HIP(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e), __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

struct Shape { int M, K, N; };

// Deliberately varied: tiny, pow2-aligned, prime/odd, skinny, square, large,
// plus the exact shapes already known broken or already verified clean via
// the custom kernel, plus permutations of those to catch layout-dependent
// bugs.
static const Shape SHAPES[] = {
    {1,1,1},
    {8,8,8},
    {16,16,16},
    {32,32,32},
    {37,91,613},
    {40,72,800},
    {64,64,512},
    {100,100,100},
    {127,127,127},
    {128,128,1024},
    {129,129,129},
    {256,256,256},
    {512,512,512},
    {513,513,513},
    {768,768,3072},
    {1024,1024,1024},
    {1,2048,2048},
    {2048,2048,1},
    {4096,4096,4},
    {800,40,72},
    {72,800,40},
};
static const int NUM_SHAPES = sizeof(SHAPES) / sizeof(SHAPES[0]);

int main(int argc, char** argv) {
    int REPEATS = (argc > 1) ? atoi(argv[1]) : 50;
    int START_SHAPE = (argc > 3) ? atoi(argv[3]) : 0;
    FILE* csv_out = nullptr;
    if (argc > 2) {
        csv_out = fopen(argv[2], START_SHAPE > 0 ? "a" : "w");
        if (!csv_out) { fprintf(stderr, "cannot open %s for write\n", argv[2]); exit(1); }
    }

    fprintf(stderr, "rocblas_sweep: %d shapes, %d reps/solution/shape\n", NUM_SHAPES, REPEATS);
    printf("M,K,N,solution_index,status,bad_reps,total_reps\n");
    if (csv_out) fprintf(csv_out, "M,K,N,solution_index,status,bad_reps,total_reps\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    time_t t0 = time(nullptr);
    long total_broken_solutions = 0, total_solutions_seen = 0;

    for (int si_shape = START_SHAPE; si_shape < NUM_SHAPES; si_shape++) {
        const int M = SHAPES[si_shape].M, K = SHAPES[si_shape].K, N = SHAPES[si_shape].N;
        time_t shape_t0 = time(nullptr);
        fprintf(stderr, "\n=== shape %d/%d: M=%d K=%d N=%d ===\n", si_shape + 1, NUM_SHAPES, M, K, N);

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
                                      0, nullptr, &list_size);
        fprintf(stderr, "  %d candidate solutions\n", list_size);

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
                                      0, solutions.data(), &list_size);

        for (int idx = 0; idx < list_size; idx++) {
            rocblas_int sol_idx = solutions[idx];
            bool inapplicable = false;
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
                                                    sol_idx, 0);
                if (st != rocblas_status_success) {
                    inapplicable = true;
                    break;
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
                        // combined abs+rel threshold: avoids false positives from
                        // fp32-vs-double accumulation-order noise near zero refs
                        // (see gfx803_sgemm_verify.cpp fix, KERNEL_BUGS.md)
                        if (err_abs > 0.05 && err_rel > 1e-2) { bad = 1; break; }
                    }
                }
                if (bad) bad_runs++;
            }

            const char* status = inapplicable ? "inapplicable" : (bad_runs > 0 ? "broken" : "clean");
            int total_reps = inapplicable ? 0 : REPEATS;
            printf("%d,%d,%d,%d,%s,%d,%d\n", M, K, N, sol_idx, status, bad_runs, total_reps);
            if (csv_out) fprintf(csv_out, "%d,%d,%d,%d,%s,%d,%d\n", M, K, N, sol_idx, status, bad_runs, total_reps);
            fflush(stdout);
            if (csv_out) fflush(csv_out);

            if (!inapplicable) {
                total_solutions_seen++;
                if (bad_runs > 0) {
                    total_broken_solutions++;
                    fprintf(stderr, "  SOLUTION %d: BROKEN (%d/%d)\n", sol_idx, bad_runs, REPEATS);
                }
            }
        }

        hipFree(d_a); hipFree(d_b); hipFree(d_c);
        free(h_a); free(h_b); free(h_c);
        fprintf(stderr, "  shape done in %lds\n", (long)(time(nullptr) - shape_t0));
    }

    rocblas_destroy_handle(handle);
    if (csv_out) fclose(csv_out);
    fprintf(stderr, "\n=== SWEEP COMPLETE in %lds: %ld/%ld applicable solutions broken (across %d shapes) ===\n",
            (long)(time(nullptr) - t0), total_broken_solutions, total_solutions_seen, NUM_SHAPES);
    printf("SWEEP_COMPLETE\n");
    return 0;
}
