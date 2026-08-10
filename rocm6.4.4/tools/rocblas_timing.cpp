// Pure GPU-side timing: no host round-trips, no CPU reference computation,
// just repeated rocblas_sgemm calls on fixed device buffers, timed via
// hipEvent after a warmup phase. Used to compare: hand-written naive kernel
// vs private-handle+workspace-zeroing vs stock (unsafe) rocBLAS.
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %s\n",hipGetErrorString(e));exit(1);} } while(0)
#define CHECK_RB(x) do { rocblas_status s=(x); if(s!=rocblas_status_success){fprintf(stderr,"RB err %d\n",(int)s);exit(1);} } while(0)

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 40;
    const int K = argc > 2 ? atoi(argv[2]) : 72;
    const int N = argc > 3 ? atoi(argv[3]) : 800;
    const int ITERS = argc > 4 ? atoi(argv[4]) : 200;
    const int WARMUP = 10;

    rocblas_handle handle;
    CHECK_RB(rocblas_create_handle(&handle));

    size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;
    void *d_a, *d_b, *d_c;
    CHECK_HIP(hipMalloc(&d_a, a_elems * 4));
    CHECK_HIP(hipMalloc(&d_b, b_elems * 4));
    CHECK_HIP(hipMalloc(&d_c, c_elems * 4));
    CHECK_HIP(hipMemset(d_a, 0, a_elems * 4));
    CHECK_HIP(hipMemset(d_b, 0, b_elems * 4));
    CHECK_HIP(hipMemset(d_c, 0, c_elems * 4));

    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < WARMUP; i++) {
        CHECK_RB(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                               M, N, K, &alpha, (const float*)d_a, M, (const float*)d_b, K,
                               &beta, (float*)d_c, M));
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < ITERS; i++) {
        CHECK_RB(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                               M, N, K, &alpha, (const float*)d_a, M, (const float*)d_b, K,
                               &beta, (float*)d_c, M));
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms = 0;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    printf("M=%d K=%d N=%d: %d iters in %.3f ms, %.4f ms/iter\n", M, K, N, ITERS, ms, ms / ITERS);

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    rocblas_destroy_handle(handle);
    return 0;
}
