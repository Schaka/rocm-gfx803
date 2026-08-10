// LD_PRELOAD shim: intercepts rocblas_sgemm / rocblas_gemm_ex and routes the
// standard-algo fp32 path to gfx803_sgemm.h instead of rocBLAS/Tensile's own
// kernels, which are unreliable on gfx803 for every shape tested so far --
// see gfx803_sgemm.h's header comment for the full investigation.
//
// SCOPE: as of this patch, this intercepts ALL standard-algo f32
// rocblas_sgemm/rocblas_gemm_ex calls, not just specific proven-broken
// shapes -- there is no known-good subset to preserve Tensile's speed for
// yet. Everything else (other dtypes, batched calls, explicit
// solution_index requests, non-standard algo) falls through untouched to
// the real rocBLAS symbol via dlsym(RTLD_NEXT, ...), so this only affects
// the path it has actually replaced and verified.
//
// NOTE 1: tools/rocblas_sweep.cpp's overnight matrix scan (see KERNEL_BUGS.md)
// found that every explicitly-selected Tensile solution (via
// rocblas_gemm_algo_solution_index) came back clean for shapes with
// min(M,K,N) >= 256. This looked like grounds to narrow the shim's scope to
// only small shapes -- but direct testing disproved it: the *default*
// dispatch path (algo=standard, solution_index=0, i.e. what every real
// caller actually uses) still corrupts large shapes like 768x768x3072
// (up to 1.4 magnitude error) even though all 55 solutions test clean when
// selected explicitly. rocBLAS's automatic solution-selection heuristic for
// the default path evidently doesn't just pick one of the enumerated
// solutions -- do not narrow this shim's scope based on the explicit
// solution-index sweep data. Keep it blanket until the default-path
// dispatch itself is understood.
//
// NOTE 2: the actual root cause (see KERNEL_BUGS.md) is now understood:
// rocBLAS's internal device memory pool reuses memory across GEMM calls as
// GlobalSplitU (split-K) scratch space WITHOUT re-zeroing it, and gfx803's
// GSU-reduction kernels accumulate into that space via a software
// compare-and-swap loop (Polaris/GCN3 has no native float atomic-add) that
// assumes it starts zeroed. A "proper" fix -- giving this shim its own
// private rocblas_handle with a self-managed, self-zeroed workspace, using
// AMD's real auto-tuned Tensile kernel instead of this hand-written one --
// was built and verified correct, but BENCHMARKED SLOWER than this simple
// kernel in every case tested (4x slower for tiny shapes due to fixed
// per-call overhead, 20-46% slower for large shapes due to the
// zeroing cost itself, up to 2048^3). This card appears to be memory-
// bandwidth-bound for GEMM regardless of kernel sophistication, and/or
// gfx803's Tensile "tuning" was never actually validated on real gfx803
// hardware (unsupported since ROCm 6.0). Kept this simple kernel as the
// shipped fix for that reason -- see git history / KERNEL_BUGS.md for the
// benchmark data and the abandoned private-handle approach.
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include "gfx803_sgemm.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

typedef rocblas_status (*rocblas_sgemm_t)(rocblas_handle, rocblas_operation, rocblas_operation,
                                          rocblas_int, rocblas_int, rocblas_int,
                                          const float*, const float*, rocblas_int,
                                          const float*, rocblas_int,
                                          const float*, float*, rocblas_int);

typedef rocblas_status (*rocblas_gemm_ex_t)(rocblas_handle, rocblas_operation, rocblas_operation,
                                            rocblas_int, rocblas_int, rocblas_int,
                                            const void*, const void*, rocblas_datatype, rocblas_int,
                                            const void*, rocblas_datatype, rocblas_int,
                                            const void*, const void*, rocblas_datatype, rocblas_int,
                                            void*, rocblas_datatype, rocblas_int,
                                            rocblas_datatype, rocblas_gemm_algo, int32_t, uint32_t);

static rocblas_sgemm_t real_rocblas_sgemm = nullptr;
static rocblas_gemm_ex_t real_rocblas_gemm_ex = nullptr;

static void ensure_real_symbols() {
    if (!real_rocblas_sgemm) {
        real_rocblas_sgemm = (rocblas_sgemm_t)dlsym(RTLD_NEXT, "rocblas_sgemm");
    }
    if (!real_rocblas_gemm_ex) {
        real_rocblas_gemm_ex = (rocblas_gemm_ex_t)dlsym(RTLD_NEXT, "rocblas_gemm_ex");
    }
}

// Escape hatch for A/B testing against the real rocBLAS path without
// rebuilding anything.
static bool env_shim_disabled() {
    const char* v = getenv("GFX803_SGEMM_SHIM_DISABLE");
    return v && v[0] == '1';
}

extern "C" rocblas_status rocblas_sgemm(rocblas_handle handle,
                                        rocblas_operation transA, rocblas_operation transB,
                                        rocblas_int m, rocblas_int n, rocblas_int k,
                                        const float* alpha,
                                        const float* A, rocblas_int lda,
                                        const float* B, rocblas_int ldb,
                                        const float* beta,
                                        float* C, rocblas_int ldc) {
    ensure_real_symbols();
    if (env_shim_disabled()) {
        return real_rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    bool tA = (transA == rocblas_operation_transpose);
    bool tB = (transB == rocblas_operation_transpose);

    gfx803_sgemm((int)m, (int)n, (int)k, *alpha,
                A, (int)lda, tA, B, (int)ldb, tB, *beta, C, (int)ldc, stream);

    hipError_t herr = hipGetLastError();
    if (herr != hipSuccess) {
        fprintf(stderr, "gfx803_sgemm_shim: kernel launch failed (%s), falling back to rocBLAS\n",
                hipGetErrorString(herr));
        return real_rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                          rocblas_operation transA, rocblas_operation transB,
                                          rocblas_int m, rocblas_int n, rocblas_int k,
                                          const void* alpha,
                                          const void* a, rocblas_datatype a_type, rocblas_int lda,
                                          const void* b, rocblas_datatype b_type, rocblas_int ldb,
                                          const void* beta,
                                          const void* c, rocblas_datatype c_type, rocblas_int ldc,
                                          void* d, rocblas_datatype d_type, rocblas_int ldd,
                                          rocblas_datatype compute_type, rocblas_gemm_algo algo,
                                          int32_t solution_index, uint32_t flags) {
    ensure_real_symbols();

    bool all_f32 = (a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
                    c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
                    compute_type == rocblas_datatype_f32_r);
    // Only take over the plain default-algo f32 path verified above; explicit
    // solution_index requests, non-f32 types, and batched calls fall through
    // to real rocBLAS untouched.
    bool take_over = all_f32 && algo == rocblas_gemm_algo_standard && solution_index == 0
                     && !env_shim_disabled() && c == d; // in-place C==D matches this kernel's beta-accumulate semantics

    if (!take_over) {
        return real_rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha,
                                    a, a_type, lda, b, b_type, ldb, beta,
                                    c, c_type, ldc, d, d_type, ldd,
                                    compute_type, algo, solution_index, flags);
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    bool tA = (transA == rocblas_operation_transpose);
    bool tB = (transB == rocblas_operation_transpose);

    gfx803_sgemm((int)m, (int)n, (int)k, *(const float*)alpha,
                (const float*)a, (int)lda, tA, (const float*)b, (int)ldb, tB,
                *(const float*)beta, (float*)d, (int)ldd, stream);

    hipError_t herr = hipGetLastError();
    if (herr != hipSuccess) {
        fprintf(stderr, "gfx803_sgemm_shim: gemm_ex kernel launch failed (%s), falling back to rocBLAS\n",
                hipGetErrorString(herr));
        return real_rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha,
                                    a, a_type, lda, b, b_type, ldb, beta,
                                    c, c_type, ldc, d, d_type, ldd,
                                    compute_type, algo, solution_index, flags);
    }
    return rocblas_status_success;
}
