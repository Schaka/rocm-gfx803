// REFERENCE ONLY -- NOT the shipped shim (see sgemm_shim.cpp for that).
//
// This is the "proper fix" approach kept alongside the shipped
// hand-written-kernel shim for future reference: it routes intercepted GEMM
// calls through AMD's own real, auto-tuned Tensile kernel instead of a
// hand-written replacement, by giving this shim its own PRIVATE
// rocblas_handle (never the caller's) with a self-managed workspace that is
// explicitly zeroed before every call -- correctly sidestepping the actual
// root cause documented in KERNEL_BUGS.md ("The actual root cause, and a
// real (failed) attempt to fix it at the source"): rocBLAS's internal
// device memory pool reuses GlobalSplitU (split-K) scratch space across
// GEMM calls without re-zeroing it, and gfx803's GSU-reduction kernels
// (software compare-and-swap loops, since gfx803/Polaris has no native
// float atomic-add) assume that workspace starts zeroed.
//
// VERIFIED CORRECT on real hardware: 0/N mismatches across every shape that
// was previously broken through the default dispatch path, including the
// worst case found by the overnight sweep (129x129x129, 100% broken
// through explicit per-solution dispatch).
//
// NOT SHIPPED because it benchmarked SLOWER than the simple hand-written
// kernel in every case tested on real hardware:
//   - 40x72x800 (small):     0.0700 ms/iter here vs 0.0166 ms/iter hand-kernel (4.2x slower)
//   - 768x768x3072 (large):  4.7973 ms/iter here vs 3.2863 ms/iter hand-kernel (46% slower)
//   - 2048^3:                16.478 ms/iter here vs 13.776 ms/iter hand-kernel (20% slower)
// In every case, the hand-kernel also matched (not just beat) STOCK
// (unpatched, unsafe) rocBLAS's own timing almost exactly -- this GPU
// appears to be memory-bandwidth-bound for GEMM at these sizes regardless
// of kernel sophistication, and/or gfx803's Tensile "tuning" was never
// actually validated on real gfx803 hardware (unsupported since ROCm 6.0).
// The two costs that make this approach slower: a whole separate
// hipMemsetAsync launch's fixed overhead (dominates for small shapes), and
// the workspace-zeroing bandwidth cost itself, sized
// M*N*sizeof(float)*min(K, MAX_GSU) since Tensile's public
// device-memory-size-query API is unreliable for non-HPA f32 (see
// KERNEL_BUGS.md) -- there is no trustworthy way to query the real,
// tighter, per-solution requirement at runtime from outside rocBLAS.
//
// Kept here, compiling and correctness-verified, in case a future session
// wants to revisit narrowing this shim's scope (e.g. only for GSU-heavy
// large-K shapes where Tensile's real optimizations might eventually win),
// or wants a working example of "attach your own zeroed workspace to a
// private rocblas_handle" for some other purpose.
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
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

// Max GlobalSplitU observed across every gfx803 f32 (Type_SS) solution in
// rocBLAS 6.4.4's installed Tensile library (219 solutions scanned via
// msgpack, 156 with GSU>1, max 32). GSU can't usefully exceed K itself
// (it splits the K/reduction dimension), so the real per-call bound is
// min(K, MAX_GSU) -- this keeps the workspace for small-K/large-MN shapes
// (e.g. 4096x4096x4) from being wastefully oversized.
#define MAX_GSU 32

// Per-thread private handle + workspace: never touches the caller's own
// handle, so calls we don't intercept on that handle are completely
// unaffected. Deliberately never destroyed (leaked for process lifetime) --
// simpler than tracking thread exit, and bounded by the number of threads
// that ever call into this shim.
struct ShimState {
    rocblas_handle handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_capacity = 0;
};
static thread_local ShimState g_shim;

static bool ensure_shim_handle() {
    if (g_shim.handle) return true;
    return rocblas_create_handle(&g_shim.handle) == rocblas_status_success;
}

static size_t compute_workspace_bytes(rocblas_int m, rocblas_int n, rocblas_int k) {
    int gsu_factor = (k < MAX_GSU) ? k : MAX_GSU;
    if (gsu_factor < 1) gsu_factor = 1;
    size_t bytes = (size_t)m * (size_t)n * sizeof(float) * (size_t)gsu_factor;
    bytes = ((bytes + 255) / 256) * 256; // match rocBLAS's own GSU workspace granularity
    if (bytes < 4096) bytes = 4096;
    return bytes;
}

static bool ensure_shim_workspace(size_t needed_bytes) {
    if (needed_bytes <= g_shim.workspace_capacity) return true;
    if (g_shim.workspace) {
        hipFree(g_shim.workspace);
        g_shim.workspace = nullptr;
        g_shim.workspace_capacity = 0;
    }
    if (hipMalloc(&g_shim.workspace, needed_bytes) != hipSuccess) return false;
    if (rocblas_set_workspace(g_shim.handle, g_shim.workspace, needed_bytes) != rocblas_status_success) return false;
    g_shim.workspace_capacity = needed_bytes;
    return true;
}

// Dispatches through our private handle with a freshly-zeroed workspace,
// via the real rocblas_gemm_ex with rocBLAS's own default solution
// selection (algo=standard, solution_index=0) -- i.e. AMD's real kernel,
// just with the buggy pool reuse sidestepped.
static rocblas_status dispatch_via_private_handle(rocblas_operation transA, rocblas_operation transB,
                                                   rocblas_int m, rocblas_int n, rocblas_int k,
                                                   const void* alpha,
                                                   const void* a, rocblas_int lda,
                                                   const void* b, rocblas_int ldb,
                                                   const void* beta,
                                                   void* c, rocblas_int ldc,
                                                   hipStream_t caller_stream) {
    if (!ensure_shim_handle()) return rocblas_status_memory_error;

    size_t needed = compute_workspace_bytes(m, n, k);
    if (!ensure_shim_workspace(needed)) return rocblas_status_memory_error;

    rocblas_set_stream(g_shim.handle, caller_stream);
    if (hipMemsetAsync(g_shim.workspace, 0, needed, caller_stream) != hipSuccess) {
        return rocblas_status_memory_error;
    }

    return real_rocblas_gemm_ex(g_shim.handle, transA, transB, m, n, k, alpha,
                                a, rocblas_datatype_f32_r, lda,
                                b, rocblas_datatype_f32_r, ldb,
                                beta,
                                c, rocblas_datatype_f32_r, ldc,
                                c, rocblas_datatype_f32_r, ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard, 0, 0);
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

    rocblas_status st = dispatch_via_private_handle(transA, transB, m, n, k, alpha, A, lda, B, ldb,
                                                     beta, C, ldc, stream);
    if (st != rocblas_status_success) {
        fprintf(stderr, "gfx803_sgemm_shim: private-handle dispatch failed (status=%d), falling back to rocBLAS\n", (int)st);
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
                     && !env_shim_disabled() && c == d; // in-place C==D matches this dispatch's beta-accumulate semantics

    if (!take_over) {
        return real_rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha,
                                    a, a_type, lda, b, b_type, ldb, beta,
                                    c, c_type, ldc, d, d_type, ldd,
                                    compute_type, algo, solution_index, flags);
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    rocblas_status st = dispatch_via_private_handle(transA, transB, m, n, k, alpha, a, lda, b, ldb,
                                                     beta, d, ldd, stream);
    if (st != rocblas_status_success) {
        fprintf(stderr, "gfx803_sgemm_shim: gemm_ex private-handle dispatch failed (status=%d), falling back to rocBLAS\n", (int)st);
        return real_rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha,
                                    a, a_type, lda, b, b_type, ldb, beta,
                                    c, c_type, ldc, d, d_type, ldd,
                                    compute_type, algo, solution_index, flags);
    }
    return rocblas_status_success;
}
