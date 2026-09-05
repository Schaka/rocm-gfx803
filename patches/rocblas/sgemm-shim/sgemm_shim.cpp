// LD_PRELOAD shim: intercepts rocblas_sgemm / rocblas_gemm_ex / 
// rocblas_gemm_strided_batched_ex and routes the standard-algo fp32 path to
// gfx803_sgemm.h instead of rocBLAS/Tensile's own kernels, which are
// unreliable on gfx803 for every shape tested so far -- see gfx803_sgemm.h's
// header comment for the full investigation.
//
// FP16 ROUTE: rocblas_gemm_ex is also taken over when both operands and both outputs are
// f16, alpha=1, beta=0, C==D, no transposes and tight leading dimensions. That route calls
// the row-major kernel in gfx803_gemm_lib.hip, and the dimension order it passes matters:
// under exactly the contract the gate checks, the row-major product that lands in the
// output buffer is b @ a with M=n, N=m, K=k. Handing it (m, n) computes the transpose and
// reads m*k elements out of a k*n buffer, so it is only right when m == n. Measured on the
// card against a float32 reference, 27 cases (tools/correctness-suite/fp16_gemm_sweep.py):
// the old order was correct on the 8 square cases and wrong or faulting on all 19 others,
// including every fp16 convolution (im2col gives K = Cin*R*S and M = H*W, so m != n
// essentially always -- a 3-channel 3x3 conv arrives as m=4096, n=64, k=27) and the
// non-square batched dots. The fixed order is correct on all 27, and the 8 square cases
// are unchanged, because for m == n the two calls are the same call.
//
// The kernel itself is not at fault: called directly on its own documented contract it is
// correct at M=4096, N=64, K=27 (max abs error 7e-05 against a double reference), so this
// was a caller bug, not a kernel bug.
//
// SCOPE: as of this patch, this intercepts ALL standard-algo f32
// rocblas_sgemm/rocblas_gemm_ex calls, not just specific proven-broken
// shapes -- there is no known-good subset to preserve Tensile's speed for
// yet -- plus the f16 gemm_ex case described above and the small-problem
// strided-batched case described below. Anything outside those three
// (bf16, i8, explicit solution_index requests, non-standard algo, large
// batched problems) falls through untouched to the real rocBLAS symbol via
// dlsym(RTLD_NEXT, ...). Each takeover is asserted at build time by a marker
// string in the compiled .so, so a build that silently predates one of these
// routes fails the image rather than shipping it.
//
// rocblas_gemm_strided_batched_ex is intercepted too, but only for
// max(m,n,k) <= 32 -- the small-problem region where gfx803's assembly
// kernels are known-broken (the 6.4.4 sweep measured them correct from
// ~48x48x48 up). Batched attention dots (QK/QV, batch collapsed into
// m/n) land here and miscompute on 7.14; large batched GEMMs stay on
// real rocBLAS because the simple kernel is correctness-verified but not
// tuned, and routing 4096^3 through it would cost far more than the fix
// is worth.
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

// fp16 path: same defect class as NOTE 2 but on the fp16 Tensile kernels --
// rocBLAS's internal device-memory pool reuses memory across GEMM calls as
// GlobalSplitU scratch without re-zeroing it, and gfx803's GSU-reduction
// kernels (software CAS, assumes zeroed scratch) accumulate garbage into the
// output on top of whatever the previous GEMM left there. The hand-written
// kernel from the vLLM fork (gfx803_gemm_lib.hip, compiled into this
// shim) stages through LDS with no scratch at all, so a preceding GEMM
// cannot leak into it. Confirmed on real hardware: torch's fp16
// bmm/matmul (2x128x64x128 etc.) produces scattered NaNs through rocBLAS
// whenever a same-shape GEMM ran first; the hand-written kernel via the swap
// mapping below is exact for every shape tested.
//
// rocBLAS is column-major; the kernel is row-major. For the no-transpose
// contiguous case (lda==m, ldb==k, ldc==ldd==m) the mapping is a pure
// dimension swap: kernel(M=n, N=m, K=k, A=b, B=a, C=d). Take-over
// contract: fp16 in/out, no transpose, contiguous, in-place C==D,
// alpha==1, beta==0 (the kernel computes C=A@B with no scaling), default
// solution. Everything else falls through to real rocBLAS untouched.
extern "C" void gfx803_gemm_launch(const void* A, const void* B, void* C,
                                    int M, int N, int K, void* stream);

static bool f16_layout_ok(rocblas_operation transA, rocblas_operation transB,
                         rocblas_datatype a_type, rocblas_datatype b_type,
                         rocblas_datatype c_type, rocblas_datatype d_type,
                         rocblas_int m, rocblas_int n, rocblas_int k,
                         const void* alpha, const void* beta,
                         const void* c, const void* d,
                         rocblas_int lda, rocblas_int ldb, rocblas_int ldc, rocblas_int ldd,
                         rocblas_gemm_algo algo, int32_t solution_index,
                         rocblas_stride stride_a, rocblas_stride stride_b,
                         rocblas_stride stride_c, rocblas_stride stride_d) {
    bool all_f16 = (a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
                    c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r);
    bool scale_one = alpha && beta && *(const float*)alpha == 1.0f && *(const float*)beta == 0.0f;
    bool contiguous = lda == m && ldb == k && ldc == m && ldd == m &&
                     stride_a == (rocblas_stride)m * k && stride_b == (rocblas_stride)k * n &&
                     stride_c == (rocblas_stride)m * n && stride_d == (rocblas_stride)m * n;
    return all_f16 && transA == rocblas_operation_none && transB == rocblas_operation_none &&
           c == d && scale_one && contiguous && algo == rocblas_gemm_algo_standard &&
           solution_index == 0;
}

static void route_f16(const void* a, const void* b, void* d, int m, int n, int k,
                   rocblas_handle handle) {
    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);
    if (getenv("GFX803_SGEMM_SHIM_DEBUG")) {
        // [f16-takeover] marker for the Dockerfile build guard
        fprintf(stderr, "shim f16: a=%p b=%p d=%p m=%d n=%d k=%d [f16-takeover] [f16-map-nm]\n", a, b, d, m, n, k);
    }
    // The kernel computes a row-major product: C[M,N] = X[M,K] @ Y[K,N], with X's
    // row stride K and Y's row stride N. The caller's problem is the plain
    // column-major contract D(i,j) = op(A)(i,k) op(B)(k,j) with no transposes and
    // lda=m, ldb=k, ldd=m (the gate above insists on all three), and under that
    // contract the buffers read row-major are: b -> [n,k] stride ldb=k,
    // a -> [k,m] stride lda=m, d -> [n,m] stride ldd=m. So the row-major product
    // that lands in d's memory is b @ a with M=n, N=m, K=k.
    //
    // Handing it (m, n) instead computes the transpose of the answer, which is
    // only the same buffer layout when m == n. With m != n the kernel also walks
    // m*k elements out of a k*n buffer, which is the out-of-bounds read that made
    // convolutions fault: an im2col call arrives as m=4096, n=64, k=27 and read
    // the 64x27-element operand as if it held 4096x27.
    gfx803_gemm_launch(b, a, d, n, m, k, stream);
}

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

typedef rocblas_status (*rocblas_gemm_strided_batched_ex_t)(
    rocblas_handle, rocblas_operation, rocblas_operation,
    rocblas_int, rocblas_int, rocblas_int,
    const void*, const void*, rocblas_datatype, rocblas_int, rocblas_stride,
    const void*, rocblas_datatype, rocblas_int, rocblas_stride,
    const void*, const void*, rocblas_datatype, rocblas_int, rocblas_stride,
    void*, rocblas_datatype, rocblas_int, rocblas_stride,
    rocblas_int, rocblas_datatype, rocblas_gemm_algo, int32_t, uint32_t);

static rocblas_sgemm_t real_rocblas_sgemm = nullptr;
static rocblas_gemm_ex_t real_rocblas_gemm_ex = nullptr;
static rocblas_gemm_strided_batched_ex_t real_rocblas_gemm_strided_batched_ex = nullptr;

static void ensure_real_symbols() {
    if (!real_rocblas_sgemm) {
        real_rocblas_sgemm = (rocblas_sgemm_t)dlsym(RTLD_NEXT, "rocblas_sgemm");
    }
    if (!real_rocblas_gemm_ex) {
        real_rocblas_gemm_ex = (rocblas_gemm_ex_t)dlsym(RTLD_NEXT, "rocblas_gemm_ex");
    }
    if (!real_rocblas_gemm_strided_batched_ex) {
        real_rocblas_gemm_strided_batched_ex =
            (rocblas_gemm_strided_batched_ex_t)dlsym(RTLD_NEXT, "rocblas_gemm_strided_batched_ex");
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

    if (f16_layout_ok(transA, transB, a_type, b_type, c_type, d_type, m, n, k,
                      alpha, beta, c, d, lda, ldb, ldc, ldd, algo, solution_index,
                      /*unused strides*/ m * k, k * n, m * n, m * n)
        && !env_shim_disabled()) {
        route_f16(a, b, d, m, n, k, handle);
        hipError_t herr = hipGetLastError();
        if (herr != hipSuccess) {
            fprintf(stderr, "gfx803_sgemm_shim: f16 gemm_ex kernel launch failed (%s), falling back to rocBLAS\n",
                    hipGetErrorString(herr));
        } else {
            return rocblas_status_success;
        }
    }

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

extern "C" rocblas_status rocblas_gemm_strided_batched_ex(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const void* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda, rocblas_stride stride_a,
    const void* b, rocblas_datatype b_type, rocblas_int ldb, rocblas_stride stride_b,
    const void* beta,
    const void* c, rocblas_datatype c_type, rocblas_int ldc, rocblas_stride stride_c,
    void* d, rocblas_datatype d_type, rocblas_int ldd, rocblas_stride stride_d,
    rocblas_int batch_count, rocblas_datatype compute_type, rocblas_gemm_algo algo,
    int32_t solution_index, uint32_t flags) {
    ensure_real_symbols();

    bool all_f32 = (a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
                    c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
                    compute_type == rocblas_datatype_f32_r);
    // Take-over contract: f32, default solution, in-place C==D, shim not
    // disabled, scoped to the small-problem region where gfx803's Tensile
    // assembly kernels are known unreliable. This is the strided-batched
    // entry point that batched gpu::gemm lowering hits for attention QK/QV
    // dots (batch dims collapsed into the gemm's M/N), which neither the
    // blanket sgemm/gemm_ex shim nor the small-gemm patch's all-dims-<=8
    // gate covers -- the GSU workspace-reuse race class from KERNEL_BUGS.md
    // still miscomputes there on 7.14 (seen: attention GQA tests, wrong by
    // up to ~4.5x). Large batched gemms stay on real rocBLAS: the simple
    // kernel is correctness-verified but not tuned, and forcing e.g. 4096^3
    // batched GEMMs through it would cost far more than the correctness fix
    // is worth. The 6.4.4 sweep measured the assembly kernels correct from
    // ~48x48x48 up, so max-dim <= 32 is safely inside the broken region.
    bool small_problem = (m <= 32 && n <= 32 && k <= 32);
    // NOTE: no algo == rocblas_gemm_algo_standard check here (unlike the
    // gemm_ex interceptor above): MIGraphX's batched gemm lowering passes
    // algo=rocblas_gemm_algo_1, not standard, so an algo gate would silently
    // reject every take-over and fall through to the broken real rocBLAS
    // kernel. The gfx803_sgemm kernel ignores the algo hint anyway -- it
    // implements the standard semantic -- so restricting on it buys nothing
    // here.
    // Marker for the Dockerfile build guard (grep for
    // "sb-takeover-no-algo-gate" in the built .so to confirm this fix is
    // actually in the shipped binary). Carried in the debug fprintf's format
    // string below so the literal survives into .rodata -- a plain unused
    // const gets optimized away.
    bool take_over = all_f32 && solution_index == 0 && !env_shim_disabled() && c == d
                     && small_problem && stride_c == stride_d && ldc == ldd;
    if (getenv("GFX803_SGEMM_SHIM_DEBUG")) {
        fprintf(stderr,
                "shim sb: m=%d n=%d k=%d batch=%d f32=%d algo=%d si=%d c==d=%d small=%d "
                "sc==sd=%d ldc==ldd=%d disabled=%d [sb-takeover-no-algo-gate]\n",
                m, n, k, batch_count, all_f32, (int)algo, solution_index, c == d,
                small_problem, stride_c == stride_d, ldc == ldd, env_shim_disabled());
    }

    if (f16_layout_ok(transA, transB, a_type, b_type, c_type, d_type, m, n, k,
                      alpha, beta, c, d, lda, ldb, ldc, ldd, algo, solution_index,
                      stride_a, stride_b, stride_c, stride_d)
        && !env_shim_disabled()) {
        for (rocblas_int i = 0; i < batch_count; i++) {
            route_f16((const char*)a + i * stride_a * 2, (const char*)b + i * stride_b * 2,
                     (char*)d + i * stride_d * 2, m, n, k, handle);
            hipError_t herr = hipGetLastError();
            if (herr != hipSuccess) {
                fprintf(stderr,
                        "gfx803_sgemm_shim: f16 strided_batched kernel launch failed (%s), "
                        "falling back to rocBLAS\n",
                        hipGetErrorString(herr));
                return real_rocblas_gemm_strided_batched_ex(
                    handle, transA, transB, m, n, k, alpha,
                    a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta,
                    c, c_type, ldc, stride_c, d, d_type, ldd, stride_d,
                    batch_count, compute_type, algo, solution_index, flags);
            }
        }
        return rocblas_status_success;
    }

    if (!take_over) {
        return real_rocblas_gemm_strided_batched_ex(
            handle, transA, transB, m, n, k, alpha,
            a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta,
            c, c_type, ldc, stride_c, d, d_type, ldd, stride_d,
            batch_count, compute_type, algo, solution_index, flags);
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    bool tA = (transA == rocblas_operation_transpose);
    bool tB = (transB == rocblas_operation_transpose);

    for (rocblas_int i = 0; i < batch_count; i++) {
        gfx803_sgemm((int)m, (int)n, (int)k, *(const float*)alpha,
                     (const float*)a + i * stride_a, (int)lda, tA,
                     (const float*)b + i * stride_b, (int)ldb, tB,
                     *(const float*)beta, (float*)d + i * stride_d, (int)ldd, stream);
        hipError_t herr = hipGetLastError();
        if (herr != hipSuccess) {
            fprintf(stderr,
                    "gfx803_sgemm_shim: strided_batched kernel launch failed (%s), "
                    "falling back to rocBLAS\n",
                    hipGetErrorString(herr));
            return real_rocblas_gemm_strided_batched_ex(
                handle, transA, transB, m, n, k, alpha,
                a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta,
                c, c_type, ldc, stride_c, d, d_type, ldd, stride_d,
                batch_count, compute_type, algo, solution_index, flags);
        }
    }
    return rocblas_status_success;
}
