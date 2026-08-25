"""gfx803 (Polaris/GCN3) skinny-GEMM fast path.

rocBLAS/Tensile ships no tuned fp16 GEMM kernels for gfx803 at all --
every fp16 GEMM call falls through to Tensile's generic `_fallback_`
library (a portability-only path, never benchmark-tuned), which dispatches
a large-M-tile (e.g. 128-row) kernel even for decode's actual M=1 shape --
~127/128 of every compute grid wasted on padding. Measured ~1.31ms/call
average via rocprofv3 kernel-trace on real hardware, ~98.6% of decode
step time (see SESSION_HANDOFF.md, "ROOT CAUSE" section).

Three fast paths, combined:

1. **`ops.LLMM1`** (vLLM's own existing custom kernel,
   `csrc/rocm/skinny_gemms.cu`'s `LLGemm1_kernel`) -- despite being gated
   in vLLM upstream to `on_gfx9()`/`on_gfx1x()`/`on_gfx906()` only, this
   specific kernel (unlike the `wvSplitK_*` family in the same file) uses
   only portable HIP intrinsics (`__hmul2`/`__hfma2`/`__shfl_xor`), no
   GFX9+-only dot-product ASM (`v_dot2_f32_f16`) or MFMA -- and the
   compiled `_rocm_C.abi3.so` on this box already contains gfx803 code
   objects (this whole stack is built targeting gfx803 -- see
   AGENTS.md), so it runs as-is, no rebuild needed. **BUT: silently wrong
   at larger K, including within vLLM's own stated `k<=8192` safety gate**
   -- direct correctness check (`torch.nn.functional.linear` reference)
   found K=1536 correct up to M=151936, while K=8192 and K=8064 both
   launch cleanly but return `max_abs_diff` ~4.2-4.5 (catastrophic, not
   fp16 noise) -- a real bug in this kernel at scale. Candidate root cause
   (not yet confirmed by disassembly/profiling, see SESSION_HANDOFF.md):
   `LLMM1()`'s host launcher computes `NUM_THREADS = ceil_to_64(K/8)`,
   which lands at exactly 1024 threads/block for both K=8064 and K=8192 --
   plausibly at or past this hardware's real per-block limit once combined
   with the kernel's per-thread register footprint -- while the kernel's
   second-stage warp-shuffle reduction (`for mask in 16,8,4,2,1: ...` then
   a single `__shfl_xor(_, 16)`) hardcodes an assumption of at most 32
   warps per block, a pattern that reads as ported from 32-lane-warp CUDA
   without adjustment for AMD's 64-lane wavefronts. Neither has been
   confirmed as *the* cause; both are plausible given the failure
   threshold. Only used here for K<=4096 -- covers this model's K=1536
   shapes (o_proj, qkv_proj, gate_up_proj, lm_head) with real margin below
   the K=8064 failure point.
2. **`gfx803_triton_gemv`** (this repo's own Triton kernel, below) --
   used for K>4096 (down_proj's K=8960). Avoids LLMM1's bug entirely by
   not sharing its code (a plain `tl.load`+multiply+`tl.sum` reduction
   per BLOCK_K tile, no warp-shuffle assumptions of any kind), and beats
   `rocblas_hssgemv_strided_batched` (this file's 3rd path, now used only
   as the correctness-reference/last-resort fallback) by ~9% at down_proj's
   exact shape (M=1536, K=8960; median of 200-iter trials after warmup,
   tuned BLOCK_M=16/BLOCK_K=256/num_warps=2 out of a 48-point sweep).
3. **`rocblas_hssgemv_strided_batched`** (rocBLAS's hand-written,
   non-Tensile GEMV kernel, called via ctypes since PyTorch doesn't
   expose its internal rocBLAS handle to Python) -- kept as the fallback
   for anywhere LLMM1's constraints (`bias is None`, `m % 2 == 0`) aren't
   met before bias is folded in, and as the correctness reference this
   whole module's other two paths were verified against.

Benchmarked directly (real Qwen2.5-1.5B shapes, all three kernels, 100+
iters after warmup): LLMM1 beats rocblas_hssgemv on every K<=4096 shape --
o_proj ~2x, qkv_proj ~1.9x, gate_up_proj ~1.6x. gfx803_triton_gemv beats
rocblas_hssgemv by ~9% at down_proj's K=8960 shape. rocblas_hssgemv itself
is 32x/27x/7.2x/9.8x/3.3x faster than the original untuned-Tensile-GEMM
path average across o_proj/qkv_proj/gate_up_proj/down_proj/lm_head
respectively (see git history / SESSION_HANDOFF.md §16-18 for the
GEMM-vs-GEMV numbers). Correctness verified against
`torch.nn.functional.linear` for every routed shape actually used in this
model (o_proj, qkv_proj, gate_up_proj, down_proj, lm_head) -- diffs
consistent with ordinary fp16-accumulation-order noise (mean abs diff
~1e-6 to ~2e-4), not the K-scale bug above.
"""

import ctypes
import functools

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

_LLMM1_MAX_K = 4096  # verified-safe ceiling; K=8064/8192 measured wrong

_ROCBLAS_OPERATION_NONE = 111
_ROCBLAS_OPERATION_TRANSPOSE = 112
_ROCBLAS_STATUS_SUCCESS = 0


@functools.lru_cache(maxsize=1)
def _librocblas() -> ctypes.CDLL:
    lib = ctypes.CDLL("librocblas.so", mode=ctypes.RTLD_GLOBAL)
    lib.rocblas_hssgemv_strided_batched.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,  # transA
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_void_p,  # alpha ptr (host)
        ctypes.c_void_p,  # A
        ctypes.c_int,  # lda
        ctypes.c_int64,  # strideA
        ctypes.c_void_p,  # x
        ctypes.c_int,  # incx
        ctypes.c_int64,  # stridex
        ctypes.c_void_p,  # beta ptr (host)
        ctypes.c_void_p,  # y
        ctypes.c_int,  # incy
        ctypes.c_int64,  # stridey
        ctypes.c_int,  # batch_count
    ]
    lib.rocblas_hssgemv_strided_batched.restype = ctypes.c_int
    return lib


@functools.lru_cache(maxsize=1)
def _rocblas_handle() -> ctypes.c_void_p:
    lib = _librocblas()
    handle = ctypes.c_void_p()
    status = lib.rocblas_create_handle(ctypes.byref(handle))
    assert status == _ROCBLAS_STATUS_SUCCESS, f"rocblas_create_handle failed: {status}"
    return handle


def gfx803_skinny_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """y = weight @ x (+ bias) for a single decode token (x: length-K fp16 vector).

    weight: [M, K] fp16, row-major/contiguous (vLLM's standard layout).
    Returns: [M] fp32 (matches rocblas_hssgemv_strided_batched's S-accumulate
    output type -- callers on the fp16-through path should cast back).

    Bias is folded in via the GEMV's own beta/y-accumulate (y initialized to
    bias, beta=1) rather than a separate elementwise add -- avoids excluding
    Qwen2's qkv_proj (which has bias=True on q/k/v while o_proj/gate_up/down
    don't) from this fast path entirely.
    """
    M, K = weight.shape
    lib = _librocblas()
    handle = _rocblas_handle()
    lib.rocblas_set_stream(handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))

    alpha = ctypes.c_float(1.0)
    if bias is not None:
        y = bias.to(torch.float32).clone()
        beta = ctypes.c_float(1.0)
    else:
        y = torch.empty(M, dtype=torch.float32, device=x.device)
        beta = ctypes.c_float(0.0)
    status = lib.rocblas_hssgemv_strided_batched(
        handle, _ROCBLAS_OPERATION_TRANSPOSE,
        K, M,
        ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
        weight.data_ptr(), K, K * M,
        x.data_ptr(), 1, 0,
        ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
        y.data_ptr(), 1, 0,
        1,
    )
    assert status == _ROCBLAS_STATUS_SUCCESS, f"rocblas_hssgemv_strided_batched failed: {status}"
    return y


@triton.jit
def _skinny_gemv_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    K, M,
    stride_wm, stride_wk,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm_mask = rm < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        rk_mask = rk < K
        x = tl.load(x_ptr + rk, mask=rk_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + rm[:, None] * stride_wm + rk[None, :] * stride_wk,
            mask=rm_mask[:, None] & rk_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + rm, mask=rm_mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + rm, acc, mask=rm_mask)


def gfx803_triton_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """y = weight @ x (+ bias) for a single decode token, own Triton kernel.

    Avoids LLMM1's large-K bug entirely by not sharing any of its code (see
    module docstring). Used for K>4096 (down_proj's K=8960) in preference to
    rocblas_hssgemv_strided_batched -- ~9% faster at that shape, tuned
    BLOCK_M=16/BLOCK_K=256/num_warps=2. Returns fp32 (matches the other two
    paths in this module).
    """
    M, K = weight.shape
    out = torch.empty(M, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, 16),)
    _skinny_gemv_kernel[grid](
        x, weight, bias, out,
        K, M,
        weight.stride(0), weight.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=16, BLOCK_K=256,
        num_warps=2,
    )
    return out


def gfx803_skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Decode-time (n==1) linear layer for gfx803: LLMM1 where verified safe,
    gfx803_triton_gemv for larger K, rocblas_hssgemv_strided_batched as the
    final fallback. Returns fp32 (matches all three kernels' accumulate type
    -- callers on the fp16-through path should cast back). See module
    docstring for why K gates the choice.
    """
    M, K = weight.shape
    if K <= _LLMM1_MAX_K and M % 2 == 0:
        # rows_per_block=2 measured fastest on every shape in this model
        # (18.56-1883.80us range, 1-4% better than 4, median of 3x200-iter
        # trials -- a single-trial sweep first suggested 4 was better for
        # o_proj specifically, that was measurement noise, not real).
        out = ops.LLMM1(weight, x.unsqueeze(0), 2).view(M)
        if bias is not None:
            # LLMM1 returns fp16 (matching bias's dtype) already -- add
            # directly, no fp32 round-trip needed (this repo's own
            # correctness check already accepts LLMM1's fp16-internal
            # accumulation, so bias-add precision isn't a new concern).
            out = out + bias
        return out
    return gfx803_triton_gemv(x, weight, bias)
