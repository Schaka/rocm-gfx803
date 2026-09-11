"""gfx803 (Polaris/GCN3) multi-token GEMV for decode at 2 <= M <= 16.

Decode at M > 1 previously had no gfx803-specific path at all: the tiled
GEMM in gfx803_prefill_gemm.py tiles 64 activation rows, so it does 8x the
MACs at M=8 and measured 12-48 GB/s of the card's ~168 GB/s copy bandwidth
at these sizes (see that module's docstring and NOTES.md). This kernel keeps
the LLMM1 structure -- a wave owns its output rows and its lanes walk K in
8-half chunks -- and adds the one thing LLMM1 cannot do: it divides the wave
between rows and lanes-per-row so that a lane's weight chunk is reused for
every token of the batch, with one fp32 accumulator per (row, token) pair.
LLMM1 asserts N == 1 for exactly the opposite reason: its grid is the output
feature axis, so a second token would need a second set of blocks to
re-read the same weights.

Measured per-decode-step GEMM time for this model's shapes (28 layers plus
lm_head, M=2/4/8/16): 10.8 / 11.1 / 14.9 / 31.4 ms, against ~33 ms for the
tiled GEMM path at M=8. The budget is a memory roofline of ~6.7 ms per step
-- 1134 MB of weights at 169 GB/s -- so what is left is not arithmetic:
adding a token costs registers (MTOK=2/4/8/16 compile to 30/42/61/101 VGPRs)
and registers cost waves per CU, which is what a latency-bound kernel needs
in order to keep enough weight loads in flight. An earlier variant that put
the tokens in the lanes instead held 28 VGPRs flat and measured slower at
every batch size, because a weight chunk then fed one token per lane and the
batch's reuse had to come from cache rather than registers.

Unlike gfx803_prefill_gemm.py this needs no transposed weight copy and so no
per-weight or total VRAM budget: the kernel reads vLLM's native [N, K]
layout, because lanes are adjacent along K and a native row-major read is
therefore already coalesced. Weights of any size are eligible, which is what
brings lm_head (a quarter of this model's weights, past the tiled GEMM's
per-weight cache cap) onto a fast path at M > 1.

Compile, from this directory (the loader builds its ctypes path from
`__file__`, so the .so has to land next to it):
    hipcc --offload-arch=gfx803 -O3 -shared -fPIC \
      -o libgfx803gemv_m.so ../../gfx803_kernels/gfx803_gemv_m.hip
"""

import ctypes
import functools
import os

import torch

_LIB_PATH = __file__.rsplit("/", 1)[0] + "/libgfx803gemv_m.so"

# Exists so that this path can be A/B'd against the tiled GEMM's inside one
# engine configuration, the way VLLM_GFX803_GEMM_CACHE_MB and
# VLLM_GFX803_ATTN_HIP_KERNEL do for the other gfx803 paths. Read at import
# because a process picks one configuration and keeps it.
_ENABLED = os.environ.get("VLLM_GFX803_GEMV_M", "1") != "0"

_ARG_TYPES = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # W
    ctypes.c_void_p,  # C
    ctypes.c_int,  # M
    ctypes.c_int,  # N
    ctypes.c_int,  # K
    ctypes.c_void_p,  # stream
]


@functools.lru_cache(maxsize=None)
def _launcher(symbol: str):
    try:
        lib = ctypes.CDLL(_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # No compiled .so next to this file, which is the normal state of a
        # fresh checkout: the kernels are built on the machine that runs them.
        return None, None
    fn = getattr(lib, symbol)
    fn.argtypes = _ARG_TYPES
    fn.restype = None
    # The library object is returned alongside the symbol because the symbol
    # alone does not keep the dlopen'd handle alive.
    return lib, fn


def gfx803_gemv_m_into(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> bool:
    """Write x @ weight.T into out, which must be [M, N] fp16.

    Exists separately from gfx803_gemv_m so that a caller can hand in a
    buffer whose contents it controls: the probe fills out with NaN first, so
    that an output the kernel never wrote is reported instead of being read
    as a numerical error.

    Args:
        x: [M, K] fp16, row-major and contiguous.
        weight: [N, K] fp16, vLLM's native row-major layout.
        out: [M, N] fp16 destination, written in full.

    Returns:
        True if the kernel ran. False if M is outside the kernel's range, the
        shapes do not line up, or the compiled kernel is absent, in which case
        out is left untouched.
    """
    if not _ENABLED:
        return False
    M, K = x.shape
    N = weight.shape[0]
    if not 2 <= M <= 16 or K != weight.shape[1] or out.shape != (M, N):
        return False
    _, fn = _launcher("gfx803_gemv_m_launch")
    if fn is None:
        return False
    fn(
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        M,
        N,
        K,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    return True


def gfx803_gemv_m(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """C = x @ weight.T for 2 <= M <= 16, fp32 accumulation throughout.

    Args:
        x: [M, K] fp16, row-major and contiguous.
        weight: [N, K] fp16, vLLM's native row-major layout.

    Returns:
        [M, N] fp16, or None if M is outside the kernel's range or the
        compiled kernel is absent, so that the caller can fall back.
    """
    if _launcher("gfx803_gemv_m_launch")[1] is None:
        return None
    M, K = x.shape
    out = torch.empty(M, weight.shape[0], dtype=torch.float16, device=x.device)
    if not gfx803_gemv_m_into(x, weight, out):
        return None
    return out

