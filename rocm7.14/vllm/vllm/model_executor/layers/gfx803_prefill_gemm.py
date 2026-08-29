"""gfx803 (Polaris/GCN3) hand-written GEMM for prefill (n>1) linear layers.

Routed to ONLY gate_up_proj-shaped weights (out_features >= 8192, see the
caller-side gate in utils.py), no upper or lower M bound -- re-measured
head-to-head against rocBLAS/Tensile's fixed gfx803 fp16 HPA kernel
(`patches/rocblas/tensile-gfx803-fp16-nond16.patch` +
`r9nano_Cijk_Ailk_Bljk_HB.yaml`) on real, VRAM-clock-fixed hardware: this
kernel wins at every M measured, 128 through 8192. Coverage is bounded
instead by VLLM_GFX803_GEMM_CACHE_MB (see _MAX_TOTAL_CACHE_BYTES below) --
weights past that budget fall back to Tensile every call, which still
wins at o_proj/down_proj-shaped weights it was never tuned to beat.

The kernel itself: root-caused via direct inspection of the compiled
gfx803 ISA (extract the code object with `clang-offload-bundler
-unbundle`, read resource usage with `llvm-readobj --notes`) -- a plain
tiled+register-blocked design was silently spilling 35 VGPRs/thread to
memory every iteration because the compiler was targeting a
higher-occupancy config than this kernel's real register need supports.
`__launch_bounds__(256, 1)` (trade occupancy for register budget)
eliminates the spilling entirely, and combined with double buffering
(prefetch the next K-tile while computing on the current one) is what
gets it competitive at all (see SESSION_HANDOFF.md section 30 for the
full investigation).

Compiled once via `hipcc --offload-arch=gfx803 -O3 -shared -fPIC -o
libgfx803gemm.so ../../gfx803_kernels/gfx803_gemm_lib.hip` (source lives
in `vllm/gfx803_kernels/`, not this directory -- the compiled `.so` still
needs to land here, next to this loader, for the `__file__`-relative
ctypes path below to find it), loaded here via ctypes -- same pattern as
gfx803_gemv.py's rocBLAS bridge, since PyTorch doesn't expose a way to
launch an arbitrary custom HIP kernel from
Python without either a full C++ extension rebuild or this ctypes
approach.

IMPORTANT layout requirement: the kernel computes C[M,N] = A[M,K] @
B[K,N], with B expected in [K,N] row-major layout -- the *transpose* of
vLLM's standard [N,K] (out_features, in_features) weight storage.
Transposing per call would erase the whole performance win, so this
module caches one transposed-contiguous copy of each weight tensor
(keyed by the weight tensor's identity) the first time it's used, exactly
the same one-time-cost-then-reuse pattern as everything else in this
gfx803 kernel set.

A native-[N,K]-layout variant (no transpose, no cache) was tried and
reverted -- see SESSION_HANDOFF.md section 31/32. Reading the weight in
its native layout means adjacent threads land on far-apart rows (K*2
bytes apart) instead of the contiguous, coalesced reads the transposed
layout gives, and that per-call slowdown outweighed the coverage gain it
would have bought back then (moot now that coverage is intentionally
narrowed to gate_up_proj only).
"""

import ctypes
import functools
import os

import torch

_LIB_PATH = __file__.rsplit("/", 1)[0] + "/libgfx803gemm.so"


@functools.lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    lib.gfx803_gemm_launch.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # C
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.c_void_p,  # stream
    ]
    lib.gfx803_gemm_launch.restype = None
    return lib


# weight tensor id() -> transposed-contiguous [K, N] fp16 copy. Weights are
# static after model load, so this is a true one-time cost per weight, not
# a per-call cost -- id() is safe here because we only ever cache a live
# nn.Parameter's storage for the process lifetime (never a transient
# tensor), matching how vLLM itself holds weight references.
#
# The caller (utils.py) now only routes gate_up_proj-shaped weights here
# (see this module's docstring), so the cap only needs to admit that one
# shape -- ~55MB/layer * 28 layers ~= 1.5GB total, a real, permanent VRAM
# cost (this box's RX 470 has 8GB total), accepted because it's the only
# shape left where this kernel actually wins once Tensile's fixed fp16
# kernel is in the built image. Previously this cap was set to EXCLUDE
# gate_up_proj and admit o_proj/down_proj instead, back when this kernel
# was the fast path for those shapes too -- see SESSION_HANDOFF.md section
# 33 for why that routing changed.
_MAX_CACHED_WEIGHT_BYTES = 64 * 1024 * 1024
# Total budget across every cached weight, not just the per-weight cap
# above. Default 768MB (~10% of an 8GB card) leaves ~8-12 of Qwen3.5-2B's
# 24 gate_up_proj layers permanently uncached (full coverage needs
# ~1.15GB: 24 * ~48MB/layer at K=2048 N=12288 fp16), so those layers'
# prefill calls fall through to Tensile every time. Raising this budget
# buys more prefill coverage; the cache is populated during the first real
# forward pass, before vLLM's KV-cache memory profiling settles, so a
# bigger cache does shrink the KV-cache pool -- but measured decode
# throughput is unaffected by that shrink at every budget tried (768MB and
# 1200MB both measured at 22.6 tok/s on this box, Qwen3.5-2B, 2101-token
# prompt, RX 470 8GB, gpu_memory_utilization=0.98, single-request batch=1,
# 2026-08-29 -- decode is bounded by the Triton paged-attention fallback
# and rocBLAS's GEMV kernel, neither of which this cache touches). Raise
# this freely for prefill-heavy workloads via the VLLM_GFX803_GEMM_CACHE_MB
# env var (MB, not bytes) -- see README.md's gfx803 section for the
# measured curve.
_MAX_TOTAL_CACHE_BYTES = int(os.environ.get("VLLM_GFX803_GEMM_CACHE_MB", "768")) * 1024 * 1024
_TRANSPOSED_WEIGHT_CACHE: dict[int, torch.Tensor] = {}
_cache_bytes_used = 0


def _transposed_weight(weight: torch.Tensor) -> torch.Tensor | None:
    global _cache_bytes_used
    key = id(weight)
    cached = _TRANSPOSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    nbytes = weight.numel() * weight.element_size()
    if nbytes > _MAX_CACHED_WEIGHT_BYTES:
        return None
    if _cache_bytes_used + nbytes > _MAX_TOTAL_CACHE_BYTES:
        return None
    cached = weight.t().contiguous()
    _TRANSPOSED_WEIGHT_CACHE[key] = cached
    _cache_bytes_used += nbytes
    return cached


def gfx803_prefill_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """C = x @ weight.T, x: [M, K] fp16, weight: [N, K] fp16 (vLLM's
    standard row-major layout). Returns [M, N] fp16, or None if `weight`
    is too large to cache a transposed copy of (see
    `_MAX_CACHED_WEIGHT_BYTES`) -- callers must fall back to their normal
    GEMM path in that case.
    """
    M, K = x.shape
    N = weight.shape[0]
    b = _transposed_weight(weight)
    if b is None:
        return None
    out = torch.empty(M, N, dtype=torch.float16, device=x.device)
    lib = _lib()
    lib.gfx803_gemm_launch(
        x.data_ptr(), b.data_ptr(), out.data_ptr(),
        M, N, K,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    return out
