NOT an auto-applied patch (vLLM isn't built from source in this repo's
Dockerfile -- box-only editable install, same situation as
`gfx803-skinny-gemv.patch.md`). This file documents the exact change,
verified working on real hardware, to be turned into a real
`git apply`-able patch once the vendored-fork step happens.

WHY -- decode's GEMV path (`gfx803-skinny-gemv.patch.md`) already beats
llama.cpp-Vulkan, but prefill (n>1) was still on rocBLAS/Tensile's fallback
GEMM. Unlike decode's pathological M=1 case, that fallback turns out to be
a legitimately competent tiled GEMM at prefill's real M=128-512 shapes --
both a naive Triton GEMM and a naive hand-written HIP GEMM lost to it by
2-5x (see SESSION_HANDOFF.md sections 27-29 for the full dead-end trail:
git archaeology on gfx803 fp16 HPA, a scoped-but-unbuilt Tensile ASM-
codegen fix in `patches/rocblas/TENSILE_GFX803_FP16_TODO.md`, pipelined
Triton, and the first three revisions of the hand-written kernel).

What actually broke the ceiling: inspecting the *compiled gfx803 ISA*
directly (`clang-offload-bundler -unbundle` to extract the code object,
`llvm-readobj --elf-output-style=GNU -S --notes` to read the AMDGPU kernel
metadata) showed the "obviously safe" 4x4-register-tile kernel was
silently spilling 35 VGPRs/thread to memory every iteration -- the
compiler was capping register allocation at an occupancy-driven 64-VGPR
budget, forcing real register need beyond that to spill. `__launch_bounds__
(256, 1)` (trade occupancy for register budget) eliminates the spill
entirely (`.vgpr_count: 110, .vgpr_spill_count: 0`), and once applied,
double buffering (prefetch the next K-tile while computing the current
one -- a regression *without* the register fix, since that comparison was
confounded by uncontrolled spilling, but a real win once compared on equal
footing) pushes it further. Standalone kernel-vs-`rocblas_gemm_ex`
benchmark, all four of this model's real linear-layer shapes, both M=128
and M=512:

| shape | M=128 | M=512 |
|---|---|---|
| qkv_proj | 1.43x | 1.24x |
| o_proj | 1.18x | 1.26x |
| gate_up_proj | 1.28x | 1.26x |
| down_proj | 1.20x | 1.32x |

Correctness verified against `rocblas_gemm_ex` (independent reference) and
`torch.nn.functional.linear` for every shape, standalone, before wiring in.

WHAT -- `gfx803_gemm_lib.hip` (this directory) compiled once via
`hipcc --offload-arch=gfx803 -O3 -shared -fPIC gfx803_gemm_lib.hip -o
libgfx803gemm.so`, dropped next to `gfx803_prefill_gemm.py` (this
directory) in `vllm/model_executor/layers/`. `gfx803_prefill_gemm.py`
loads it via ctypes (same pattern as `gfx803_gemv.py`'s rocBLAS bridge)
and exposes `gfx803_prefill_gemm(x, weight) -> Tensor | None`.

`rocm_unquantized_gemm_impl()` in `vllm/model_executor/layers/utils.py`
gets a new branch, inserted right after the existing gfx803-decode
(`n==1`) branch and before the `if not on_gfx906():` fallback chain:

```python
    # gfx803, prefill/chunked-prefill (n>1): rocBLAS/Tensile's fallback GEMM
    # turns out to already be a legitimately competent tiled kernel at these
    # M=batch-tokens shapes (unlike decode's M=1 case above) -- confirmed by
    # both a naive Triton GEMM and a naive hand-written HIP GEMM losing to
    # it by 2-5x. What actually beat it: inspecting the *compiled gfx803
    # ISA* directly (clang-offload-bundler -unbundle + llvm-readobj --notes)
    # showed the obvious-looking hand-written kernel was silently spilling
    # 35 VGPRs/thread to memory every iteration -- __launch_bounds__(256, 1)
    # (trade occupancy for register budget) eliminates the spill entirely,
    # and combined with double buffering (a regression *without* the
    # register fix, a further win once it's applied) beats rocblas_gemm_ex
    # by 18-43% at every real shape this model uses. See
    # SESSION_HANDOFF.md sections 27-30 for the full investigation.
    if (
        on_gfx803()
        and n > 1
        and bias is None
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and weight.is_contiguous()
        and x.is_contiguous()
    ):
        from vllm.model_executor.layers.gfx803_prefill_gemm import gfx803_prefill_gemm

        out = gfx803_prefill_gemm(x.reshape(n, k), weight)
        # None: weight too large to cache a transposed copy of (this box's
        # 8GB VRAM can't afford permanently caching gate_up_proj-sized
        # weights across all layers -- see gfx803_prefill_gemm.py's
        # _MAX_CACHED_WEIGHT_BYTES). Fall through to the normal GEMM path.
        if out is not None:
            return out.reshape(*x.shape[:-1], m)

    if not on_gfx906():
```

`bias is None` excludes qkv_proj's prefill path (Qwen2's q/k/v proj has a
bias) -- the kernel doesn't support a fused bias add, so qkv_proj still
falls through to Tensile in real runs despite the standalone kernel
benchmark showing a win for that shape's bare matmul. Not yet addressed.

Two real bugs hit and fixed while wiring this in, both worth keeping in
mind for any similar fast-path addition on this stack:

1. **Unbounded weight cache -> OOM.** `gfx803_prefill_gemm.py` caches one
   transposed-contiguous copy of each weight tensor it touches (the kernel
   needs `[K,N]`, vLLM stores `[N,K]`; transposing per-call would erase the
   win). Caching *every* weight this fast path could reach costs ~2.4GB
   across this 28-layer 1.5B model on an 8GB card -- confirmed via a real
   `Available KV cache memory: -0.4 GiB` failure. Fixed with a
   `_MAX_CACHED_WEIGHT_BYTES = 32MB` cap: `_transposed_weight()` returns
   `None` for anything bigger (gate_up_proj, ~55MB/layer), and the call
   site falls through to Tensile for those. Keeps the win for
   o_proj/down_proj (small enough to cache, ~0.9GB total) while gate_up_proj
   is unaffected either way (bigger shape, smaller relative win, more VRAM
   risk).
2. **`vllm bench latency`'s default CUDA-graph capture set (51 sizes) OOMs
   on an 8GB card** even with the kernel disabled -- unrelated to this
   kernel, a pre-existing property of the bench harness's defaults vs.
   `sanity_gen.py`'s explicit `cudagraph_capture_sizes=[1]`. Reproduce this
   kernel's real number with `--gpu-memory-utilization 0.7 --max-model-len
   1024 --cudagraph-capture-sizes 1` to match `sanity_gen.py`'s production
   settings.

STATUS: correctness verified end-to-end via `sanity_gen.py` (coherent,
correct generation, no hang, no OOM, clean shutdown) with the size-capped
cache and real graphed (`enforce_eager=False`) settings. Real end-to-end
`vllm bench latency` A/B (512-token prefill + 1 decode step, kernel on vs.
off, everything else identical): **0.791-0.798s vs 0.836-0.838s avg
latency, 1.048-1.059x (-4.8% to -5.9%)**. Real but much smaller than the
18-43% standalone GEMM numbers above -- prefill-GEMM time is only part of
the total step (attention, sampling, framework overhead dilute it), and
qkv_proj is currently excluded (bias not supported). Not yet promoted to
an auto-applied Dockerfile patch -- gated on the "vendored vLLM fork"
decision, same as the other `patches/vllm/*.patch.md` files.

**A native-`[N,K]`-layout variant (no transpose, no cache, no VRAM cost,
covers gate_up_proj too) was tried and reverted** -- see
SESSION_HANDOFF.md section 32. It measured *worse* end-to-end (2.8-3.3%
vs this version's 4.8-5.9%) despite looking better on paper (lower total
standalone GEMM time summed across shapes): reading the weight in its
native layout means adjacent threads land on rows far apart in memory
instead of the coalesced access this transposed-cache version gets, and
that real per-call throughput loss outweighs the coverage gain, confirmed
by isolating the two shapes both versions share (o_proj, down_proj) --
the native version is unambiguously slower per call for identical work,
not a launch-count or coverage artifact. This transposed-cache version
remains the shipped one. The VRAM cost this implies (~0.9GB permanently,
confirmed by measurement, not the earlier unbounded ~2.4GB bug) was raised
explicitly and accepted for now -- see SESSION_HANDOFF.md section 32 for
the in-place-transpose alternative considered and deferred.

A second, parallel effort on this same box is fixing Tensile's own gfx803
fp16 HPA codegen gap (`patches/rocblas/TENSILE_GFX803_FP16_TODO.md`).
Once that produces a working kernel, the plan is to benchmark it
per-shape against this one (especially gate_up_proj, which this kernel
still doesn't accelerate) and route each shape to whichever wins, with
Tensile as the fallback -- not to keep blanket-excluding gate_up_proj
indefinitely.
