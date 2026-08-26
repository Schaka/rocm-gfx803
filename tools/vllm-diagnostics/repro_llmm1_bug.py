"""Minimal standalone repro for a suspected gfx803-specific ROCm/HIP bug in
vLLM's LLMM1 kernel (csrc/rocm/skinny_gemms.cu, LLGemm1_kernel).

No vLLM engine, no model, no scheduler -- just the compiled op and a
correctness check against torch.nn.functional.linear. Needs only:
  - vLLM installed (for `vllm._custom_ops.LLMM1`, which calls straight into
    the compiled `_rocm_C.abi3.so` extension)
  - a gfx803 (or other ROCm) card

Usage: python3 repro_llmm1_bug.py
Confirmed on gfx803 (2026-08-25): K=1536 correct; K=8064 and K=8192 both
launch cleanly but silently return wrong values (max_abs_diff ~2.1-2.4,
catastrophic, not fp16 noise); K=8960 (down_proj's real shape in
Qwen2.5-1.5B) doesn't even launch -- `hipErrorInvalidConfiguration`.

LIKELY root cause (strong circumstantial evidence, not yet confirmed by
disassembly/profiling -- see SESSION_HANDOFF.md and gfx803_gemv.py's
module docstring in this repo for the fuller writeup): skinny_gemms.cu's
LLMM1() host launcher computes
    NUM_THREADS = ceil_to_multiple_of(WARP_SIZE, K * 2 / 16)
i.e. essentially ceil_to_64(K/8) threads per block, with no upper-bound
check against the hardware's max threads/block (1024 on gfx803, and
typically on other ROCm/CUDA hardware too). The three observed outcomes
line up with this exactly:
  - K=1536 -> NUM_THREADS=384: well under the limit, correct.
  - K=8064 -> NUM_THREADS=1024 (1008 rounds up to 1024): *at* the limit --
    silently wrong.
  - K=8192 -> NUM_THREADS=1024 (already a multiple of 64): *at* the
    limit -- silently wrong.
  - K=8960 -> NUM_THREADS=1152 (1120 rounds up to 1152): *past* the
    limit -- outright rejected by the driver.
  Landing exactly at 1024 doesn't itself explain wrongness (1024 should
  be a legal thread count) -- the leading theory is that the kernel's
  per-thread register footprint at that width causes register spilling
  or an occupancy/scheduling problem the compiler doesn't handle
  correctly on this backend, and/or that LLGemm1_kernel's second-stage
  warp-shuffle reduction (hardcodes an assumption of at most 32 warps
  per block: `for mask in 16,8,4,2,1: ...` then one
  `__shfl_xor(_, 16)`, a pattern that reads as ported from 32-lane-warp
  CUDA without adjustment for AMD's 64-lane wavefronts) breaks down at
  NUM_THREADS/64=16 warps for reasons not yet traced. To chase this down
  further:
  - Dump the compiled kernel's actual thread/register/LDS usage at K=8064
    vs K=1536 (`rocprofv3`/`rocminfo`, or hipcc `--save-temps` + inspect
    the .s/.hsaco) to check whether register spilling or an occupancy
    collapse is actually happening at NUM_THREADS=1024.
  - Binary-search the exact K threshold between "correct" and "wrong"
    (this script only checks 1536/8064/8192; a tighter sweep, e.g. every
    64 between 4096 and 8064, would show whether the break is sharp at
    the 1024-thread boundary or gradual -- e.g. does K=7936 (960 threads,
    still under the limit) also fail, or is it clean right up to 1024?).
  - Compare against a modern (gfx90a/gfx942/gfx11x) card with the same
    vLLM build -- if it reproduces there too, it's a generic/portability
    bug worth reporting upstream to vLLM regardless of gfx803's
    unsupported status here; if it's gfx803-only, it stays this repo's to
    fix (matches AGENTS.md's differential-testing guidance). Per this
    repo's standing scope, gfx803 is unsupported upstream since ROCm 6.0
    and nothing here gets filed/upstreamed -- any fix stays local
    (already shipped: gfx803_gemv.py's `gfx803_triton_gemv` Triton
    kernel, which routes around this bug entirely for K>4096 rather than
    fixing skinny_gemms.cu in place). This script exists so the
    investigation can be picked back up
    without re-deriving the repro from scratch, not as a precursor to an
    upstream report.
"""

import torch

from vllm import _custom_ops as ops

torch.manual_seed(0)


def check(M: int, K: int, rows_per_block: int = 2) -> None:
    weight = (torch.randn(M, K, dtype=torch.float32) * 0.02).half().cuda()
    x = (torch.randn(K, dtype=torch.float32) * 0.5).half().cuda()

    ref = torch.nn.functional.linear(x.float(), weight.float())
    try:
        out = ops.LLMM1(weight, x.unsqueeze(0), rows_per_block).view(M)
        # LLMM1 launches async; an invalid launch config doesn't raise here
        # or even at torch.cuda.synchronize() -- it only actually surfaces
        # on the *next* real device-touching op, which is why the whole
        # correctness check (not just the launch) has to be inside this
        # try block.
        diff = (out.float() - ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
    except Exception as e:
        print(f"M={M} K={K} rows_per_block={rows_per_block}: LAUNCH FAILED -- {type(e).__name__}: {e}".splitlines()[0])
        return

    status = "OK" if max_diff < 1e-2 else "WRONG"
    print(
        f"M={M} K={K} rows_per_block={rows_per_block}: "
        f"max_abs_diff={max_diff:.4e} mean_abs_diff={mean_diff:.4e}  [{status}]"
    )


if __name__ == "__main__":
    check(M=1536, K=1536)   # expected: OK
    check(M=1536, K=8064)   # expected: WRONG (silent miscompute, NUM_THREADS==1024)
    check(M=1536, K=8192)   # expected: WRONG (silent miscompute, NUM_THREADS==1024)
    check(M=1536, K=8960)   # down_proj's real shape -- expected: LAUNCH FAILED (NUM_THREADS==1152, hipErrorInvalidConfiguration)
