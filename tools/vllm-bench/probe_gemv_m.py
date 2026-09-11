#!/usr/bin/env python3
"""Correctness and throughput of the gfx803 multi-token GEMV, per shape.

The reference is a float32 matmul, not another fp16 path, so the error column
says what the kernel actually computes rather than what another reduced
precision kernel happens to compute. The output is fp16, so a relative error
near 1e-3 is the output rounding, and an accumulator that is genuinely losing
precision shows a figure well above that and growing with K.

GB/s counts the weight bytes each call must read (N*K*2). That is the
quantity that decides decode speed: the activation for a whole batch is a
few kilobytes and stays cache-resident, so the weight stream is the traffic.
The copy bandwidth is printed first as the card's own reference point.

The batch list defaults to every size the kernel accepts, not the powers of
two. The dispatcher instantiates the next power of two at or above M, so a
batch of 3 runs the MTOK=4 kernel with one slot to mask off, and a kernel
that fails to mask that slot reads and writes past the end of x and c
instead of reporting anything: it aborts on the gate, and on hardware it can
corrupt the allocation that follows. Sampling only M=2/4/8/16 misses every
case that masking exists for.

Run with the box's gfx803 environment sourced (env.sh): the float32
reference matmul goes through rocBLAS, whose default gfx803 dispatch is only
correct with the sgemm shim preloaded.
"""

import argparse
import importlib.util
import statistics
import sys

import torch

_LOADER = "/data/vllm-mobydick/vllm/model_executor/layers/gfx803_gemv_m.py"

SHAPES = {
    "qkv_proj": (4096, 1024),
    "o_proj": (1024, 1024),
    "gate_up_proj": (6144, 1024),
    "down_proj": (3072, 1024),
    "lm_head": (151936, 1024),
}


def load_gemv_m():
    spec = importlib.util.spec_from_file_location("gfx803_gemv_m", _LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-list", default=",".join(str(m) for m in range(2, 17)))
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()

    dev = "cuda"
    gemv_m = load_gemv_m()

    src = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    ms = bench(lambda: dst.copy_(src), iters=10, warmup=3)
    print(f"copy bandwidth: {2 * src.numel() / (ms * 1e-3) / 1e9:.1f} GB/s\n")

    worst_ok = True
    for name in args.shapes.split(","):
        N, K = SHAPES[name]
        weight = (torch.randn(N, K, device=dev) * 0.02).half()
        weight32 = weight.float()
        for m in [int(v) for v in args.m_list.split(",")]:
            if m < 2 or m > 16:
                continue
            x = (torch.randn(m, K, device=dev) * 0.5).half()
            ref = (x.float() @ weight32.T).half()
            # NaN marks a slot the kernel never wrote, which a numerical
            # comparison alone would read as a small error if the previous
            # contents happened to be close.
            out = torch.full((m, N), float("nan"), dtype=torch.float16, device=dev)
            gemv_m.gfx803_gemv_m_into(x, weight, out)
            unwritten = bool(torch.isnan(out.float()).any())
            diff = (out.float() - ref.float()).abs()
            scale = ref.float().abs().mean().clamp_min(1e-6).item()
            relerr = diff.mean().item() / scale
            maxerr = diff.max().item()
            ok = not unwritten and relerr < args.tol
            worst_ok = worst_ok and ok
            ms = bench(lambda: gemv_m.gfx803_gemv_m(x, weight))
            gbps = N * K * 2 / (ms * 1e-3) / 1e9
            print(
                f"{name:13s} M={m:3d} | {ms:8.4f}ms {gbps:6.1f}GB/s "
                f"err={relerr:.1e} max={maxerr:.3f} "
                f"{'FAIL unwritten' if unwritten else ('ok' if ok else 'FAIL')}"
            )
    print("\nall sizes ok" if worst_ok else "\nFAILURES PRESENT")


if __name__ == "__main__":
    sys.exit(main())
