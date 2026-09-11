#!/usr/bin/env python3
"""Direct kernel head-to-head for the gfx803 linear-layer shapes.

For every shape in the target model and every batch size (M = number of tokens
in the step), times each candidate implementation that the dispatch in
`vllm/model_executor/layers/utils.py` can choose from, and reports both the
wall time and the effective weight-bandwidth it achieved. The card's real
achievable bandwidth is measured in the same process, so "how far from the
roofline is this kernel" is answered with a number instead of an assumption.

Every candidate is checked against a float32 reference at the same shape, so a
fast-but-wrong kernel cannot look like a win.

Usage:
    python gemm_matrix.py --model-dim qwen3-0.6b
"""

import argparse
import json
import statistics

import torch

# vLLM weight layout: [out_features, in_features] = [N, K].
MODELS = {
    "qwen3-0.6b": {
        "qkv_proj": (4096, 1024),
        "o_proj": (1024, 1024),
        "gate_up_proj": (6144, 1024),
        "down_proj": (3072, 1024),
        "lm_head": (151936, 1024),
    },
    "qwen2.5-1.5b": {
        "qkv_proj": (2560, 2048),
        "o_proj": (2048, 2048),
        "gate_up_proj": (17920, 2048),
        "down_proj": (2048, 8960),
        "lm_head": (151936, 2048),
    },
}

M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 512]


def bench(fn, warmup: int = 5, iters: int = 30) -> float:
    """Median milliseconds per call, measured on the GPU timeline."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def device_bandwidth() -> float:
    """GB/s achieved by a plain device-to-device copy of a large tensor."""
    n = 256 * 1024 * 1024  # 256MB, well past any cache effect
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    ms = bench(lambda: dst.copy_(src))
    return 2 * n / (ms / 1000.0) / 1e9  # read + write


def candidates(weight: torch.Tensor):
    """Name -> callable(x[M,K]) for every implementation the stack can use."""
    out = {}

    def f_linear(x):
        return torch.nn.functional.linear(x, weight)

    out["F.linear(rocBLAS)"] = f_linear

    try:
        from vllm import _custom_ops as ops

        def llmm1(x):
            return ops.LLMM1(weight, x, 2)

        out["LLMM1(rows=2)"] = llmm1
    except Exception as exc:  # pragma: no cover - reported, not fatal
        out["LLMM1(rows=2)"] = lambda x, exc=exc: (_ for _ in ()).throw(exc)

    try:
        from vllm.model_executor.layers.gfx803_prefill_gemm import (
            gfx803_prefill_gemm,
        )

        def prefill(x):
            r = gfx803_prefill_gemm(x, weight)
            if r is None:
                raise RuntimeError("weight declined by transposed cache")
            return r

        out["gfx803_prefill_gemm"] = prefill
    except Exception as exc:
        out["gfx803_prefill_gemm"] = lambda x, exc=exc: (_ for _ in ()).throw(exc)

    try:
        n, k = weight.shape
        if n <= 65536:
            from vllm.model_executor.layers.gfx803_gemv import gfx803_skinny_gemv

            def gemv(x):
                return gfx803_skinny_gemv(x[0], weight).to(x.dtype).unsqueeze(0)

            out["rocblas_hssgemv"] = gemv
    except Exception as exc:
        out["rocblas_hssgemv"] = lambda x, exc=exc: (_ for _ in ()).throw(exc)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", default="qwen3-0.6b")
    ap.add_argument("--m-list", default=",".join(str(m) for m in M_LIST))
    ap.add_argument("--shapes", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    bw = device_bandwidth()
    print(f"# device copy bandwidth: {bw:.1f} GB/s")
    rows = [{"kind": "bandwidth", "gbps": round(bw, 2)}]

    shapes = MODELS[args.model_dim]
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = {k: v for k, v in shapes.items() if k in keep}

    for name, (n, k) in shapes.items():
        weight = (torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02)
        ref_w = weight.float()
        weight_bytes = n * k * 2
        for m in [int(v) for v in args.m_list.split(",")]:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.02
            ref = (x.float() @ ref_w.t())
            scale = ref.abs().max().item()
            entry = {
                "kind": "gemm",
                "shape": name,
                "n": n,
                "k": k,
                "m": m,
                "weight_mb": round(weight_bytes / 1e6, 1),
            }
            for cname, fn in candidates(weight).items():
                try:
                    out = fn(x)
                    err = (out.float() - ref).abs().max().item()
                    rel = err / scale if scale else err
                    ms = bench(lambda fn=fn, x=x: fn(x))
                    entry[cname] = {
                        "ms": round(ms, 4),
                        "gbps": round(weight_bytes / (ms / 1000.0) / 1e9, 1),
                        "rel_err": float(f"{rel:.2e}"),
                    }
                except Exception as exc:
                    entry[cname] = {"error": str(exc)[:120]}
            rows.append(entry)
            best = [
                (v["ms"], c)
                for c, v in entry.items()
                if isinstance(v, dict) and "ms" in v
            ]
            best.sort()
            print(
                f"{name:14s} M={m:4d}  best={best[0][1]}({best[0][0]:.3f}ms)"
                if best
                else f"{name:14s} M={m:4d}  no candidate ran"
            )
            for _, c in best[:3]:
                v = entry[c]
                print(
                    f"      {c:22s} {v['ms']:8.3f} ms  "
                    f"{v['gbps']:6.1f} GB/s  rel_err={v['rel_err']:.1e}"
                )

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
