#!/usr/bin/env python3
"""Standalone sweep of the hand-written gfx803 GEMM (libgfx803gemm.so).

Loads the kernel the way `vllm/model_executor/layers/gfx803_prefill_gemm.py`
does, but with no vLLM import, so it can measure the kernel's behaviour across
the whole M range -- decode-sized (M=1..16), chunked-prefill-sized, and
prefill-sized -- against the card's measured copy bandwidth and against
rocBLAS's own path at the same shapes.

The interesting question is small M: the weight traffic per call is the same
for every M up to the tile height, so effective GB/s at M=8 says whether
batching amortises the weight read or not.
"""

import argparse
import ctypes
import statistics

import torch

LIB = "/data/vllm-mobydick/vllm/model_executor/layers/libgfx803gemm.so"


def bench(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


@torch.no_grad
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-list", default="1,2,4,8,16,32,64,128,256,512,1024")
    ap.add_argument(
        "--shapes",
        default="o_proj:1024:1024,qkv_proj:4096:1024,gate_up_proj:6144:1024,"
        "down_proj:3072:1024,lm_head:151936:1024",
        help="name:N:K triples (N = out_features, K = in_features)",
    )
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    n_bytes = 256 * 1024 * 1024
    src = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    bw = 2 * n_bytes / (bench(lambda: dst.copy_(src)) / 1000.0) / 1e9
    print(f"# copy bandwidth: {bw:.1f} GB/s")
    del src, dst
    torch.cuda.empty_cache()

    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    lib.gfx803_gemm_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    for spec in args.shapes.split(","):
        name, n_s, k_s = spec.split(":")
        n, k = int(n_s), int(k_s)
        b = (torch.randn(k, n, device="cuda", dtype=torch.float16) * 0.02)
        w = b.t().contiguous()  # vLLM layout, for the rocBLAS comparison
        ref_b = b.float()
        wbytes = n * k * 2
        for m in [int(v) for v in args.m_list.split(",")]:
            a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.02
            out = torch.empty(m, n, dtype=torch.float16, device="cuda")
            ref = a.float() @ ref_b
            scale = max(ref.abs().max().item(), 1e-9)

            def run():
                lib.gfx803_gemm_launch(
                    a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, stream
                )

            run()
            torch.cuda.synchronize()
            err = (out.float() - ref).abs().max().item() / scale
            ms = bench(run)
            flops = 2.0 * m * n * k
            line = (
                f"{name:14s} M={m:5d} | hipgemm {ms:9.4f}ms "
                f"{wbytes / (ms / 1000) / 1e9:6.1f}GB/s "
                f"{flops / (ms / 1000) / 1e12:5.2f}TF "
                f"relerr={err:.1e}"
            )
            if not args.skip_torch:
                tms = bench(lambda: torch.nn.functional.linear(a, w))
                line += f" || F.linear {tms:9.4f}ms {wbytes / (tms / 1000) / 1e9:6.1f}GB/s"
            print(line, flush=True)
            del a, out, ref
        del b, w, ref_b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
