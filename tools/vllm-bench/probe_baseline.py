#!/usr/bin/env python3
"""Baseline probe: device bandwidth and per-shape GEMM cost, no vLLM import.

Answers three questions that any optimisation plan depends on, before touching
the model: what bandwidth can this card actually reach, what does the current
dispatch (rocBLAS/Tensile, optionally through the sgemm shim's takeover) cost
per shape and batch size, and how far each of those is from the bandwidth
roofline. Runs standalone so it can be used while vLLM itself is being built.
"""

import argparse
import ctypes
import statistics

import torch

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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", default="qwen3-0.6b")
    ap.add_argument("--m-list", default="1,2,4,8,16,32,64,128,512")
    args = ap.parse_args()

    torch.manual_seed(0)

    n_bytes = 256 * 1024 * 1024
    src = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    ms = bench(lambda: dst.copy_(src))
    bw = 2 * n_bytes / (ms / 1000.0) / 1e9
    print(f"DEVICE_COPY_BW_GBPS {bw:.1f}")
    del src, dst
    torch.cuda.empty_cache()

    # rocBLAS's hand-written GEMV, the kernel the decode path falls back to.
    lib = ctypes.CDLL("librocblas.so", mode=ctypes.RTLD_GLOBAL)
    lib.rocblas_hssgemv_strided_batched.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int,
    ]
    lib.rocblas_hssgemv_strided_batched.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    lib.rocblas_create_handle(ctypes.byref(handle))

    def hssgemv(w, x):
        m, k = w.shape
        y = torch.empty(m, dtype=torch.float32, device=w.device)
        lib.rocblas_set_stream(
            handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        )
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        rc = lib.rocblas_hssgemv_strided_batched(
            handle, 112, k, m,
            ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
            w.data_ptr(), k, k * m,
            x.data_ptr(), 1, 0,
            ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
            y.data_ptr(), 1, 0, 1,
        )
        assert rc == 0, rc
        return y

    for name, (n, k) in MODELS[args.model_dim].items():
        w = (torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02)
        ref_w = w.float()
        wbytes = n * k * 2
        for m in [int(v) for v in args.m_list.split(",")]:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.02
            ref = x.float() @ ref_w.t()
            scale = max(ref.abs().max().item(), 1e-9)
            line = f"{name:14s} M={m:4d} w={wbytes / 1e6:6.1f}MB"

            out = torch.nn.functional.linear(x, w)
            err = (out.float() - ref).abs().max().item() / scale
            ms = bench(lambda: torch.nn.functional.linear(x, w))
            line += (
                f" | F.linear {ms:8.3f}ms {wbytes / (ms / 1000) / 1e9:6.1f}GB/s"
                f" relerr={err:.1e}"
            )

            if m == 1:
                y = hssgemv(w, x[0])
                err = (y - ref[0]).abs().max().item() / scale
                ms = bench(lambda: hssgemv(w, x[0]))
                line += (
                    f" | hssgemv {ms:8.3f}ms"
                    f" {wbytes / (ms / 1000) / 1e9:6.1f}GB/s relerr={err:.1e}"
                )
            print(line, flush=True)
        del w, ref_w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
