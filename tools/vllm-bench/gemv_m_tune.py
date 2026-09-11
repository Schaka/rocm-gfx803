#!/usr/bin/env python3
"""Compile variants of the gfx803 multi-token GEMV and compare them.

This drives hipcc over a matrix of -D flags and reports, per variant, the
per-decode-step cost of this model's whole weight set (28 layers of
qkv/o_proj/gate_up/down plus lm_head) at several batch sizes, after a
correctness gate. The per-step figure is the one that matters: a single
shape's time hides which of the 113 weight reads a decode step makes is
paying for the change.

One process per variant, always. The HIP runtime resolves kernels by symbol
name across every module loaded into the process, and these variants differ
only in template arguments, so loading two of them in one process gives
whichever instantiation registered first to both -- which shows up as a
variant reporting another variant's timing.

The correctness gate runs every batch size in [2, 16], not the powers of two:
the dispatcher instantiates the next power of two at or above M, so the sizes
between them exercise slots the kernel has to mask off, and an unmasked slot
writes past the end of the output buffer.

Usage, on the box with its environment sourced:
    source /data/bench/env.sh
    python gemv_m_tune.py --build          # compile the matrix
    for f in */*.so; do python gemv_m_tune.py --so $f; done
"""

import argparse
import collections
import ctypes
import os
import statistics
import subprocess
import sys

import torch

# Name -> the -D matrix that defines it. Rows are the per-shape weights of
# this model: 28 layers of the four attention/MLP projections plus lm_head,
# which is why it carries a multiplier of 28.
SHAPES = {
    "qkv_proj": (4096, 1024, 28),
    "o_proj": (1024, 1024, 28),
    "gate_up_proj": (6144, 1024, 28),
    "down_proj": (3072, 1024, 28),
    "lm_head": (151936, 1024, 1),
}

# (LPR, THREADS, MIN_BLOCKS, UNROLL) per variant. The name is derived from the
# knobs so that a .so file says what built it.
DEFAULT_MATRIX = {
    "p32T256U2": (32, 256, 2, 2),
    "p32T256U1": (32, 256, 2, 1),
    "p16T256U2": (16, 256, 2, 2),
    "p64T256U1": (64, 256, 2, 1),
    "p32T128U2": (32, 128, 4, 2),
}


def flags_for(lpr: int, threads: int, min_blocks: int, unroll: int) -> str:
    return (
        f"-DGEMV_M_LPR={lpr} -DGEMV_M_THREADS={threads} "
        f"-DGEMV_M_MIN_BLOCKS={min_blocks} -DGEMV_M_UNROLL={unroll}"
    )


HIPCC = "/opt/rocm/bin/hipcc"
ARCH = "--offload-arch=gfx803"


def build(src: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for name, knobs in DEFAULT_MATRIX.items():
        dest = os.path.join(outdir, f"{name}.so")
        cmd = [
            HIPCC, ARCH, "-O3", "-shared", "-fPIC",
            *flags_for(*knobs).split(), "-o", dest, src,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        if proc.returncode != 0 or "error" in log.lower():
            print(f"BUILD FAILED {name}\n{log[:2000]}")
        else:
            print(f"built {dest}")


def load(path: str):
    lib = ctypes.CDLL(path)
    fn = getattr(lib, "gfx803_gemv_m_launch")
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    fn.restype = None
    return lib, fn


def call(fn, x, w, out, stream):
    fn(x.data_ptr(), w.data_ptr(), out.data_ptr(),
       x.shape[0], w.shape[0], x.shape[1], ctypes.c_void_p(stream))


def bench(fn, x, w, out, stream, iters):
    for _ in range(3):
        call(fn, x, w, out, stream)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(2):
            call(fn, x, w, out, stream)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 2)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--src", default=None, help="kernel source for --build")
    ap.add_argument("--outdir", default=".", help="where --build puts the .so files")
    ap.add_argument("--so", default=None, help="one variant to measure")
    ap.add_argument("--m-list", default="2,4,8,16")
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()

    if args.build:
        src = args.src or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "vllm", "vllm", "gfx803_kernels", "gfx803_gemv_m.hip",
        )
        build(src, args.outdir)
        return

    if not args.so:
        ap.error("one --so per process: the HIP runtime resolves kernels by "
                 "name across loaded modules, so variants would share one "
                 "instantiation")
    path = os.path.abspath(args.so)
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(0)

    tensors = {}
    for name, (N, K, _) in SHAPES.items():
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        tensors[name] = (w, w.float())

    ok = True
    for m in range(2, 17):
        for name, (N, K, _) in SHAPES.items():
            w, w32 = tensors[name]
            x = (torch.randn(m, K, device="cuda") * 0.5).half()
            ref = (x.float() @ w32.T).half().float()
            lib, fn = load(path)
            out = torch.full((m, N), float("nan"), dtype=torch.float16, device="cuda")
            call(fn, x, w, out, stream)
            torch.cuda.synchronize()
            if bool(torch.isnan(out.float()).any()):
                print(f"WRONG {os.path.basename(path)} {name} M={m} unwritten output")
                ok = False
            else:
                err = ((out.float() - ref).abs().mean()
                       / ref.abs().mean().clamp_min(1e-6)).item()
                if err > args.tol:
                    print(f"WRONG {os.path.basename(path)} {name} M={m} err={err:.2e}")
                    ok = False
            del lib
    print("correctness:", "ok" if ok else "FAILED")
    if not ok:
        return

    for m in [int(v) for v in args.m_list.split(",")]:
        lib, fn = load(path)
        total = 0.0
        detail = []
        for name, (N, K, mult) in SHAPES.items():
            w, _ = tensors[name]
            x = (torch.randn(m, K, device="cuda") * 0.5).half()
            out = torch.empty(m, N, dtype=torch.float16, device="cuda")
            iters = 8 if N > 50000 else 40
            ms = bench(fn, x, w, out, stream, iters)
            total += ms * mult
            detail.append(f"{name.split('_')[0]}={ms:.3f}")
        print(f"M={m:3d} {os.path.basename(path):20s} step={total:7.2f}ms  "
              + " ".join(detail), flush=True)
        del lib


if __name__ == "__main__":
    sys.exit(main())
