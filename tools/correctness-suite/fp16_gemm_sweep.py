#!/usr/bin/env python3
"""fp16 GEMM/conv sweep for the gfx803 SGEMM shim.

The shim's fp16 takeover must satisfy the column-major contract its gate asks for,
which means the row-major product it hands the kernel is (n, m, k), not (m, n, k).
Getting that backwards is silent for square problems and wrong for every other
shape, so this sweep deliberately contains both orientations, K values that are not
a multiple of the kernel's 16-wide K tile, the batched form that attention dots use,
and convolutions, whose im2col GEMM is what exposed the bug.

Run it as an A/B across two builds of the shim, one case per process: an out-of-range
read in the fp16 kernel faults the context, so one shared process cannot finish the
grid.

  for i in $(seq 0 26); do podman exec C python3 fp16_gemm_sweep.py --case $i; done

With the (n, m) mapping every case prints ok=true. Expect 0 regressions against an
older shim: the cases an older shim already got right are exactly the square ones.
See patches/rocblas/sgemm-shim/sgemm_shim.cpp.
"""
import argparse, json, sys
import torch

CASES = [
    ("mm", (128, 128, 128)), ("mm", (64, 64, 64)), ("mm", (256, 256, 256)),
    ("mm", (4096, 64, 27)), ("mm", (64, 4096, 27)), ("mm", (2048, 128, 2048)),
    ("mm", (128, 2048, 2048)), ("mm", (256, 64, 512)), ("mm", (64, 512, 256)),
    ("mm", (1024, 64, 27)), ("mm", (512, 2048, 512)), ("mm", (17, 33, 5)),
    ("mm", (257, 65, 129)), ("mm", (2048, 2048, 27)), ("mm", (32, 32, 2048)),
    ("bmm", (8, 128, 128, 128)), ("bmm", (8, 4096, 64, 27)), ("bmm", (4, 64, 4096, 27)),
    ("bmm", (2, 512, 512, 512)), ("bmm", (16, 256, 64, 17)),
    ("conv", (3, 64, 64)), ("conv", (3, 64, 32)), ("conv", (64, 64, 32)),
    ("conv", (320, 320, 16)), ("conv", (3, 320, 16)), ("conv", (64, 3, 32)),
    ("conv", (128, 256, 16)),
]


def run(kind, d, dev):
    torch.manual_seed(101)
    if kind == "mm":
        M, N, K = d
        a = (torch.randn(M, K, device=dev) * 0.2).half()
        b = (torch.randn(K, N, device=dev) * 0.2).half()
        return a @ b, a.float() @ b.float()
    if kind == "bmm":
        B, M, N, K = d
        a = (torch.randn(B, M, K, device=dev) * 0.2).half()
        b = (torch.randn(B, K, N, device=dev) * 0.2).half()
        return torch.bmm(a, b), torch.bmm(a.float(), b.float())
    Cin, Cout, H = d
    x = (torch.randn(1, Cin, H, H, device=dev) * 0.3).half()
    w = (torch.randn(Cout, Cin, 3, 3, device=dev) * 0.3).half()
    bias = (torch.randn(Cout, device=dev) * 0.3).half()
    return (torch.nn.functional.conv2d(x, w, bias, padding=1),
            torch.nn.functional.conv2d(x.float(), w.float(), bias.float(), padding=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=-1, help="run one index; -1 runs all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    picks = range(len(CASES)) if a.case < 0 else [a.case]
    fails = 0
    for i in picks:
        kind, d = CASES[i]
        row = {"i": i, "kind": kind, "dims": list(d)}
        try:
            got, ref = run(kind, d, a.device)
            torch.cuda.synchronize()
            g, r = got.float().double().cpu(), ref.double().cpu()
            row["err"] = float((g - r).abs().max())
            row["scale"] = float(r.abs().max())
            row["ok"] = row["err"] <= 0.02 + 0.02 * row["scale"]
        except Exception as e:
            row["err"] = None
            row["ok"] = False
            row["exc"] = type(e).__name__
        fails += 0 if row["ok"] else 1
        print(json.dumps(row), flush=True)
    print(f"F16SWEEP cases={len(list(picks))} wrong={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
