#!/usr/bin/env python3
"""Canary for the one triton code path vLLM cannot run without on gfx803.

    python3 triton_dot_probe.py --dtype fp16

The ROCm attention backend lowers its prefill kernel as an fp16 dot with f32
accumulation. On a target without v_dot, triton must legalize that dot to a
uniform f32 FMA; when it instead emits `llvm.amdgcn.fdot2`, LLVM has no
instruction to select and **aborts the compiler**, which kills this process
before it can print anything. An i8 dot does the same through
`llvm.amdgcn.sdot4`. A caller that sees no output has hit that abort, not a
silent skip, so run one dtype per process and treat missing output as failure.

Nothing here is vLLM- or gfx803-specific in its setup: 16x16 blocked-encoding
dots, GPU result against a float64 CPU reference.
"""

import argparse
import json
import sys

import torch
import triton
import triton.language as tl

DTYPES = {
    "fp16": (torch.float16, torch.float32),
    "bf16": (torch.bfloat16, torch.float32),
    "fp32": (torch.float32, torch.float32),
    "i8": (torch.int8, torch.int32),
}


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], tl.dot(a, b))


def run(dtype):
    in_dt, acc_dt = DTYPES[dtype]
    bm = bn = bk = 16
    if in_dt.is_floating_point:
        a = torch.randn(bm, bk, dtype=in_dt, device="cuda")
        b = torch.randn(bk, bn, dtype=in_dt, device="cuda")
        c = torch.empty(bm, bn, dtype=acc_dt, device="cuda")
        dot_kernel[(1,)](a, b, c, bm, bn, bk)
        torch.cuda.synchronize()
        ref = a.to(torch.float64).cpu() @ b.to(torch.float64).cpu()
        return float((c.cpu().double() - ref).abs().max().item())
    a = torch.randint(-8, 8, (bm, bk), dtype=in_dt, device="cuda")
    b = torch.randint(-8, 8, (bk, bn), dtype=in_dt, device="cuda")
    c = torch.empty(bm, bn, dtype=acc_dt, device="cuda")
    dot_kernel[(1,)](a, b, c, bm, bn, bk)
    torch.cuda.synchronize()
    ref = a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)
    return float((c.cpu().to(torch.int64) - ref).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    args = ap.parse_args()
    record = {"dtype": args.dtype, "triton": triton.__version__}
    try:
        record["max_err"] = run(args.dtype)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        record["ok"] = False
        record["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
    print("TRITONDOT " + json.dumps(record), flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
