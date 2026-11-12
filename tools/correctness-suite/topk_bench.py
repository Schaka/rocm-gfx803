#!/usr/bin/env python3
"""Compare native gfx803 torch.topk performance and correctness per process.

Requested cases are from the review of Schaka PR #8. Run on BOTH the
unpatched Schaka Torch and the patched Torch, with identical ROCm inputs.
One CUDA/HIP context per case; a signal/timeout halts by default because a
process crash can leave the card unhealthy. Never infer correctness from timing.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

CASES = (
    ((2, 4), 2, "float32"),
    ((2048, 64), 8, "float32"),
    ((2048, 128), 8, "float16"),
    ((1, 151936), 50, "float32"),
    ((32, 151936), 50, "float32"),
    ((32, 151936), 50, "float16"),
    ((1000000,), 10, "float32"),
    ((1000000,), 1000, "float32"),
    ((16000000,), 100, "float32"),
)
FIELDS=("case","shape","k","dtype","status","median_ms","min_ms",
        "values_ok","indices_ok","torch","hip","arch","error")

def base(index):
    shape,k,dtype=CASES[index]
    return {"case":index,"shape":"x".join(str(x) for x in shape),
            "k":k,"dtype":dtype}

def one(index,warmup,iterations):
    import torch
    shape,k,dtype=CASES[index]
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU unavailable")
    arch=str(torch.cuda.get_device_properties(0).gcnArchName)
    if not arch.startswith("gfx803"):
        raise RuntimeError("gfx803 hardware required: "+arch)
    torch.manual_seed(0)
    cpu=torch.randn(shape,dtype=getattr(torch,dtype))
    gpu=cpu.cuda()
    values,indices=torch.topk(gpu,k,dim=-1)
    torch.cuda.synchronize()
    values,indices=values.cpu(),indices.cpu()
    expected=torch.topk(cpu.float(),k,dim=-1).values
    # Sort value multisets to avoid treating unspecified tie order as failure.
    values_ok=bool(torch.equal(torch.sort(values.float(),dim=-1).values,
                               torch.sort(expected,dim=-1).values))
    gathered_ok=bool(torch.equal(torch.gather(cpu,-1,indices),values))
    distinct_ok=bool(indices.sort(dim=-1).values.diff(dim=-1).ne(0).all())
    for _ in range(warmup):
        torch.topk(gpu,k,dim=-1)
    torch.cuda.synchronize()
    times=[]
    for _ in range(iterations):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        torch.topk(gpu,k,dim=-1)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"median_ms":round(times[len(times)//2],4),
            "min_ms":round(times[0],4),"values_ok":values_ok,
            "indices_ok":gathered_ok and distinct_ok,
            "torch":torch.__version__,"hip":torch.version.hip,"arch":arch}

def write_rows(path,rows):
    if path.exists():
        raise ValueError("refuse overwriting an existing benchmark CSV")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,restval="")
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--iterations",type=int,default=50)
    parser.add_argument("--timeout",type=int,default=300)
    parser.add_argument("--case",type=int,choices=range(len(CASES)))
    parser.add_argument("--continue-after-fault",action="store_true",
                        help="dangerous: only after manually confirming GPU recovery")
    args=parser.parse_args()
    if args.warmup<0 or args.iterations<1 or args.timeout<1:
        parser.error("warmup must be nonnegative and iterations/timeout positive")
    if args.case is not None:
        print(json.dumps(one(args.case,args.warmup,args.iterations)),flush=True)
        return 0
    if args.out is None:parser.error("--out required")
    if args.out.exists():parser.error("refuse overwriting a prior CSV")
    rows=[]
    # Run long-row benchmarks before the already-known short-row VM fault.\n    # The original input still runs, but if it faults we retain useful prior\n    # measurements and halt without testing on a potentially wedged device.\n    for i in list(range(1,len(CASES)))+[0]:\n        row=base(i)
        cmd=[sys.executable,str(Path(__file__).resolve()),"--case",str(i),
             "--warmup",str(args.warmup),"--iterations",str(args.iterations)]
        try:
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout)
            if proc.returncode==0:
                try:
                    receipt=json.loads(proc.stdout.strip().splitlines()[-1])
                    row.update(receipt)
                    row["status"]="ok" if row["values_ok"] and row["indices_ok"] else "wrong result"
                except (ValueError,IndexError,KeyError):
                    row.update(status="error",error="invalid successful child receipt")
            else:
                row.update(status=f"crash(signal {-proc.returncode})"
                           if proc.returncode<0 else f"error(exit {proc.returncode})",
                           error=(proc.stderr.strip().splitlines() or [""])[-1][:240])
        except subprocess.TimeoutExpired:
            row.update(status="timeout",error="child exceeded timeout")
        rows.append(row)
        # Durable CSV after every child; never fabricate unexecuted rows.
        partial=args.out.with_suffix(args.out.suffix+".partial")
        with partial.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial,args.out)
        print(f"case {i} {row['shape']} k={row['k']} {row['dtype']}: {row['status']}",flush=True)
        if (row["status"].startswith("crash") or row["status"]=="timeout") and not args.continue_after_fault:
            print("HALT: GPU health inspection/recovery required before another case",file=sys.stderr)
            return 2
    return 0 if all(row["status"]=="ok" for row in rows) else 1

if __name__=="__main__":
    raise SystemExit(main())
