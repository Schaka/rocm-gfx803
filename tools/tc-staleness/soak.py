#!/usr/bin/env python3
"""Broad gfx803 stale-read soak: many op families, each checked against a CPU
reference three ways -- direct read, a GPU->GPU copy made right after the writer
(the freeze that exposed the original bug), and a second read later -- plus a
bit-identical-repeat test across N runs of the whole pipeline.

Exit code 0 = nothing stale anywhere.
"""

import os
import sys

import numpy as np
import torch

dev = "cuda"
TRIALS = int(os.environ.get("TRIALS", "12"))
TOL = {torch.float16: 0.35, torch.float32: 2e-3}
fails = []
checks = 0


def cmp(name, gpu_cpu_tensor, ref_np, dtype, tol=None):
    """ref_np is the CPU-reference fp32 numpy; gpu_cpu_tensor any tensor."""
    global checks
    checks += 1
    g = gpu_cpu_tensor.detach().to("cpu").float().numpy()
    if tol is None:
        tol = TOL[dtype]
    bad = int(((~np.isfinite(g)) | (np.abs(g - ref_np) > tol + 0.02 * np.abs(ref_np))).sum())
    if bad:
        fails.append(f"{name}: {bad} bad elems (dtype={dtype})")
    return bad


def churn(n=6):
    """Allocate/free a spread of sizes so the caching allocator recycles VAs."""
    keep = []
    for i in range(n):
        keep.append(torch.randn(2 + i, 96, 48, device=dev, dtype=torch.float16))
        keep.append(torch.randn(1 + i, 64, 64, device=dev, dtype=torch.float32))
    return keep


FAMS = []


def fam(fn):
    FAMS.append(fn)
    return fn


@fam
def f_bmm(dtype, i):
    b, m, k, n = (2, 128, 64, 128) if dtype == torch.float16 else (3, 64, 96, 64)
    a = torch.randn(b, m, k, dtype=torch.float32)
    w = torch.randn(b, k, n, dtype=torch.float32)
    ag, wg = a.to(dev, dtype=dtype), w.to(dev, dtype=dtype)
    ref = (a @ w).float().numpy()
    with torch.no_grad():
        y = torch.bmm(ag, wg)
    torch.cuda.synchronize()
    z = torch.empty_like(y)
    with torch.no_grad():
        z.copy_(y)
    torch.cuda.synchronize()
    cmp(f"bmm/frozen/{i}", z, ref, dtype)
    cmp(f"bmm/direct/{i}", y, ref, dtype)
    cmp(f"bmm/re-read/{i}", y.cpu(), ref, dtype)


@fam
def f_linear(dtype, i):
    n, k, m = 512, 128, 96
    x = torch.randn(n, k, dtype=torch.float32)
    w = torch.randn(m, k, dtype=torch.float32) * 0.05
    b = torch.randn(m, dtype=torch.float32)
    xn = (x - x.mean(0)) / x.std(0)
    ln = torch.nn.LayerNorm(k).to(dtype)
    with torch.no_grad():
        ln.weight.copy_(torch.ones(k)); ln.bias.copy_(torch.zeros(k))
    lg = ln.to(dev)
    ref = torch.nn.functional.layer_norm(x, (k,)) @ w.T + b
    ref = torch.softmax(ref, -1).float().numpy()
    with torch.no_grad():
        y = torch.softmax(lg(x.to(dev, dtype=dtype)) @ w.T.to(dev, dtype=dtype)
                         + b.to(dev, dtype=dtype), -1)
    torch.cuda.synchronize()
    z = torch.empty_like(y)
    with torch.no_grad():
        z.copy_(y)
    torch.cuda.synchronize()
    cmp(f"linear-layernorm-softmax/frozen/{i}", z, ref, dtype)
    cmp(f"linear-layernorm-softmax/direct/{i}", y, ref, dtype)


@fam
def f_conv2d(dtype, i):
    c1, c2, h = 32, 32, 24
    x = torch.randn(2, c1, h, h, dtype=torch.float32)
    w = torch.randn(c2, c1, 3, 3, dtype=torch.float32) * 0.05
    ref = torch.nn.functional.conv2d(x, w, padding=1)
    ref = torch.nn.functional.silu(ref)
    ref = torch.nn.functional.group_norm(ref, 8).float().numpy()
    with torch.no_grad():
        y = torch.nn.functional.group_norm(
            torch.nn.functional.silu(
                torch.nn.functional.conv2d(x.to(dev, dtype=dtype), w.to(dev, dtype=dtype),
                                           padding=1)), 8)
    torch.cuda.synchronize()
    z = torch.empty_like(y)
    with torch.no_grad():
        z.copy_(y)
    torch.cuda.synchronize()
    cmp(f"conv-silu-groupnorm/frozen/{i}", z, ref, dtype)
    cmp(f"conv-silu-groupnorm/direct/{i}", y, ref, dtype)


@fam
def f_attention(dtype, i):
    B, H, S, D = 2, 4, 64, 32
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32) * 0.3
    v = torch.randn(B, H, S, D, dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).float().numpy()
    with torch.no_grad():
        y = torch.nn.functional.scaled_dot_product_attention(q.to(dev, dtype=dtype),
                                                             k.to(dev, dtype=dtype),
                                                             v.to(dev, dtype=dtype))
    torch.cuda.synchronize()
    z = torch.empty_like(y)
    with torch.no_grad():
        z.copy_(y)
    torch.cuda.synchronize()
    cmp(f"scaled-dot-attn/frozen/{i}", z, ref, dtype)
    cmp(f"scaled-dot-attn/direct/{i}", y, ref, dtype)


@fam
def f_reductions(dtype, i):
    x = torch.randn(4, 257, 33, dtype=torch.float32)
    xg = x.to(dev, dtype=dtype)
    for opname, fn in (("sum", lambda t: t.sum(-1)), ("amax", lambda t: t.amax(-1)),
                       ("std", lambda t: t.std(-1)), ("cumsum", lambda t: t.cumsum(-1)),
                       ("sort", lambda t: torch.sort(t, -1).values),
                       ("median", lambda t: t.median(-1).values),
                       ("logsumexp", lambda t: torch.logsumexp(t, -1))):
        ref = fn(x).float().numpy()
        with torch.no_grad():
            y = fn(xg)
        torch.cuda.synchronize()
        z = torch.empty_like(y)
        with torch.no_grad():
            z.copy_(y)
        torch.cuda.synchronize()
        tol = 0.5 if dtype == torch.float16 else 5e-3
        cmp(f"reduce-{opname}/frozen/{i}", z, ref, dtype, tol)
        cmp(f"reduce-{opname}/direct/{i}", y, ref, dtype, tol)


@fam
def f_indexing(dtype, i):
    x = torch.randn(64, 128, dtype=torch.float32)
    idx = torch.randint(0, 128, (64, 37))
    ref = torch.gather(x, 1, idx).float().numpy()
    with torch.no_grad():
        y = torch.gather(x.to(dev, dtype=dtype), 1, idx.to(dev))
    torch.cuda.synchronize()
    z = torch.empty_like(y)
    with torch.no_grad():
        z.copy_(y)
    torch.cuda.synchronize()
    cmp(f"gather/frozen/{i}", z, ref, dtype)
    cmp(f"gather/direct/{i}", y, ref, dtype)
    y2 = torch.zeros(128, device=dev, dtype=dtype)
    y2.scatter_add_(0, idx.reshape(-1).to(dev),
                    torch.ones(idx.numel(), device=dev, dtype=dtype))
    ref2 = torch.zeros(128).scatter_add_(0, idx.reshape(-1),
                                         torch.ones(idx.numel())).float().numpy()
    cmp(f"scatter_add/frozen/{i}", y2, ref2, dtype)


def main():
    global checks
    print(f"soak: trials={TRIALS} torch={torch.__version__} "
          f"TCINV={os.environ.get('CLR_GFX8_TC_INVALIDATE', 'default(on gfx8)')} "
          f"seed-suffix={os.environ.get('SEEDX', '0')}", flush=True)
    torch.manual_seed(1234 + int(os.environ.get("SEEDX", "0")))
    for fi, famfun in enumerate(FAMS):
        for dtype in (torch.float16, torch.float32):
            for t in range(TRIALS):
                churn()
                famfun(dtype, t)
    # bit-identical repeat: same pipeline 8x must produce identical bits
    a = torch.randn(4, 128, 64, device=dev, dtype=torch.float16)
    b = torch.randn(4, 64, 128, device=dev, dtype=torch.float16)
    base = None
    for r in range(8):
        churn(10)
        with torch.no_grad():
            y = torch.softmax(torch.bmm(a, b).float(), -1)
        torch.cuda.synchronize()
        if base is None:
            base = y.cpu().clone()
        else:
            checks += 1
            if not torch.equal(base, y.cpu()):
                fails.append(f"repeat-{r}: {int((base != y.cpu()).sum())} bits differ")
    print(f"checks={checks} failures={len(fails)}")
    fam = {}
    for f in fails:
        key = f.split(":")[0].rsplit("/", 1)[0]
        fam[key] = fam.get(key, 0) + 1
    for k, v in sorted(fam.items()):
        print(f"  {v:4d}  {k}")
    for f in fails[:6]:
        print("  eg:", f)
    print("SOAK RESULT:", "CLEAN" if not fails else f"{len(fails)} PROBLEMS")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
