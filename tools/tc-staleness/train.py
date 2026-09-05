#!/usr/bin/env python3
"""Training-loop (backward + Adam) staleness test on gfx803.

Covers the torch surface the forward-only soak misses: gradient kernels read
activations that the forward pass just wrote, into VAs the caching allocator
recycles every step -- the exact stale-TC condition. Three signals:
  * loss trajectory vs a CPU run with identical init/data,
  * bit-reproducibility of the same seed across two GPU runs,
  * a frozen-copy check on the gradients themselves.
"""

import os
import sys

import numpy as np
import torch

dev = "cuda"
steps = int(os.environ.get("STEPS", "60"))
TCINV = os.environ.get("CLR_GFX8_TC_INVALIDATE", "default(on gfx8)")


def make():
    torch.manual_seed(7)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GroupNorm(4, 16), torch.nn.SiLU(),
        torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.GroupNorm(4, 16), torch.nn.SiLU(),
        torch.nn.Conv2d(16, 8, 3, padding=1), torch.nn.SiLU(),
        torch.nn.Flatten(), torch.nn.Linear(8 * 8 * 8, 10), torch.nn.LayerNorm(10),
    )
    xb = torch.randn(4, 3, 8, 8)
    yb = torch.randint(0, 10, (4,))
    return net, xb, yb


def run(device, collect_grads=False):
    net, xb, yb = make()
    net = net.to(device)
    x, y = xb.to(device), yb.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses, gsnaps = [], []
    for s in range(steps):
        opt.zero_grad(set_to_none=False)
        out = net(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        if collect_grads:
            g = torch.nn.utils.parameters_to_vector(
                [p.grad.reshape(-1) for p in net.parameters()])
            z = torch.empty_like(g)
            z.copy_(g)                      # frozen copy right after the writer
            torch.cuda.synchronize()
            gsnaps.append((g.detach().cpu().clone(), z.cpu().clone()))
        losses.append(float(loss.item()))
        opt.step()
    return losses, gsnaps


print(f"### training-loop staleness  TCINV={TCINV}  steps={steps}")
gl, gs = run(dev, collect_grads=True)
cl, _ = run("cpu")

# loss trajectory vs CPU
d = np.abs(np.array(gl) - np.array(cl))
bad_at = [i for i in range(steps) if d[i] > 0.02 + 0.02 * abs(cl[i])]
print(f"loss: gpu[0]={gl[0]:.5f} gpu[-1]={gl[-1]:.5f} cpu[0]={cl[0]:.5f} cpu[-1]={cl[-1]:.5f}")
print(f"      max |loss diff| = {d.max():.5f} at step {int(d.argmax())}; steps over tol: {len(bad_at)}")

# frozen-copy check on gradients, and reproducibility
frozen_bad = 0
for i, (g, z) in enumerate(gs):
    if not torch.equal(g, z):
        frozen_bad += 1
print(f"      grad frozen-copy mismatches: {frozen_bad}/{len(gs)} steps")

gl2, _ = run(dev)
same = gl == gl2
print(f"      bit-identical loss trajectory across two GPU runs: {same}")

fails = []
if bad_at:
    fails.append(f"loss diverges from CPU at {len(bad_at)} steps")
if frozen_bad:
    fails.append(f"{frozen_bad} grad frozen-copy mismatches")
if not same:
    fails.append("loss trajectory not reproducible")
print("TRAIN RESULT:", "CLEAN" if not fails else "PROBLEMS: " + "; ".join(fails))
sys.exit(0 if not fails else 1)
