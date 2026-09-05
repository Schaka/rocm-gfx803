#!/usr/bin/env python3
"""gfx803 cross-dispatch read-staleness probe.

On CIK (gfx7/gfx8) the compute shader's TC (L1) is not dropped at a dispatch
boundary, and the AQL SCACQUIRE/SCRELEASE fence-scope bits that ROCm uses as its
only expression of cross-dispatch coherence do not drop it either. A kernel that
reads a virtual address whose contents an earlier kernel overwrote can be served
the pre-overwrite bytes, silently, with hipSuccess -- while the correct bytes are
already in VRAM. Any workload that recycles virtual addresses eats it (torch's
caching allocator is the reproducible case).

The probe does both observations that pinned the bug down:
  * `zz` -- a GPU->GPU copy of the GEMM output, read back once. A copy issued
    while the reader's TC lines are stale freezes the wrong bytes permanently,
    so this catches staleness that a later re-read would hide.
  * two back-to-back host reads of the GEMM output -- the first can be stale
    while the second (different CUs, or evicted lines) is correct.

Exit code 0 = clean, 1 = poisoned. Run it twice with CLR_GFX8_TC_INVALIDATE=0 to
confirm the bug is present, then with =1 (the default once
patches/rocm-systems/gfx803-tc-invalidate-acquire-mem.patch is in the runtime) to
confirm the fix.
"""

import argparse
import os

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--shape", type=int, nargs=3, default=[2, 128, 64],
                    help="bmm batch/m/k; n is taken equal to m")
    ap.add_argument("--tol", type=float, default=0.25)
    args = ap.parse_args()

    bsz, m, k = args.shape
    n = m
    dev = "cuda"
    torch.manual_seed(1234)

    a_cpu = torch.randn(bsz, m, k)
    b_cpu = torch.randn(bsz, k, n)
    ref = (a_cpu @ b_cpu).half().float().numpy()
    a = a_cpu.to(dev, dtype=torch.float16)
    b = b_cpu.to(dev, dtype=torch.float16)

    frozen_bad = 0
    first_read_bad = 0
    flip_trials = 0
    poisoned_trials = 0
    vas = set()

    for _ in range(args.trials):
        with torch.no_grad():
            # Churn allocations so the caching allocator hands back recycled VAs
            # -- this is the condition the bug needs, not the GEMM itself.
            x1 = torch.randn(bsz, k, k, device=dev, dtype=torch.float16)
            x2 = torch.randn(bsz, k, k, device=dev, dtype=torch.float16)
            torch.bmm(x1, x2)
            x3 = torch.randn(bsz, m, k, device=dev, dtype=torch.float32)
            x4 = torch.randn(bsz, k, m, device=dev, dtype=torch.float32)
            torch.bmm(x3, x4)
            y = torch.bmm(a, b)
        torch.cuda.synchronize()
        vas.add(int(y.data_ptr()))

        with torch.no_grad():
            z = torch.empty_like(y)
            z.copy_(y)
        torch.cuda.synchronize()

        hz = z.cpu().float().numpy()
        zf = int(((~np.isfinite(hz)) | (np.abs(hz - ref) > args.tol)).sum())
        frozen_bad += zf

        r1 = y.cpu().float().numpy()
        r2 = y.cpu().float().numpy()
        n1 = int(((~np.isfinite(r1)) | (np.abs(r1 - ref) > args.tol)).sum())
        n2 = int(((~np.isfinite(r2)) | (np.abs(r2 - ref) > args.tol)).sum())
        first_read_bad += n1
        if not np.array_equal(r1, r2):
            flip_trials += 1
        if zf or n1 or n2:
            poisoned_trials += 1

    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)} "
          f"({bsz}x{m}x{k}x{n} bmm, {args.trials} trials, "
          f"CLR_GFX8_TC_INVALIDATE={os.environ.get('CLR_GFX8_TC_INVALIDATE', 'default(on for gfx8)')})")
    print(f"  distinct output VAs (allocator recycling): {len(vas)}")
    print(f"  stale bytes frozen into the GPU->GPU copy: {frozen_bad}")
    print(f"  stale bytes in the first host read:        {first_read_bad}")
    print(f"  trials where two back-to-back reads differ: {flip_trials}")
    print(f"  poisoned trials: {poisoned_trials}/{args.trials}")
    ok = poisoned_trials == 0
    print("RESULT:", "CLEAN" if ok else "STALE READS PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
