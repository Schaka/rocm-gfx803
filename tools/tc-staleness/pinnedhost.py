# The remaining handoff: a kernel writes into *pinned host* memory and the CPU
# reads it immediately, with no further GPU work in between. If the GPU's write
# is still sitting in a cache, the CPU sees the old bytes. The marker hook is
# expected to cover this (a stream sync publishes a marker on the writer's ring).
import os, torch
dev = "cuda"
TRIALS = int(os.environ.get("TRIALS", "60"))
N = 1 << 20
gpu = torch.randn(N, device=dev)
pin = torch.empty(N, dtype=torch.float32, pin_memory=True)
bad = 0
for t in range(TRIALS):
    gpu.normal_()
    want = gpu.clone()                       # keep a GPU-side copy of the truth
    pin.copy_(gpu, non_blocking=True)        # H2D-direction: GPU writes host memory
    torch.cuda.current_stream().synchronize()
    got_host = pin.clone()                   # CPU read of the pinned buffer
    ref = want.cpu()
    err = float((got_host - ref).abs().max())
    if err > 0:
        bad += 1
        nz = int((got_host != ref).sum().item())
        print(f"[pin-anom] trial={t} err={err:.6f} elems={nz}", flush=True)
print(f"PINNEDHOST TRIALS={TRIALS} anomalies={bad}", flush=True)
