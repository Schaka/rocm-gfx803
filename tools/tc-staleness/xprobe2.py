# Cross-queue READER direction, done right this time (the earlier version compared
# pre+a, which is 2t+1, against t+1 -- a test bug, not a hardware finding).
# A kernel on the default ring reads a buffer that a side-ring kernel wrote, after
# the default ring had already cached its previous contents. No copy engine here:
# this isolates the dispatch-side acquire from the pre-copy fence.
import os, torch
dev = "cuda"
TRIALS = int(os.environ.get("TRIALS", "40"))
side = torch.cuda.Stream()
N = 1 << 20
a = torch.full((N,), 5.0, device=dev)
pre = torch.empty(N, device=dev)
bad = 0
for t in range(TRIALS):
    pre.copy_(a)                                  # default ring caches a's lines
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        a.fill_(float(t + 1))                     # side ring rewrites them
        ev = torch.cuda.Event(); ev.record(side)
    torch.cuda.current_stream().wait_event(ev)
    mism = int((a != float(t + 1)).sum().item())  # default ring reads again
    if mism:
        bad += 1
        print(f"[xread-anom] trial={t} mismatches={mism} sample={float(a[0])} want={t+1}", flush=True)
print(f"XREADER TRIALS={TRIALS} anomalies={bad}", flush=True)
