import os, torch

dev = "cuda"
TRIALS = int(os.environ.get("TRIALS", "150"))
N = 1 << 20
side = torch.cuda.Stream()

buf = torch.zeros(N, device=dev)
out = torch.zeros(N, device=dev)
host = torch.zeros(N, dtype=torch.float32)

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out.copy_(buf)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    # The graph's only reader. Its dispatch goes out through the batch packet path and
    # therefore carries no ACQUIRE_MEM of its own.
    out.copy_(buf)

anom = {"sdma_then_graph": 0, "other_stream_then_graph": 0}
for t in range(TRIALS):
    v = float(t + 1)

    # (a) copy engine writes the buffer the graph then reads
    host.fill_(v)
    buf.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    got = float(out[0].item()), float(out[-1].item())
    if got != (v, v):
        anom["sdma_then_graph"] += 1

    # (b) a kernel on another queue writes it, then the graph reads it after the wait
    with torch.cuda.stream(side):
        buf.fill_(v)
        e = torch.cuda.Event()
        e.record(side)
    torch.cuda.current_stream().wait_event(e)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    got = float(out[0].item()), float(out[-1].item())
    if got != (v, v):
        anom["other_stream_then_graph"] += 1

print(f"GRAPHPROBE TRIALS={TRIALS} checks={2*TRIALS} anomalies={sum(anom.values())} {anom}", flush=True)
