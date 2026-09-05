# Same as multistream.py, plus SYNC=1 which fully quiesces the GPU between the
# side-stream producer and the copy. If SYNC=1 is clean while SYNC=0 leaks, the
# residual is the cross-stream wait path, not the placement of the flush.
import os, torch
import torch.nn.functional as F
dev = "cuda"
ALL = [(3,64,64),(64,64,64),(64,128,32),(128,128,32),(128,256,16),(320,320,16),(320,640,16),(640,640,8)]
TRIALS = int(os.environ.get("TRIALS", "30"))
SYNC = os.environ.get("SYNC", "0") == "1"
side = torch.cuda.Stream()
torch.manual_seed(7)
anom = 0
for t in range(TRIALS):
    for i, (Cin, Cout, H) in enumerate(ALL):
        x = torch.randn(1,Cin,H,H); w = torch.randn(Cout,Cin,3,3); b = torch.randn(Cout)
        ref = F.conv2d(x.double(), w.double(), b.double(), padding=1)
        with torch.cuda.stream(side):
            y = F.conv2d(x.to(dev), w.to(dev), b.to(dev), padding=1)
            ev = torch.cuda.Event(); ev.record(side)
        torch.cuda.current_stream().wait_event(ev)
        if SYNC:
            torch.cuda.synchronize()
        err = float((y.cpu().double() - ref).abs().max())
        if err > 0.05:
            anom += 1
print(f"MS2 SYNC={int(SYNC)} TRIALS={TRIALS} checks={TRIALS*len(ALL)} anomalies={anom}", flush=True)
