import os, torch
import torch.nn.functional as F

dev = "cuda"
ALL = [(3,64,64),(64,64,64),(64,128,32),(128,128,32),(128,256,16),(320,320,16),(320,640,16),(640,640,8)]
TRIALS = int(os.environ.get("TRIALS", "30"))
SYNC = os.environ.get("SYNC", "0") == "1"
side = torch.cuda.Stream()
torch.manual_seed(7)

kinds = {"bad_gpu": 0, "bad_copy": 0, "copy_race": 0, "flush_too_early": 0, "other": 0}
anom = 0
detail = []
for t in range(TRIALS):
    for (Cin, Cout, H) in ALL:
        x = torch.randn(1, Cin, H, H)
        w = torch.randn(Cout, Cin, 3, 3)
        b = torch.randn(Cout)
        ref = F.conv2d(x.double(), w.double(), b.double(), padding=1)
        with torch.cuda.stream(side):
            xd, wd, bd = x.to(dev), w.to(dev), b.to(dev)
            y = F.conv2d(xd, wd, bd, padding=1)
            ev = torch.cuda.Event()
            ev.record(side)
        torch.cuda.current_stream().wait_event(ev)
        if SYNC:
            torch.cuda.synchronize()
        h1 = y.cpu().double()                      # the copy under test
        yc = y.clone()                             # kernel reader, no sync in between
        torch.cuda.synchronize()
        h4 = yc.cpu().double()                     # what a kernel saw of y
        h2 = y.cpu().double()                      # re-copy the same buffer
        y3 = F.conv2d(xd, wd, bd, padding=1)       # recompute on current stream
        torch.cuda.synchronize()
        h3 = y3.cpu().double()
        e1 = float((h1 - ref).abs().max())
        if e1 > 0.05:
            anom += 1
            e2 = float((h2 - ref).abs().max())
            e4 = float((h4 - ref).abs().max())
            e3 = float((h3 - ref).abs().max())
            d12 = float((h1 - h2).abs().max())
            if e3 > 0.05:
                k = "bad_gpu"          # side-stream conv produced a wrong result
            elif e4 <= 0.05:
                k = "flush_too_early"  # a kernel still read the right bytes
            elif e2 <= 0.05 and d12 > 0.05:
                k = "copy_race"        # buffer is right, first copy read something else
            elif e2 > 0.05:
                k = "bad_copy"         # both host copies wrong, recompute right
            else:
                k = "other"
            kinds[k] += 1
            if len(detail) < 6:
                detail.append(f"t={t} {Cin}x{Cout}x{H} e1={e1:.3g} e2={e2:.3g} e3={e3:.3g} e4={e4:.3g} d12={d12:.3g} {k}")

print(f"MS3 SYNC={int(SYNC)} BLIT={os.environ.get('GPU_FORCE_BLIT_COPY_SIZE','-')} checks={TRIALS*len(ALL)} "
      f"anomalies={anom} kinds={kinds}", flush=True)
for d in detail:
    print("   ", d, flush=True)
