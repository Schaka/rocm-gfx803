import os, torch
import torch.nn.functional as F

dev = "cuda"
ALL = [(3, 64, 64), (64, 64, 64), (64, 128, 32), (128, 128, 32), (128, 256, 16),
       (320, 320, 16), (320, 640, 16), (640, 640, 8)]
TRIALS = int(os.environ.get("TRIALS", "30"))
side1 = torch.cuda.Stream()
side2 = torch.cuda.Stream()
torch.manual_seed(11)

anom = 0
kinds = {"pinned_dest": 0, "fan_in": 0}
detail = []
for t in range(TRIALS):
    for (Cin, Cout, H) in ALL:
        x = torch.randn(1, Cin, H, H)
        w = torch.randn(Cout, Cin, 3, 3)
        b = torch.randn(Cout)
        ref = F.conv2d(x.double(), w.double(), b.double(), padding=1)
        n = ref.numel()

        # (a) one producer, consumer copies into PINNED host memory
        with torch.cuda.stream(side1):
            xd, wd, bd = x.to(dev), w.to(dev), b.to(dev)
            y = F.conv2d(xd, wd, bd, padding=1)
            e1 = torch.cuda.Event()
            e1.record(side1)
        cur = torch.cuda.current_stream()
        cur.wait_event(e1)
        out = torch.empty(y.shape, dtype=y.dtype, pin_memory=True)
        out.copy_(y, non_blocking=True)
        torch.cuda.synchronize()
        err = float((out.double().reshape(ref.shape) - ref).abs().max())
        if err > 0.05:
            anom += 1
            kinds["pinned_dest"] += 1
            if len(detail) < 8:
                detail.append(f"t={t} PIN {Cin}x{Cout}x{H} err={err:.3g}")

        # (b) two producers converge on the consumer's single copy
        with torch.cuda.stream(side1):
            p = F.conv2d(xd, wd, bd, padding=1)
            ea = torch.cuda.Event()
            ea.record(side1)
        with torch.cuda.stream(side2):
            q = F.conv2d(xd, wd, bd, padding=1)
            eb = torch.cuda.Event()
            eb.record(side2)
        cur.wait_event(ea)
        cur.wait_event(eb)
        s = (p + q).cpu().double()
        expect = (ref + ref)
        err2 = float((s.reshape(expect.shape) - expect).abs().max())
        if err2 > 0.1:
            anom += 1
            kinds["fan_in"] += 1
            if len(detail) < 8:
                detail.append(f"t={t} FAN {Cin}x{Cout}x{H} err={err2:.3g}")

print(f"MS4 TRIALS={TRIALS} checks={2*TRIALS*len(ALL)} anomalies={anom} kinds={kinds}", flush=True)
for d in detail:
    print("   ", d, flush=True)
