# For each anomaly, separate the three candidate fault sites:
#  inputs on device != inputs on host  -> input buffer corruption
#  GPU-side output error != 0          -> conv computed/consumed wrong data
#  GPU-side clean but host copy wrong   -> the D2H copy read stale memory
# Also: are the bad elements contiguous 256B-aligned runs (cache-line signature)?
import os, torch
import torch.nn.functional as F
dev = "cuda"
ALL = [(3,64,64),(64,64,64),(64,128,32),(128,128,32),(128,256,16),(320,320,16),(320,640,16),(640,640,8)]
TRIALS = int(os.environ.get("TRIALS", "30"))
torch.manual_seed(7)
stats = dict(anom=0, bad_input=0, bad_gpu=0, copy_only=0, runs=0, scattered=0)
for t in range(TRIALS):
    for i, (Cin, Cout, H) in enumerate(ALL):
        x = torch.randn(1,Cin,H,H); w = torch.randn(Cout,Cin,3,3); b = torch.randn(Cout)
        ref = F.conv2d(x.double(), w.double(), b.double(), padding=1)
        xg, wg, bg = x.to(dev), w.to(dev), b.to(dev)
        y = F.conv2d(xg, wg, bg, padding=1)
        yc = y.cpu()
        err_host = float((yc.double() - ref).abs().max())
        if err_host <= 0.05:
            continue
        stats["anom"] += 1
        in_ok = bool(torch.equal(xg, x.to(dev)) and torch.equal(wg, w.to(dev)) and torch.equal(bg, b.to(dev)))
        err_gpu = float((y.to(torch.float64) - ref.to(dev)).abs().max())
        if not in_ok: stats["bad_input"] += 1
        if err_gpu > 0.05: stats["bad_gpu"] += 1
        else: stats["copy_only"] += 1
        bad = ((yc.double() - ref).abs() > 0.05).flatten().nonzero().flatten().tolist()
        if bad:
            # contiguous 64-element (256B) aligned runs
            blocks = sorted(set(e // 64 for e in bad))
            contig = all(blocks[k+1] - blocks[k] == 1 for k in range(len(blocks)-1)) and len(blocks) > 1
            stats["runs" if contig else "scattered"] += 1
            span = f"bad={len(bad)} elems first={bad[0]} last={bad[-1]} blocks={len(blocks)} contig_runs={contig}"
        else:
            span = "bad=0"
        print(f"[anom] trial={t} idx={i} C{Cin}->{Cout} {H}x{H} err_host={err_host:.4f} "
              f"err_gpu={err_gpu:.4f} inputs_intact={in_ok} {span}", flush=True)
print("STATS", stats, flush=True)
