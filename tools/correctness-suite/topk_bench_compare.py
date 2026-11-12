#!/usr/bin/env python3
"""Join patched/unpatched TopK benchmark CSVs without obscuring crashes.

Same input case identity required on both arms; report per-case cost relative
to baseline only when both arms have correct results and valid timings.
"""
import argparse
import csv
import json
from pathlib import Path

KEYS=("shape","k","dtype")

def load(path):
    with Path(path).open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError("empty CSV: "+str(path))
    result={}
    for r in rows:
        key=tuple(r[k] for k in KEYS)
        if key in result:raise ValueError("duplicate benchmark case: "+repr(key))
        result[key]=r
    return result

def compare(before,after):
    if set(before)!=set(after):
        raise ValueError("two arms must have identical measured shape/k/dtype cases")
    out=[]
    for key in sorted(before):
        a,b=before[key],after[key]
        row=dict(zip(KEYS,key))
        row.update(unpatched_status=a.get("status"),patched_status=b.get("status"))
        if a.get("status")=="ok" and b.get("status")=="ok":
            x,y=float(a["median_ms"]),float(b["median_ms"])
            if x<=0 or y<=0:raise ValueError("non-positive time")
            row.update(unpatched_median_ms=x,patched_median_ms=y,
                       slowdown_pct=round((y/x-1)*100,2),
                       over_10pct=(y/x>1.10))
        else:
            row["slowdown_pct"]=None
            row["over_10pct"]=None
        out.append(row)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unpatched",required=True,type=Path)
    p.add_argument("--patched",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error("refusing to overwrite benchmark result")
    rows=compare(load(a.unpatched),load(a.patched))
    report={"schema":1,"source":"Schaka PR #8 request",
            "results":rows,"total_cases":len(rows),
            "both_arms_correct":sum(x["slowdown_pct"] is not None for x in rows),
            "performance_regression_gate":"request review if any correct case >10pct",
            "all_large_input_performance_acceptable":all(
                x["over_10pct"] is False for x in rows
                if x["slowdown_pct"] is not None and
                (int(x["shape"].split("x")[-1])>=151936)),
            "release_admitted":False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"total":report["total_cases"],
                      "comparable":report["both_arms_correct"],
                      "large_input_gate":report["all_large_input_performance_acceptable"]}))
if __name__=="__main__":main()
