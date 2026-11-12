# gfx803 TopK PR #8 — consolidated review and paired benchmark

This branch addresses all source changes requested in Schaka's PR review,
while leaving the large-input route choice to **measured** GPU evidence.
Do not submit performance assertions without both original CSVs.

## Review changes integrated

1. Replace manual `gcnArchName`/`strncmp` matching with PyTorch's existing
   `at::detail::getCUDAHooks().isGPUArch({"gfx803"})`. Add
   `<ATen/detail/CUDAHooksInterface.h>` and marker
   `GFX803_TOPK_SORT_FALLBACK_PATCH`.
2. Add `patches/pytorch/apply-gfx803-topk-sort-fallback.sh`, following the
   existing upstream patch-driver conventions (`patch -p1`, marker check,
   fail-closed). Use that driver in `docker/pytorch.Dockerfile`.
3. Reuse the reviewer's nine benchmark cases through
   `tools/correctness-suite/topk_bench.py`: small [2,4] k=2;
   router [2048,64] and [2048,128]; sampler [1,151936] and
   [32,151936]; 1D arrays of 1 million and 16 million elements.
   Both fp16 and fp32 are represented.

The source patch currently forces full sort for all non-Boolean gfx803 inputs.
That is the **previous physically validated** behavior, not an assumption
that large-row full sort is efficient. Do NOT route untested large shapes
back to `mbtopk` based solely on expected asymptotics: compare actual
hardware correctness and performance first.

## Paired Nix test — no Docker or mutation of immutable environments

Use the pinned original Schaka ROCm 10.1 binary and patched TopK binary
*with exactly the same ROCm store*, device, kernel, thermal state and
benchmark parameters. Run on a recovered RX 580 without other GPU tenants.

`UNPATCHED_PY` must be the selected original imported Schaka venv Python
under a wrapper supplying the same ROCm library/shim mappings.
`PATCHED_PY` must be the composed Nix patched-wheel wrapper.
Do not assume the old unpatched venv Python has its runtime environment
configured correctly: verify mapped libraries before running a benchmark.

```bash
cd /path/to/rocm-gfx803
UNPATCHED_PY=/nix/store/EXACT-ORIGINAL-SCHAKA-PYTHON-WRAPPER
PATCHED_PY=/nix/store/EXACT-PATCHED-TOPK-PYTHON-WRAPPER
BENCH=$PWD/tools/correctness-suite/topk_bench.py
OUT=/mnt/truenas/comet-net/experiments/gfx803_topk_pr8_review
mkdir -p "$OUT"

# Original binary first. Large inputs run before the known-fault [2,4]
# canary, which is LAST. A fault stops the sweep and retains its CSV.
"$UNPATCHED_PY" "$BENCH" --out "$OUT/unpatched.csv" \
    --warmup 5 --iterations 50 --timeout 300

# Inspect kernel logs and GPU health. Reset/recover the device if a VM fault
# occurred, before the patched arm. Do not auto-continue on a fault.
"$PATCHED_PY" "$BENCH" --out "$OUT/patched.csv" \
    --warmup 5 --iterations 50 --timeout 300

# Compare only on cases where both arms produced correct values and indices.
python3 tools/correctness-suite/topk_bench_compare.py \
    --unpatched "$OUT/unpatched.csv" --patched "$OUT/patched.csv" \
    --out "$OUT/comparison.json"
```

These paths are placeholders; use the **actual** Nix store roots from
the already recorded candidate selector and ABI receipts. The currently
cached patched wheel contains the earlier manual-arch-check source,
which uses the same intended gfx803 dispatch but does **not** validate
the newly reviewed helper or driver build. A new patched-wheel build
from this PR revision is required for source-to-binary acceptance of
these review changes. It must remain separate from the existing
physically validated wheel until matched.

## Results required before a merge decision

Retain `unpatched.csv`, `patched.csv`, `comparison.json`, benchmark
source hash, composed wheel SHA, ROCm root, GPU driver/kernel identity,
and kernel logs. Publish both CSVs to PR #8 only after real RX 580 runs.

* If the unpatched short-row canary faults, record it as a **fault**, not
  a missing or zero-millisecond timing.
* If a large unpatched selection case passes numerically and beats the
  sort fallback, decide on a dispatch heuristic for **that measured
  path**, rebuild the modified wheel, and rerun correctness plus 120-case
  semantics and OpInfo regressions.
* If the patched full-sort large-row cost exceeds the reviewer's
  roughly 5–10% target, do not infer acceptance from correctness alone.
* If both arms fault for a case, keep it open and investigate separately.

The unrelated complex64 OpInfo VM faults (acos, acosh, asin, asinh) and
three ordinary OpInfo discrepancies are tracked in the factory repository.
Those are *not* silently mixed into this TopK patch.
