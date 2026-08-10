# MIOpen correctness suite

Standalone, MIOpen-only (no ORT/PyTorch/model involved) correctness sweeps.
Each sweep exercises one MIOpen op API directly with synthetic random inputs,
computes a host CPU reference, and compares via cosine similarity (falling
back to max-abs-diff when the reference is near-zero, e.g. clamped
activations). This is the tooling that found and verified all three MIOpen
bugs fixed by `gfx803/patches/miopen/` so far.

## Why this exists

Real-model bisection (extracting a suspect node from an actual ONNX graph
and comparing ROCM EP vs CPU EP) is how these bugs were *first* found, but
it's slow and only covers whatever ops and shapes happen to appear in the
models you're currently running. These sweeps instead target each MIOpen
solver's `IsApplicable` boundary conditions directly — the same
Ceiling()/padding/threshold-math boundaries where every bug found so far
actually lived — so they're both faster to run and cover shapes no
production model may ever hit but a future one might.

## What's covered

- `conv_solver_sweep.cpp` + `shapes/*.txt` — one shape list per MIOpen conv
  solver, run with that specific solver forced via
  `MIOPEN_DEBUG_FIND_ONLY_SOLVER`. Add a new solver by dropping a new
  `shapes/<SolverName>.txt` file (format: `C H W K R S group stride pad`,
  one shape per line, `#` comments allowed) — `run_all.sh` picks it up
  automatically, no code changes needed.
- `activ_sweep.cpp` — all 10 `miopenActivationMode_t` modes, packed +
  strided tensors, vectorized-read-unit boundaries (widths not divisible
  by 4/2).
- `pool_sweep.cpp` — Max/Average/AverageInclusive, output sizes straddling
  the 8/16/32/64/128 work-group tile thresholds.
- `bn_sweep.cpp` — BatchNorm forward inference, Spatial + PerActivation.
- `softmax_sweep.cpp` — CHANNEL/INSTANCE modes, vector_size straddling the
  256 `num_batch` threshold.
- `layernorm_sweep.cpp` — norm_dim straddling the 256 `LOCAL_SIZE` reduction
  threshold.
- `groupnorm_sweep.cpp` — boundary cases around `IsApplicable`'s
  `N*num_groups>=32` / `C/num_groups<64` constraints.
- `tensorop_sweep.cpp` — elementwise Add/Mul/Min/Max, full-tensor and
  broadcast (bias-shaped) variants.
- `reduce_sweep.cpp` — Sum/Prod over a non-last axis (this is what caught
  the Prod-always-returns-zero bug).
- `reduce_extreme_sweep.cpp` — Min/Max/Argmin/Argmax over a non-last axis
  (sibling source-file family to reduce_sweep.cpp's bug; checked on its own
  since an adjacent identity/init bug is exactly the kind of thing likely
  to repeat in adjacent code — came back clean).

- `glu_sweep.cpp` — GLU (gated linear unit); only solver requires dim==0,
  a flat first-half/second-half split, not a real axis split.
- `cat_sweep.cpp` — tensor concatenation; `IsImprovementOverROCm` requires
  output element count >=1,000,000, below which MIOpen has no solver
  reachable from this direct API call (not a bug, just no fallback outside
  ORT's own dispatch layer).
- `rope_sweep.cpp` — Rotary Position Embedding, relevant to any modern
  transformer decoder; interleaved-pair rotate_half convention.
- `kthvalue_sweep.cpp` — k-th smallest value + index along an axis;
  `IsImprovementOverROCm` requires the reduce dim to be the *last*,
  contiguous axis with size >=300.

Deliberately not covered: `adam`, `prelu` (no forward solver in this
bucket, only backward), `getitem`, `multimarginloss`, `softmarginloss` --
training-optimizer/loss/gradient ops with no inference-time callsite in
this project's stack. `mha` (multi-head attention) is real inference-path
math (arch-blind, no gfx803 exclusion) but uses MIOpen's newer
Problem/Find2.0 API (build a problem descriptor, find solutions, run one
-- not a single function call like everything else here); a fragile rushed
harness for something this complex risks a false result more than it's
worth, so it's flagged as a separate follow-up rather than covered here.

## Running it

Build inside any container that has `hipcc` + MIOpen installed (any image
this repo builds qualifies) with the repo mounted in:

```sh
podman run --rm -it \
  -v /path/to/rocm-migraphx-ort-builder:/work:Z \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  <image> /bin/bash

# inside the container:
sh /work/gfx803/tools/correctness-suite/build.sh /tmp/suite-bin /opt/rocm-6.4.4
sh /work/gfx803/tools/correctness-suite/run_all.sh /tmp/suite-bin
```

`run_all.sh` exits non-zero if anything failed, so it's safe to use as a
CI/regression gate. Output is per-shape (`cos=1.00000  OK` /
`<-- WRONG`), followed by a `[PASS]`/`[FAIL]` line per sweep and a final
summary count. Results (full per-sweep logs + `summary.csv` +
`summary.txt`) land in a timestamped `results/<timestamp>/` dir by default
(or pass an explicit dir as the 2nd arg) -- safe to kick off with `nohup`/
`&` and walk away, nothing depends on an attached terminal:

```sh
nohup sh run_all.sh /tmp/suite-bin /tmp/suite-results > /tmp/suite-results.out 2>&1 &
```

### Running just one thing

Once a bug is found, you'll want to rerun *only* that one sweep (or one
conv solver) repeatedly while iterating on a fix -- not the whole suite
every time. Both `build.sh` and `run_all.sh` take an `ONLY` env var
(space-separated names) for exactly this:

```sh
ONLY=reduce_sweep sh build.sh /tmp/suite-bin /opt/rocm-6.4.4
ONLY=reduce_sweep sh run_all.sh /tmp/suite-bin /tmp/suite-results

# conv solver sweeps are named "conv:<SolverName>" (matching shapes/*.txt):
ONLY="conv:ConvBinWinogradRxS" sh run_all.sh /tmp/suite-bin /tmp/suite-results

# multiple at once:
ONLY="reduce_sweep conv:ConvBinWinogradRxS" sh run_all.sh /tmp/suite-bin /tmp/suite-results
```

## Using it for regression / cross-arch testing

None of these sweeps hardcode gfx803 — they run whatever solver MIOpen
actually selects (or force a specific one by name for the conv sweep).
That means the exact same suite runs unmodified against any other
architecture's image, which makes it useful beyond gfx803:

- **Regression gate**: rerun after every ROCm/MIOpen version bump or patch
  change, to catch a reintroduced or newly-introduced bug before it ships
  in a real model months later.
- **Cross-arch differential**: run identically on two different GPUs (e.g.
  gfx803 vs gfx900+) and diff the outputs. Divergence between archs on the
  *same* solver/shape is a strong bug signal even without a host reference.
- **User-reproducible bug reports**: if a user hits a suspected correctness
  issue on their own GPU and we can't reproduce it locally, this is meant
  to be something they can build and run themselves and send back the
  output — no model, dataset, or ORT session required.

## Adding a new sweep

Follow the existing files as a template: `#include "common.hpp"` for the
shared `CHECK_HIP`/`CHECK_MIO` macros and `cos_sim`/`vectors_match`, write
a host CPU reference function, a `run_one(...)` that builds MIOpen
descriptors + calls the op + compares, and a `main()` that sweeps a list of
cases printing one `PASS`/`WRONG` line per case plus a final `SUMMARY: N
tested, M WRONG` line to stderr — that summary line is what `run_all.sh`'s
exit-code check relies on (via the binary's own exit code, not string
matching). Bias shapes toward the target solver's `IsApplicable` boundary
conditions (thresholds, Ceiling()/padding math, mode-specific branches),
not just "a few random typical shapes" — that's where every bug found so
far actually lived.
