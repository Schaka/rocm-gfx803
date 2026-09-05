# MIOpen correctness suite

These are standalone MIOpen-only correctness sweeps. No ORT, PyTorch, or model
is involved. Each sweep drives one MIOpen op API directly with synthetic random
inputs, computes a host CPU reference, and compares the two with cosine
similarity. Where the reference is near zero, such as a clamped activation, it
compares the maximum absolute difference instead. This tooling found and
confirmed the MIOpen bugs that `patches/miopen/` fixes today.

## Why this exists

Real-model bisection found these bugs first. That method extracts a suspect node
from a real ONNX graph and compares the ROCM EP against the CPU EP. It is slow,
and it covers only the ops and shapes that happen to appear in the models you run
at the time. These sweeps instead target each MIOpen solver's `IsApplicable`
boundaries: the Ceiling, padding, and threshold math where every bug found so far
actually lived. They run faster, and they cover shapes that no current model
reaches but a future one may.

## What is covered

- `conv_solver_sweep.cpp` plus `shapes/*.txt`: one shape list per MIOpen conv
  solver, each run with that solver forced through
  `MIOPEN_DEBUG_FIND_ONLY_SOLVER`. To add a solver, drop a new
  `shapes/<SolverName>.txt` file in place. The format is
  `C H W K R S group stride pad`, one shape per line, and `#` starts a comment.
  `run_all.sh` picks the new file up with no code change.
- `activ_sweep.cpp`: all 10 `miopenActivationMode_t` modes, packed and strided
  tensors, and widths that are not divisible by 4 or 2, so vectorized read units
  are crossed.
- `pool_sweep.cpp`: Max, Average, and AverageInclusive, with output sizes that
  straddle the 8, 16, 32, 64, and 128 work-group tile thresholds.
- `bn_sweep.cpp`: BatchNorm forward inference, Spatial and PerActivation.
- `softmax_sweep.cpp`: CHANNEL and INSTANCE modes, with vector_size straddling
  the 256 `num_batch` threshold.
- `layernorm_sweep.cpp`: norm_dim straddling the 256 `LOCAL_SIZE` reduction
  threshold.
- `groupnorm_sweep.cpp`: boundary cases around `IsApplicable`'s
  `N*num_groups>=32` and `C/num_groups<64` constraints.
- `tensorop_sweep.cpp`: elementwise Add, Mul, Min, and Max, full-tensor and
  broadcast (bias-shaped).
- `reduce_sweep.cpp`: Sum and Prod over a non-last axis. This caught the bug
  where Prod always returned zero.
- `reduce_extreme_sweep.cpp`: Min, Max, Argmin, and Argmax over a non-last axis.
  This is the sibling source-file family of the bug above, and it is checked on
  its own, because an adjacent identity or init fault is the kind of thing that
  repeats in nearby code. It came back clean.
- `glu_sweep.cpp`: gated linear unit. Its only solver requires dim==0, which is a
  flat first-half and second-half split and not a real axis split.
- `cat_sweep.cpp`: tensor concatenation. `IsImprovementOverROCm` requires an
  output element count of at least 1,000,000. Below that no MIOpen solver is
  reachable from this direct API call. That is not a bug; there is simply no
  fallback outside ORT's own dispatch layer.
- `rope_sweep.cpp`: Rotary Position Embedding, which every modern transformer
  decoder uses, with the interleaved-pair rotate_half convention.
- `kthvalue_sweep.cpp`: the k-th smallest value and its index along an axis.
  `IsImprovementOverROCm` requires the reduce dim to be the last, contiguous axis
  with size at least 300.

These are deliberately not covered: `adam`, `prelu` (no forward solver in this
group, backward only), `getitem`, `multimarginloss`, and `softmarginloss`. They
are training, optimizer, or loss ops with no inference-time call site in this
project's stack. `mha` (multi-head attention) is real inference-path math. It is
arch-blind, with no gfx803 exclusion, but it uses MIOpen's newer Problem and
Find2.0 API: build a problem descriptor, find solutions, then run one. Everything
else here is a single function call. A rushed harness for something this complex
risks a false result more than it is worth, so `mha` is a separate follow-up and
is not covered here.

## Running it

Build inside any container that has `hipcc` and MIOpen installed. Every image
this repo builds qualifies. Mount the repo in:

```sh
podman run --rm -it \
  -v /path/to/rocm-gfx803:/work:Z \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  <image> /bin/bash

# inside the container:
sh /work/tools/correctness-suite/build.sh /tmp/suite-bin /opt/rocm
sh /work/tools/correctness-suite/run_all.sh /tmp/suite-bin
```

`run_all.sh` exits non-zero when anything fails, so it works as a CI or
regression gate. Output is one line per shape (`cos=1.00000  OK` or
`<-- WRONG`), then one `[PASS]` or `[FAIL]` line per sweep, then a final summary
count. Results (the full per-sweep logs plus `summary.csv` and `summary.txt`) go
to a timestamped `results/<timestamp>/` directory by default, or to a directory
you pass as the second argument. Nothing depends on an attached terminal, so
`nohup` and `&` are safe:

```sh
nohup sh run_all.sh /tmp/suite-bin /tmp/suite-results > /tmp/suite-results.out 2>&1 &
```

### Running one sweep only

After you find a bug you iterate on a fix, and you want to rerun that one sweep,
or one conv solver, again and again instead of the whole suite. `build.sh` and
`run_all.sh` both take an `ONLY` env var with space-separated names for this:

```sh
ONLY=reduce_sweep sh build.sh /tmp/suite-bin /opt/rocm
ONLY=reduce_sweep sh run_all.sh /tmp/suite-bin /tmp/suite-results

# conv solver sweeps are named "conv:<SolverName>", matching shapes/*.txt:
ONLY="conv:ConvBinWinogradRxS" sh run_all.sh /tmp/suite-bin /tmp/suite-results

# several at once:
ONLY="reduce_sweep conv:ConvBinWinogradRxS" sh run_all.sh /tmp/suite-bin /tmp/suite-results
```

## Using it as a regression gate and for cross-arch testing

No sweep hardcodes gfx803. Each one runs whatever solver MIOpen actually selects,
except the conv sweeps, which force one solver by name. So the same suite runs
unmodified against an image for any other architecture, which makes it useful
beyond gfx803:

- Regression gate: rerun it after every ROCm or MIOpen version bump and after
  every patch change. It catches a reintroduced or newly introduced bug before a
  real model shows it months later.
- Cross-arch differential: run it the same way on two different GPUs, for example
  gfx803 and gfx900 or newer, and diff the outputs. Divergence between
  architectures on the same solver and shape is a strong bug signal even without
  a host reference.
- Bug reports a user can reproduce: when a user hits a suspected correctness
  problem on their own GPU and it cannot be reproduced here, they can build and
  run this themselves and send the output back. It needs no model, no dataset,
  and no ORT session.

## Adding a new sweep

Use the existing files as the template. Include `common.hpp` for the shared
`CHECK_HIP` and `CHECK_MIO` macros and for `cos_sim` and `vectors_match`. Write a
host CPU reference function, then a `run_one(...)` that builds the MIOpen
descriptors, calls the op, and compares, then a `main()` that sweeps a list of
cases. Print one `PASS` or `WRONG` line per case and one final
`SUMMARY: N tested, M WRONG` line to stderr. `run_all.sh` relies on the exit code
of the binary for that check, not on string matching. Bias the shapes toward the
target solver's `IsApplicable` boundaries: thresholds, Ceiling and padding math,
and mode-specific branches. A few random typical shapes are not the point. Every
bug found so far lived at a boundary.
