# gfx803 on ROCm 7.14 -- proof of concept

**Status: builds end-to-end and confirmed working on real gfx803
hardware** -- `rocminfo` enumerates the card as a real `KERNEL_DISPATCH`
agent, and rocBLAS, MIOpen, MIGraphX, and ORT's MIGraphX EP have all done
real GPU work on it (single-shot correctness checks, not the full
correctness-suite yet, not a sustained/performance run). See
`MIGRATION_NOTES.md` for the detailed, as-found investigation log, and
`rocm6.4.4/EVALUATION_ROCM_7.md` for the original pre-implementation
feasibility read this implements.

This is a **third build target**, alongside `rocm6.4.4/Dockerfile` (pinned
ROCm 6.4.4, hardware-verified, actively maintained) and the main
repo's nightly/release lines (gfx900+). It does not replace either. ROCm 7
rejects Polaris at HSA agent creation by default (legacy doorbell type),
which 6.4.4 never had to work around -- this directory exists to find out
whether patching around that, and everything downstream of it, is
actually viable, not to commit to it as an ongoing target yet.

## What this is

- `Dockerfile` -- multi-stage build, same shape as `rocm6.4.4/Dockerfile` but
  with a new first stage (`rocr-clr-builder`) that rebuilds ROCR-Runtime
  and CLR (HIP only) from source with the legacy-doorbell restore, before
  the familiar rocBLAS/MIOpen/MIGraphX/PyTorch/ORT stages run on top of
  the patched runtime.
- `patches/` -- gfx803-specific source patches, grouped by project.
  `patches/rocm-systems/` is new for this line (the doorbell/OpenCL
  restore); the rest are re-diffed or unchanged carryovers from
  `rocm6.4.4/patches/`, documented per-patch in `MIGRATION_NOTES.md`.
- `tools/correctness-suite/` -- copied from `rocm6.4.4/tools/`, not yet
  adapted or re-run against 7.14.

Everything here is a **copy, deliberately, not a shared asset**: this
directory is a self-contained fork of the 6.4.4 line that can be worked on
in isolation, so nothing done while chasing 7.14 can reach the
hardware-verified 6.4.4 build. Both lines are still moving. Once gfx803 on
7.14 is fully confirmed working, the copies get diffed and merged back --
which is also why anything written here should stay as close as possible to
its counterpart in the main ROCm 7 build (`docker/`, `scripts/build/`),
not just to 6.4.4's.

For on-hardware checks, `rocm6.4.4/verify.py` works against an image from this
Dockerfile unchanged (it only asserts the MIGraphX EP is present and does
real GPU work) -- see
[`rocm6.4.4/README.md`](../README.md#verifying-on-hardware); only the image tag
differs.

## What's confirmed

- ROCm 7.14 resolves to TheRock's `therock-7.14` meta-release, not a
  classic per-repo tag -- see `MIGRATION_NOTES.md` for how the exact
  source commits were pinned.
- The legacy-doorbell fix restores real dispatch, not just enumeration --
  **now confirmed on real hardware**: `rocminfo` shows `gfx803` as a
  `KERNEL_DISPATCH` agent, and a real `torch` matmul executed on it.
- ROCm 7.14's LLVM still emits real gfx803 device code objects (compiled
  and verified directly, not assumed).
- Most carried-over rocBLAS/MIOpen patches re-diff clean against 7.14.
  One (`conv-direct-fwd-grouped-oob`) is **blocked**: its target MIOpen
  solver was removed upstream and replaced with a different
  implementation whose correctness on this bug is unverified. See
  `MIGRATION_NOTES.md`.
- MIGraphX needed two *new* 7.14-specific patches, not zero -- both
  genuine upstream gaps (declared-but-unstubbed functions behind
  disabled feature flags), not gfx803-specific. See
  `MIGRATION_NOTES.md`'s MIGraphX section.
- PyTorch's `TensorTopK.hip` needs a build-time `-O1` override or it
  OOM-kills the build host (~20GB RSS at `-O3`) -- see
  `MIGRATION_NOTES.md`.
- rocBLAS, MIOpen, MIGraphX, and ORT's MIGraphX EP have all done real
  GPU work on the actual card (single-shot correctness checks, not the
  full correctness-suite).

## What's NOT done yet

- The correctness-suite hasn't been run or adapted for 7.14 -- broader
  shape/op coverage than today's single-shot checks.
- No sustained or performance run -- today's hardware validation was
  all tiny tensors, one-shot, proving correctness of the happy path,
  not stability under load or how it compares to the 6.4.4 line.
- PyTorch/torchvision/torchaudio ref pins (`release/2.13` /
  `release/0.28` / `release/2.11.0.2`) are traced to this repo's own CI
  decision logic (`scripts/torch-package-build-decide.sh`), not an
  independent compatibility-matrix check.

## Next steps

1. Adapt and run `tools/correctness-suite/` against this line, expect
   new findings distinct from the 6.4.4 line's own `KERNEL_BUGS.md` -- a
   Tensile/MIOpen version bump can change *which* shapes are broken
   without changing *whether* something is broken.
2. A longer/heavier real-workload run (not just single-shot ops) to
   check for stability under sustained dispatch, matching the class of
   bug `rocm6.4.4/KERNEL_BUGS.md` documents for the 6.4.4 line (e.g. the
   GSU CAS-accumulate race that only manifests under back-to-back
   dispatch).
