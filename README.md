# rocm-gfx803

AMD Polaris (gfx803: RX 460/470/480/560/570/580/590 and friends) support for
the MIGraphX + ONNX Runtime + PyTorch stack, split out from
[`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) into its own
repository.

## Prebuilt images -- don't build this yourself

CI already builds and pushes the final image to GHCR. You do not need to
build the Dockerfile locally unless you're patching something. Pull it:

```bash
# rocm7 (main line, ROCm 7.14) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx803

# rocm7 -- always the newest successful rocm7 build
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803

# rocm6.4.4 (older, hardware-verified line) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm6.4.4-gfx803
```

See "Repository layout" and `.github/workflows/gfx803-component.yml` for
the full tag scheme (per-component images, cache tags, dated tags).

## Why a separate repo

gfx803 is unsupported upstream as of ROCm 6.0 -- AMD stopped building for it,
and ROCm 7 rejects the card outright at HSA agent creation. Everything that
makes gfx803 work at all is a local patch: a legacy-doorbell restore just to
get dispatch working on ROCm 7, plus a growing set of Tensile/MIOpen/MIGraphX
correctness patches for bugs that only manifest on this old GCN3 hardware.
That patch set changes independently of, and faster than, the mainline
nightly/release pipeline `rocm-migraphx-ort-builder` runs for every other
architecture -- keeping it there meant every gfx803-specific investigation
was noise in a repo that otherwise doesn't need any of it. This repo is
where that investigation and patch history actually lives now.

The two repos stay linked in one direction: `rocm-migraphx-ort-builder`'s
docs point here for gfx803; this repo doesn't try to track or duplicate the
mainline build's own per-arch matrix.

## Repository layout

```
rocm-gfx803/
├── Dockerfile              # ROCm 7.14 (TheRock), the primary/actively-developed line
├── .dockerignore
├── patches/                # gfx803 patches against the 7.14 pin
├── tools/                  # correctness-suite, reduce-harness (copies, not shared)
├── verify.py                # on-hardware smoke test
├── MIGRATION_NOTES.md      # running investigation log, written as findings happen
├── rocm6.4.4/               # the older, hardware-verified, classic-tag ROCm 6.4.4 line
│   ├── Dockerfile
│   ├── patches/
│   ├── tools/
│   ├── verify.py
│   ├── KERNEL_BUGS.md      # the original gfx803 bug-hunting methodology and record
│   └── wip_patches/        # rejected/superseded patch designs, kept for the record
└── .github/workflows/      # CI, same image tags/ghcr.io paths as before the split
```

**rocm7 (this repo's root) is the actively developed line.** ROCm 7.14 is a
TheRock meta-release, not a classic per-repo tag -- every component is pinned
to a release *branch*, not a frozen commit: ROCR-Runtime, rocBLAS and MIOpen
(now the `rocm-systems`/`rocm-libraries` monorepos) track
`release/therock-7.14`, MIGraphX tracks `release/rocm-rel-7.14`, same
`release/rocm-rel-<major.minor>` convention this repo's mainline
(`rocm-migraphx-ort-builder`) uses for its own manual releases -- no nightly
scheduling, no prebuilt wheels, just "build from the named release branch's
current tip when someone runs it." See `MIGRATION_NOTES.md` for how those refs
were resolved and why.

**`rocm6.4.4/` is the older, stable line** -- classic per-repo `rocm-6.4.4`
tags, hardware-verified over a longer period, actively maintained but not
where new investigation happens. It's kept because it's still the only line
with a full correctness-suite pass and extensive real-hardware mileage; the
rocm7 line inherits its patches and methodology but does not share code with
it -- **the two lines are deliberately independent copies**, not a shared
asset. Both are under active development; a shared file would mean a bug
found while chasing 7.14 could reach the hardware-verified 6.4.4 build. Once
7.14 is confirmed at least as solid as 6.4.4 across the board, the two get
diffed and consciously merged back together -- see "Convergence" below.

## Status

- **rocm7**: builds end-to-end, confirmed on real gfx803 hardware --
  `rocminfo` enumerates the card as a real `KERNEL_DISPATCH` agent, and
  rocBLAS/MIOpen/MIGraphX/PyTorch/ORT have all done real GPU work on it. The
  full `tools/correctness-suite/` (23 MIOpen op/solver sweeps) passes clean.
ORT's own `onnx_backend_test_series.py` (3828 tests) run and diffed
   against both the 6.4.4 line and a gfx1201 (mainline ROCm 7) image to
   separate real regressions from generic upstream gaps. One confirmed
   gfx803-specific regression (the `attention_*_gqa_with_past_and_present_expanded`
   tests, traced to the rocBLAS strided-batched gemm GSU workspace-reuse
   miscompute) was root-caused and fixed via the sgemm-shim's new
   strided-batched interceptor -- see `MIGRATION_NOTES.md`. One regression
   remains open (`ConvTranspose`, traced to an upstream MIGraphX bug, not
   gfx803-specific). Real-model
   validation (faster-whisper/CTranslate2, whisper.cpp, parakeet.cpp, all
   HIP-accelerated) all pass with correct transcripts on real audio.
- **rocm6.4.4**: hardware-verified, the longer-running of the two lines. See
  `rocm6.4.4/README.md` and `rocm6.4.4/KERNEL_BUGS.md`.
- **therock-experimental**: EXPERIMENTAL, not hardware-verified, a third
  independent line. Builds gfx803 through TheRock's own build
  orchestration (`fetch_sources.py --patch-tag` + `cmake
  -DTHEROCK_AMDGPU_FAMILIES` + `ninja`) instead of this repo's hand-rolled
  Dockerfile, to test whether this repo's patches survive AMD's own build
  system -- see `therock-experimental/README.md` and
  `ROCM_UPSTREAM_ANALYSIS.md`.

## Building

Both lines build the same way -- multi-stage Dockerfile, patches applied
per-project before each component compiles from source:

```sh
# rocm7 (this repo's root)
docker build -t rocm-gfx803:rocm7 .

# rocm6.4.4
docker build -t rocm-gfx803:rocm6.4.4 -f rocm6.4.4/Dockerfile rocm6.4.4/
```

Every component (ROCR-Runtime/CLR, rocBLAS, MIOpen, MIGraphX, PyTorch,
torchvision, torchaudio, ONNX Runtime) is compiled from source, every time --
there is no prebuilt-wheel shortcut for gfx803 anywhere upstream, unlike the
mainline repo's newer/more-common architectures, which can sometimes import
an already-published wheel instead of recompiling. CI reflects that: no
`ARG *_IMAGE=...`-style "pull instead of build" branch exists for gfx803.

For on-hardware checks, `verify.py` (and `rocm6.4.4/verify.py`, identical in
spirit) asserts the MIGraphX EP is present and does real GPU work -- run it
inside a container started with `--device=/dev/kfd --device=/dev/dri
--group-add video`.

## Patches: philosophy and conventions

Every patch under `patches/`/`rocm6.4.4/patches/` documents, in its own
header, the WHY (what's broken, how it was found, hardware measurements
where applicable) and the WHAT (the actual fix), plus a re-diff note when a
patch was carried from one line to the other and something in the upstream
source shifted. Read the patch header before touching the code it targets --
the reasoning usually isn't obvious from the diff alone.

**Two apply dialects, on purpose, not by accident**: `.sh` drivers under
`patches/rocm-systems/` use `git apply` (their target, `rocm-systems`, is
cloned as a real git repo root); everything else (`rocblas/`, `miopen/`,
`migraphx/`, `pytorch/`) uses `patch -p1`, because those targets are
sparse-checked-out *subdirectories* of a monorepo, and this box's git
version silently no-ops `git apply --check` there ("Skipped patch", exit 0,
nothing modified) instead of failing loudly. Every driver script is
self-verifying: it greps for a marker string after applying and fails the
build if the marker isn't there, so a patch that silently stopped applying
can't ship unpatched code.

## When a patch needs updating

A gfx803 patch stops applying (or starts applying with fuzz) whenever the
pinned upstream commit moves and the target file changed shape around it --
that's expected, not a sign something is wrong with the patch itself. Before
re-diffing:

1. **Check whether the bug is even still there.** Re-pinning to a newer
   upstream commit sometimes fixes the underlying issue outright (it's
   happened before -- see `MIGRATION_NOTES.md`'s MIGraphX section, where two
   6.4.4-era ONNX-parser patches turned out to be fully obsolete against
   7.14, one because the fix already landed upstream, one because the whole
   code path it patched was replaced). Grep the new source for the
   patch's target function/struct before assuming a straight re-diff is
   needed.
2. **Check whether the fix is still gfx803-specific.** Some of these bugs
   are architecture-general defects that just happen to be *exposed* by
   gfx803's kernel/solver selection (the WGM Tensile swizzle bug, the
   small-GEMM assembly miscompute) rather than genuine hardware quirks. If
   re-investigating turns up a bug that would also misfire on other
   architectures using the same code path, that's a signal to report it
   upstream instead of (or in addition to) patching around it here -- see
   the `rewrite_convolution.cpp`/ConvTranspose writeup in
   `MIGRATION_NOTES.md` for a worked example of that judgment call.
3. **Re-verify on real hardware, not just "applies clean."** A patch that
   compiles is not a patch that's confirmed fixed -- several patches in this
   repo's history were re-diffed successfully but flagged "not yet
   re-verified on real hardware" until someone actually ran the repro
   against the new binaries. Don't assume a clean apply means the original
   bug is still handled correctly.

## Host BIOS setting: keep PCIe ASPM off

On at least one gfx803 host, PCIe ASPM (link power management) being enabled
in the BIOS caused rare, extremely hard-to-diagnose stalls/hangs under GPU
load -- the kind that look like a driver or kernel bug and burn hours of
debugging before the actual cause turns out to be a power-management setting
outside the software stack entirely. Keep ASPM disabled in BIOS (and via
`setpci`/kernel cmdline if the board won't hand OS-level control to Linux)
on any gfx803 host until proven otherwise on that specific board. See
`tools/host-setup/` for a working `setpci`-based systemd unit for boards
where BIOS/firmware won't actually hand ASPM control to Linux.

## What needs real gfx803 hardware to validate, and what doesn't

- **Needs the real card**: anything that dispatches a GPU kernel --
  `verify.py`, `tools/correctness-suite/`, `tools/reduce-harness/`, any real
  transcription/inference run, MIOpen's own `MIOpenDriver -V 1` verification.
  Silent miscompute is the recurring bug class here (`rocblas_status_success`
  returned with wrong numbers) -- CPU/emulation cannot reproduce it, and a
  patch that only "applies clean" and "compiles" has verified nothing about
  correctness.
- **Doesn't need the card**: whether a Dockerfile builds at all, whether a
  patch applies against a given pin, source-level tracing of *where* a bug
  lives (MIOpen's own `MIOPEN_ENABLE_LOGGING_CMD`/`MIGRAPHX_TRACE_COMPILE`
  traces and upstream source diffs found several root causes in this repo's
  history without ever touching a GPU), and cross-arch differential testing
  against an image for a *different* card (used repeatedly in
  `MIGRATION_NOTES.md` to separate "this line broke it" from "upstream never
  worked here").

## Convergence

Once the rocm7 line is confirmed at least as solid as `rocm6.4.4/` --
correctness-suite clean (already true), ORT suite parity (one open item, see
Status above), and comparable real-model results (already true) -- the two
lines get diffed and deliberately merged: whatever's still 6.4.4-only that
should generalize moves up, and the two independent copies collapse back
into a shared structure. Not done yet; both lines are still receiving
patches independently.

## See also

- `MIGRATION_NOTES.md` -- the detailed, as-found rocm7 investigation log.
  Read this before assuming something is broken or fixed; it's the primary
  source of truth for what's been checked and how.
- `rocm6.4.4/KERNEL_BUGS.md` -- the original gfx803 bug-hunting methodology
  and bug record for the 6.4.4 line.
- [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) -- the
  mainline (gfx900+) build this repo split off from.
