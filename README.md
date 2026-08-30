# rocm-gfx803

AMD Polaris (gfx803: RX 460/470/480/560/570/580/590 and friends) support for
the MIGraphX + ONNX Runtime + PyTorch stack, split out from
[`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) into its own
repository.

## Prebuilt images -- don't build this yourself

CI already builds and pushes the final image to GHCR. You do not need to
build the Dockerfile locally unless you're patching something. Pull it:

```bash
# rocm10 (main line, TheRock 10.0) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm10.0-gfx803

# rocm10 -- always the newest successful rocm10 build
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803

# rocm7.14 (older, hardware-verified line) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx803

# rocm6.4.4 (older, hardware-verified line) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm6.4.4-gfx803
```

See "Repository layout" and `.github/workflows/gfx803-component.yml` for
the full tag scheme (per-component images, cache tags, dated tags).

## Why a separate repo

gfx803 is unsupported upstream as of ROCm 6.0 -- AMD stopped building for it,
and ROCm 7+ rejects the card outright at HSA agent creation. Everything that
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
mainline build's own per-arch matrix. This repo's *versions* do track the
mainline release track, though: every pinned ref here matches what the
mainline repo's `release.yml` ships for the same ROCm line (MIGraphX
`release/rocm-rel-10.0`, ORT `v1.29.0`, PyTorch `2.14.0` / `release/2.14`),
so gfx803 doesn't silently lag the supported line it's split off from.

## Repository layout

```
rocm-gfx803/
├── Dockerfile              # ROCm 10.0 (TheRock), the primary/actively-developed line
├── .dockerignore
├── patches/                # gfx803 patches against the 10.0 pin (copied from 7.14,
│                           #   NOT yet re-verified against 10.0 -- see Status)
├── tools/                  # correctness-suite, host-setup, etc. (shared across lines)
├── verify.py               # on-hardware smoke test
├── MIGRATION_NOTES.md      # running investigation log for the 10.0 line
├── vllm/                   # the gfx803 vLLM hard fork (moved to the 10.0 line;
│                           #   re-targeting from the 7.14 stack in progress)
├── rocm7.14/               # the hardware-verified TheRock 7.14 line
│   ├── Dockerfile
│   ├── patches/
│   ├── verify.py
│   ├── README.md           # incl. the full vLLM-on-gfx803 investigation
│   └── MIGRATION_NOTES.md
├── rocm6.4.4/              # the older, hardware-verified, classic-tag ROCm 6.4.4 line
│   ├── Dockerfile
│   ├── patches/
│   ├── tools/
│   ├── verify.py
│   ├── KERNEL_BUGS.md      # the original gfx803 bug-hunting methodology and record
│   └── wip_patches/        # rejected/superseded patch designs, kept for the record
├── llama-cpp-gfx803/       # llama.cpp gfx803 patches (arch-level, shared)
├── RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md  # hardware-level, shared across lines
└── .github/workflows/      # CI, same image tags/ghcr.io paths as before the split
```

**rocm10 (this repo's root) is the actively developed line.** ROCm 10.0 is a
TheRock meta-release, not a classic per-repo tag -- every component is pinned
to a release *branch*, not a frozen commit: ROCR-Runtime, rocBLAS and MIOpen
(now the `rocm-systems`/`rocm-libraries` monorepos) track
`release/therock-10.0`, MIGraphX tracks `release/rocm-rel-10.0`, same
`release/rocm-rel-<major.minor>` convention this repo's mainline
(`rocm-migraphx-ort-builder`) uses for its own manual releases -- no nightly
scheduling, no prebuilt wheels, just "build from the named release branch's
current tip when someone runs it." See `MIGRATION_NOTES.md` for how those refs
were resolved.

**`rocm7.14/` is the immediately-preceding line** -- the same hand-rolled
build pattern, hardware-verified, and where the current gfx803 vLLM fork and
its investigation live. It's kept because it's the most recent line with a
full correctness-suite pass and extensive real-hardware mileage; rocm10
inherits its patches and methodology but does not share code with it.

**`rocm6.4.4/` is the older, stable line** -- classic per-repo `rocm-6.4.4`
tags, hardware-verified over the longest period, actively maintained but not
where new investigation happens.

**The three lines are deliberately independent copies, not a shared asset.**
All are under active maintenance at different verification levels; a shared
file would mean a bug found while chasing 10.0 could reach the
hardware-verified 7.14 or 6.4.4 builds. Once 10.0 is confirmed at least as
solid as 7.14 across the board, the lines get diffed and consciously merged
back together -- see "Convergence" below.

## Status

- **rocm10 (this repo's root)**: pin-bumped to TheRock 10.0 (see
  `MIGRATION_NOTES.md`). **NOT yet built, NOT yet hardware-verified.** The
  Dockerfile builds the same stages as the 7.14 line against the 10.0 refs
  with the 7.14 patch set copied over un-re-diffed. A build has not been
  attempted; do not treat the copied patches as confirmed. The 7.14 line's
  full validation arc (build, `verify.py`, `tools/correctness-suite/`, ORT
  backend-test series, real-model runs) must be re-run on this line against
  a real gfx803 card before it can be called verified.
- **rocm7.14**: hardware-verified. `rocminfo` enumerates the card as a real
  `KERNEL_DISPATCH` agent; rocBLAS/MIOpen/MIGraphX/PyTorch/ORT all do real
  GPU work on it. The full `tools/correctness-suite/` (23 MIOpen op/solver
  sweeps) passes clean. ORT's `onnx_backend_test_series.py` (3828 tests) has
  one open, non-gfx803-specific regression (`ConvTranspose`, an upstream
  MIGraphX bug -- reported upstream, not patched here). Real-model validation
  (faster-whisper/CTranslate2, whisper.cpp, parakeet.cpp) passes with correct
  transcripts on real audio. The gfx803 vLLM fork is built and verified
  against this line's stack. See `rocm7.14/README.md` for the full detail.
- **rocm6.4.4**: hardware-verified, the older and longer-running of the lines.
  See `rocm6.4.4/README.md` and `rocm6.4.4/KERNEL_BUGS.md`.

### Required host setup

- **VBIOS/VRAM clock**: this card's VRAM must run within its rated speed
  ceiling (1750MHz on the mining-tuned VBIOS this box shipped with,
  confirmed via core overdrive; a stock/correct-vendor VBIOS needs no
  overdrive). Running VRAM above spec causes real GPU VM faults under
  MIOpen (`pool_sweep`) and GPU hangs under vLLM -- both the same
  hardware-level root cause, not a software bug. See "Host VBIOS setting"
  below and `RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md` for the
  investigation.
- `patches/rocm-systems/aql-ring-queue-full-workaround.patch` restores the
  AQL ring's GFXIP 7/8 double mapping (64 -> 131072 packets, a 2048x
  increase over the unpatched cap) and is required for
  `graph-replay-batch-chunk-deadlock.patch` to be unnecessary. Needs a
  kernel WITHOUT `REFERENCE-amdkfd-gfx7-8-queue-size-writeback`, and must
  never be combined with `graph-replay-queue-size-cap.patch`.

### vLLM on gfx803

The gfx803 vLLM hard fork lives at `vllm/` (repo root, the 10.0 line),
moved there when the 7.14 line was archived. It is being re-targeted from
the 7.14 stack to the 10.0 stack: the prebuilt `libgfx803gemm.so`/
`libgfx803attn.so` currently link the 7.14 `hipcc` and `librocblas.so` is
picked up via `LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib`; on 10.0 those
become `core-10.0` and the kernels must be recompiled with the 10.0 `hipcc`
(the `.so` linkage is stack-specific). Re-verify on hardware after
re-targeting -- the box's editable vLLM install tracks whichever ROCm stack
the box runs, so the re-target happens when the box moves to 10.0. The full
investigation and tuning notes for the 7.14-era state are in
`rocm7.14/README.md`.

## Building

All lines build the same way -- multi-stage Dockerfile, patches applied
per-project before each component compiles from source:

```sh
# rocm10 (this repo's root)
docker build -t rocm-gfx803:rocm10 .

# rocm7.14
docker build -t rocm-gfx803:rocm7.14 -f rocm7.14/Dockerfile rocm7.14/

# rocm6.4.4
docker build -t rocm-gfx803:rocm6.4.4 -f rocm6.4.4/Dockerfile rocm6.4.4/
```

Every component (ROCR-Runtime/CLR, rocBLAS, MIOpen, MIGraphX, PyTorch,
torchvision, torchaudio, ONNX Runtime) is compiled from source, every time --
there is no prebuilt-wheel shortcut for gfx803 anywhere upstream, unlike the
mainline repo's newer/more-common architectures, which can sometimes import
an already-published wheel instead of recompiling. CI reflects that: no
`ARG *_IMAGE=...`-style "pull instead of build" branch exists for gfx803.

For on-hardware checks, `verify.py` (and `rocm7.14/verify.py` /
`rocm6.4.4/verify.py`, identical in spirit) asserts the MIGraphX EP is
present and does real GPU work -- run it inside a container started with
`--device=/dev/kfd --device=/dev/dri --group-add video`.

## Patches: philosophy and conventions

Every patch under `patches/`/`rocm7.14/patches/`/`rocm6.4.4/patches/`
documents, in its own header, the WHY (what's broken, how it was found,
hardware measurements where applicable) and the WHAT (the actual fix), plus
a re-diff note when a patch was carried from one line to another and
something in the upstream source shifted. Read the patch header before
touching the code it targets -- the reasoning usually isn't obvious from the
diff alone.

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
   happened before -- see `rocm7.14/MIGRATION_NOTES.md`'s MIGraphX section,
   where two 6.4.4-era ONNX-parser patches turned out to be fully obsolete
   against 7.14, one because the fix already landed upstream, one because
   the whole code path it patched was replaced). Grep the new source for the
   patch's target function/struct before assuming a straight re-diff is
   needed.
2. **Check whether the fix is still gfx803-specific.** Some of these bugs
   are architecture-general defects that just happen to be *exposed* by
   gfx803's kernel/solver selection (the WGM Tensile swizzle bug, the
   small-GEMM assembly miscompute) rather than genuine hardware quirks. If
   re-investigating turns up a bug that would also misfire on other
   architectures using the same code path, that's a signal to report it
   upstream instead of (or in addition to) patching around it here.
3. **Re-verify on real hardware, not just "applies clean."** A patch that
   compiles is not a patch that's confirmed fixed -- several patches in this
   repo's history were re-diffed successfully but flagged "not yet
   re-verified on real hardware" until someone actually ran the repro
   against the new binaries. Don't assume a clean apply means the original
   bug is still handled correctly. This is the standing state of the whole
   10.0 line's patch set right now.

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

## Host VBIOS setting: mining-tuned VRAM clocks cause random GPU faults and hangs

At least one gfx803 card in use with this repo (Sapphire RX 470 8GB Mining
UEFI, Hynix `H5GQ8H24MJR` VRAM) shipped with a mining-tuned VBIOS running
VRAM (MCLK) at 2000-2100MHz -- above what a 7Gbps-rated Hynix chip is
actually spec'd for. Under a correctness-checked compute workload this
produced two symptoms that both took real investigation to correctly rule
out as driver/software bugs (ioctl tracing, PM4 dispatch tracing,
kernel-side TLB-flush review, GPU-side wave-state capture via debugfs --
see `RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md` for the full investigation on both):

- `tools/correctness-suite/pool_sweep`: an intermittent (~50% of runs),
  deterministic-address GPU VM fault.
- vLLM and other sustained-load workloads: an intermittent, unrecoverable
  hang -- a wave parked forever in `s_waitcnt vmcnt(0)`, waiting on a
  vector-memory op that never returns. Killing the stuck process fails a
  KFD queue eviction and needs a reboot.

Mining workloads tolerate occasional VRAM bit errors that a
correctness-checked or long-running one won't -- it'll eventually surface
as a fault, a hang, or (worse) silently wrong results.

**Fix: flash a VBIOS whose VRAM clock matches the installed memory's real
rated speed**, via `amdvbflash`'s force-flash mode (`amdvbflash -f -p 0
<rom>`; always dump/keep the existing ROM first). Confirmed on hardware two
independent ways:

- Keeping the card's own mining VBIOS but capping MCLK to 1750MHz via core
  overdrive (`amdgpu.ppfeaturemask=0xffffffff` + `pp_od_clk_voltage`):
  64/64 clean runs, vs. repeated same-boot hangs/crashes at 2000MHz with
  every other binary held identical.
- Flashing a real Sapphire RX570 Nitro VBIOS with correct-vendor Hynix
  straps (`113-2E366AU-X56`, downloaded from
  https://www.techpowerup.com/vgabios/212597/212597) whose stock MCLK table
  tops out at 1750MHz, no overdrive needed: 75/75 clean runs at stock
  settings. This is the recommended fix for anyone with the same
  Sapphire RX 470 8GB Mining UEFI card and Hynix memory -- no software
  workaround required at all.

Two other RX570 VBIOS files with **Samsung** straps (wrong vendor for this
card's Hynix chips) failed to probe entirely (`SMU load firmware failed`,
`probe with driver amdgpu failed with error -22`) rather than hang -- a
different, harder failure mode. Match the VBIOS's memory-vendor strap to
the physically installed chips, not just the card model/VRAM size.

Check VRAM clock (`cat /sys/class/drm/card*/device/pp_dpm_mclk`) against
the card's actual rated spec before assuming a gfx803 GPU fault or hang
report is a software bug.

## What needs real gfx803 hardware to validate, and what doesn't

- **Needs the real card**: anything that dispatches a GPU kernel --
  `verify.py`, `tools/correctness-suite/`, any real transcription/inference
  run, MIOpen's own `MIOpenDriver -V 1` verification. Silent miscompute is
  the recurring bug class here (`rocblas_status_success` returned with wrong
  numbers) -- CPU/emulation cannot reproduce it, and a patch that only
  "applies clean" and "compiles" has verified nothing about correctness.
- **Doesn't need the card**: whether a Dockerfile builds at all, whether a
  patch applies against a given pin, source-level tracing of *where* a bug
  lives (MIOpen's own `MIOPEN_ENABLE_LOGGING_CMD`/`MIGRAPHX_TRACE_COMPILE`
  traces and upstream source diffs found several root causes in this repo's
  history without ever touching a GPU), and cross-arch differential testing
  against an image for a *different* card (used repeatedly in
  `rocm7.14/MIGRATION_NOTES.md` to separate "this line broke it" from
  "upstream never worked here").

## Convergence

Once the rocm10 line is confirmed at least as solid as `rocm7.14/` --
correctness-suite clean, ORT suite parity, comparable real-model results --
the lines get diffed and deliberately merged: whatever's still
7.14/6.4.4-only that should generalize moves up, and the independent copies
collapse back into a shared structure. Not done yet; all lines are still
under maintenance at different verification levels.

## See also

- `MIGRATION_NOTES.md` -- the 10.0 migration log (pins, what's inherited,
  what's still open).
- `rocm7.14/MIGRATION_NOTES.md` -- the detailed, as-found 7.14 investigation
  log. Read this before assuming something is broken or fixed on that line;
  it's the primary source of truth for what's been checked and how.
- `rocm7.14/README.md` -- the 7.14 line's full detail, including the vLLM
  on gfx803 investigation and tuning notes.
- `rocm6.4.4/KERNEL_BUGS.md` -- the original gfx803 bug-hunting methodology
  and bug record for the 6.4.4 line.
- [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) -- the
  mainline (gfx900+) build this repo split off from and tracks version-wise.