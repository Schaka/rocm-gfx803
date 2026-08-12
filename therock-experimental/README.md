# gfx803 via TheRock -- EXPERIMENTAL

> [!WARNING]
> **Not hardware-verified. Not a replacement for the root `rocm7` line or
> `rocm6.4.4/`.** This directory exists to answer one question: do this
> repo's gfx803 patches survive being built through TheRock's own
> orchestration (`fetch_sources.py --patch-tag` + `cmake
> -DTHEROCK_AMDGPU_FAMILIES=...` + `ninja`), rather than this repo's
> hand-rolled per-component Dockerfile. It is a third, fully independent
> line -- same "independent copies, don't link" convention `AGENTS.md`
> already applies between `rocm7` and `rocm6.4.4/` -- and nothing here
> feeds back into either of those until this line is itself confirmed
> working on real hardware.

## Why this exists

Following [ROCm/clr#269](https://github.com/ROCm/clr/issues/269#issuecomment-5240585153):
lucbruni-amd (AMD) has a personal, never-PR'd fork branch
(`lucbruni-amd/TheRock@lb/gfx803-polaris-support`) restoring gfx803
HIP/OpenCL support under TheRock, verified on his own RX 550. This repo's
root (`rocm7`) line already carries near-identical patches plus several
correctness fixes his branch doesn't, but has never been run through
TheRock's actual build system -- it's pinned to the same source refs
TheRock uses, but orchestrates the build itself (hand-cloning
`rocm-systems`/`rocm-libraries`, calling `cmake`/`rmake.py -a gfx803`
directly per component). See `../ROCM_UPSTREAM_ANALYSIS.md` for the full
upstream-feasibility writeup this pipeline is meant to generate evidence
for.

This pipeline vendors this repo's patches into TheRock's own
`patches/<tag>/<project>/` layout and builds with TheRock's real tooling,
to find out whether they survive AMD's own build graph and packaging --
independent-of-this-repo evidence for the eventual upstream ask, not a
claim that gfx803 is "done" here.

## What's different from the root (`rocm7`) line

- **`therock-builder`** replaces `rocr-clr-builder` + `rocblas-builder` +
  `miopen-builder` with a single stage: TheRock's cmake super-project
  builds ROCR-Runtime/CLR, rocBLAS and MIOpen together for one
  `-DTHEROCK_AMDGPU_FAMILIES=gfx803-dgpu` target, via `fetch_sources.py`
  and `ninja` -- not three separately-built, separately-cached images.
- **MIGraphX/PyTorch/torchvision/torchaudio/ORT** stages are otherwise
  unchanged from the root Dockerfile -- TheRock doesn't build any of these
  either, upstream or here, so they keep this repo's existing
  clone-and-patch approach, just repointed at `therock-builder`'s
  `/opt/rocm` instead of the root line's rocr-clr+rocblas+miopen stack.
- **TheRock's `gfx803` AMDGPU target does not exist upstream.** Added here
  via `therock-cmake-patch/0001-cmake-add-gfx803-target.patch`, a
  one-line-block addition to `cmake/therock_amdgpu_targets.cmake` modeled
  on the existing `gfx900` entry, applied directly to the TheRock checkout
  (not via `fetch_sources.py` -- that mechanism only patches submodules,
  never TheRock's own top-level repo).
- **Pinned to an exact commit on TheRock's own `release/therock-7.14`
  branch** (`418cd5f63abb7a604bad5874cd7b2e29334e640f`, the tip at the
  time this was written) -- same branch-pin convention every other ref in
  this repo uses, not `main`/`develop`. That branch doesn't carry the
  gfx803 target line itself (added here via
  `therock-cmake-patch/0001-cmake-add-gfx803-target.patch`, applied
  directly to the TheRock checkout since `fetch_sources.py` only patches
  submodules, never TheRock's own top-level repo) -- but its
  `rocm-systems`/`rocm-libraries` submodule pins are what actually
  matters for the vendored patches below, and those line up exactly with
  `../patches/`'s own pins. An earlier version of this pipeline pinned
  `THEROCK_REF` to `main`'s tip instead, which resolved those two
  submodules to different commits and forced two patches to be
  re-ported for no real reason -- see "Patches" below for what that
  looked like and why the branch pin was the actual fix, not the
  re-port.

## Patches: vendored, reformatted, re-verified to apply

TheRock's `fetch_sources.py --patch-tag` applies patches via `git am`,
which requires proper commit-formatted patches (`From:`/`Date:`/
`Subject:` headers), not the plain unified diffs this repo's own `.sh`
drivers consume via `patch -p1`/`git apply`. Every patch under
`therock-patches/gfx803/` is a reformatted copy of a patch that already
exists under `../patches/`, carrying the WHY/hardware-verification summary
in its commit body and pointing back at the original for the full writeup.

**Every patch below was re-verified with a real `git am --whitespace=nowarn`
against the exact commit each submodule actually resolves to under
`THEROCK_REF`** (`rocm-systems @ 2b22ab0195`, `rocm-libraries @
cd95740230`) -- confirmed by directly reading `release/therock-7.14`'s
pinned submodule tree, not assumed. These are the same two commits
`../patches/` was originally verified against, so every patch here
applies with no re-porting needed.

That wasn't true on the first pass: an earlier version of this pipeline
pinned `THEROCK_REF` to `main`'s tip instead of a `release/therock-*`
branch (reasoning at the time: no release branch carries the gfx803
target line, so why would one matter for the pin) and resolved
`rocm-systems`/`rocm-libraries` to different commits than
`../patches/`'s own pins -- `0002-clr-enable-opencl-support-for-gfx803`
needed a real re-port to apply there (upstream had independently moved
some unrelated code into the same function between the two commits).
Switching `THEROCK_REF` to `release/therock-7.14` -- TheRock's actual
release branch, which does exist even though it doesn't carry the gfx803
target itself -- made that re-port unnecessary: its submodule pins turned
out to already match `../patches/`'s pins exactly.

- `rocm-systems/`:
  - `0001-rocr-restore-legacy-doorbell-support-for-gfx803.patch`
  - `0002-clr-enable-opencl-support-for-gfx803.patch`
  - `0003-rocr-defer-gpu-va-reuse-for-gfx803.patch` -- this repo's own
    addition; not in lucbruni-amd's fork.
- `rocm-libraries/`:
  - `0001-rocblas-route-small-gfx803-gemms-off-broken-assembly.patch`
  - `0002-miopen-scope-fused-winograd-to-genuine-3x3-on-gfx803.patch`
  - `0003-miopen-fix-reducecalculation-prod-identity.patch`
  - `0004-rocblas-wgm-miscompute-source-fix.patch` -- see below.

### wgm-miscompute: source-level fix, replaces the sed, hardware-verified

`../patches/rocblas/wgm-miscompute.sh` fixes the WGM8 assembly-kernel
miscompute (see that script's own header for the hardware measurements)
via a `sed` rewrite of every `*.yaml` under Tensile's `Logic/` tree -- not
`git am`-able, and brittle against any future reorganization of that
tree. Tensile is vendored in `rocm-libraries` at `shared/tensile` (not
pip-fetched, despite what `rocblas/CMakeLists.txt`'s deprecated fallback
path suggests). `Tensile/SolutionStructs.py`'s `assignDerivedParameters()`
-- the normalization pass every solution's config dict goes through --
already does `WorkGroupMapping` normalization right above where
`0004-rocblas-wgm-miscompute-source-fix.patch` adds a targeted,
`ISA == (8, 0, 3) && KernelLanguage == "Assembly"`-gated override: a
~15-line, single-file, `git am`-able patch instead of a whole-tree sed.

**A same-pin static comparison initially looked like a regression**: a
real `rocblas-builder` build of the root `rocm7` line (same source patch,
same pinned commit) showed 518 `_WGM8` kernel-name occurrences remaining
in `TensileLibrary_Type_*_fallback_gfx803.{hsaco,dat}`, where the sed
leaves zero. Investigated rather than treated as disqualifying: every one
of those 518 names carries `ISA000_KLS` (Tensile's HIP-*source*
kernel-language marker), not `ISA803_KLA` (assembly) -- exactly the
kernel class `wgm-miscompute.sh`'s own header already measured as correct
regardless of WGM value ("WGM8 source kernels: 2/2 correct... the swizzle
is emitted by the compiler rather than by Tensile"). **Confirmed directly
on real gfx803 hardware** (RX 470, 192.168.1.214): `rocblas_dgemm`
(`TensileLibrary_Type_DD_*_fallback`) and `rocblas_zgemm`
(`..._ZZ_*_fallback`) both correct to ~1e-16 relative error across
multiple shapes, and zero `_WGM8` kernels remain anywhere *outside* the
fallback libraries -- i.e. the actual `ISA803_KLA` kernels this patch
targets are clean. The static symbol count was measuring an irrelevant
naming artifact on already-correct kernels, not a real gap; the sed's
"whole tree, zero WGM8 anywhere" approach was over-fixing (also touching
already-fine source kernels), not under-fixing.

**The gfx803-vs-architecture-general scoping question from earlier in
this file is still open** -- the hardware test above answers "does the
ISA-gated patch actually work," not "is the underlying bug
gfx803-specific." That still needs the AGENTS.md step-3 differential test
(same WGM!=1 assembly codegen path, a second GCN-family card, identical
source pin) before this patch is proposed anywhere as more than a local
gfx803 fix. Working assumption per Schaka (gfx803-specific is the more
likely explanation, given how much more real-world/CI exercise
gfx900/gfx906/etc. get) still stands, unconfirmed.

## Known gaps (this first pass)

- **`mlir-stub-missing-symbols` and
  `gfx-default-rocblas-hipblaslt-off-build-failure`** (both target
  MIGraphX, a standalone repo outside TheRock's submodule tree) are
  applied the existing way in `migraphx-builder`, under
  `patches-migraphx/`, unchanged from `../patches/migraphx/`.
- **No real-hardware validation yet.** Building clean is not the same
  claim as correctness -- see `AGENTS.md`'s "a patch that applies clean has
  confirmed nothing about correctness." `verify.py` needs to actually run
  against a built image on a real gfx803 card before any of these patches
  can be called confirmed under this pipeline.

## Building

```sh
docker build -f therock-experimental/Dockerfile -t gfx803:therock-experimental therock-experimental/
```

Long build (full ROCm-library stack from source, plus MIGraphX/PyTorch/
ORT) -- expect it to take at least as long as the root line's build.

## Verifying on hardware

Same shape as the root line's verification: transfer the built image to
the gfx803 box and run `verify.py` inside a container started with
`--device=/dev/kfd --device=/dev/dri --group-add video`.

```sh
docker save gfx803:therock-experimental | ssh user@<gfx803-host> podman load
ssh user@<gfx803-host> podman run --rm -it \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $PWD/verify.py:/verify.py \
    gfx803:therock-experimental /ort-venv/bin/python3 /verify.py
```
