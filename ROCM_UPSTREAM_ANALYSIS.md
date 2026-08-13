# AI generated initial analysis - take with grain of salt

# Upstream feasibility analysis: gfx803 in TheRock/ROCm 7.14

Written from the perspective of assessing this repo's patch set against
AMD's actual upstream process, prompted by
https://github.com/ROCm/clr/issues/269#issuecomment-5240585153 (Schaka
asking lucbruni-amd, AMD, whether restoring gfx803 support is worth
pursuing upstream; lucbruni-amd's answer: break the patches into small
chunks, target TheRock directly, let maintainer feedback decide).

## TL;DR

- **Layer 1 (HSA agent enumeration + OpenCL device accept) is
  upstream-shaped and has a strong precedent, but is NOT upstream-reviewed
  or queued anywhere.** lucbruni-amd (AMD) has a 3-file branch on his
  *personal* fork (`lucbruni-amd/TheRock@lb/gfx803-polaris-support`,
  never PR'd against `ROCm/TheRock` -- confirmed via `gh pr list --search
  gfx803 --state all`, zero results, and no open/closed/merged PR
  references that branch) doing almost exactly what this repo's
  `hsa-agent-rejects-legacy-doorbell.patch` +
  `opencl-gfx8-hardcoded-rejection.patch` do. It's a proof-of-concept he
  built and shared in a GitHub comment while investigating issue #269, not
  something AMD has queued, reviewed, or committed to landing. The value
  is still real -- independent convergence on the same fix by someone with
  upstream context is a good signal, and it's a template for how the diff
  should look -- but "queued in a TheRock fork" overstates it. No one at
  AMD has opened a PR for this. **This is still the one to send first**,
  but sending it means *this repo* (or lucbruni-amd, if asked) opening the
  actual PR -- it doesn't exist yet.
- **Layer 2 (rocBLAS/Tensile, MIOpen correctness patches) is real,
  root-caused, hardware-verified work, but is not clean by upstream's
  bar yet** -- not because the fixes are wrong, but because they carry
  gfx803-only framing (env-gated, `-a gfx803`-only assumptions, sed
  passes over the whole Logic tree) that upstream will want re-scoped to
  "does this affect other archs on the same code path" before merging
  into shared Tensile/MIOpen logic. WGM and small-GEMM assembly
  miscompute look architecture-general on their face (broken assembly
  codegen, not a gfx803 hardware quirk) and are the best next candidates
  to break out.
- **This repo does not currently build through TheRock's own build
  system** -- it clones `rocm-systems`/`rocm-libraries`/MIGraphX/PyTorch
  individually inside a hand-rolled multi-stage Dockerfile and applies
  patches with custom `.sh` drivers, pinned to the same release branches
  TheRock uses but not invoked via `fetch_sources.py --patch-tag` /
  `cmake -DTHEROCK_AMDGPU_FAMILIES=...` / `ninja`. Moving the base-enablement
  patches (Layer 1) into an actual `patches/gfx803/` tree structured like
  lucbruni-amd's fork, and building through TheRock's own tooling for at
  least a validation pass, is both the fastest way to get from "applies
  against our Dockerfile" to "AMD can `git apply` this into TheRock and
  it just works," and the honest way to test whether it still works
  under TheRock's actual orchestration (not just this repo's copy of the
  same source pin).
- **gfx900 is the right target, not the right template.** gfx900 shows
  `Build Passing` on `SUPPORTED_GPUS.md` but not `Sanity Tested` or
  `Release Ready` -- i.e. AMD's own bar for a *newly restored* legacy
  arch is "produces a wheel/tarball," not "someone ran it on hardware."
  gfx803 achieving the same status is a realistic, scoped ask: it does
  not currently appear in `SUPPORTED_GPUS.md` at all (GCN4/Polaris has no
  row), so the first upstream milestone is literally adding a row and
  getting it to ✅ Build Passing -- exactly what lucbruni-amd's 3-file
  branch already does for HIP+OpenCL. This repo's hardware validation
  (real `KERNEL_DISPATCH` agent, real GPU matmul/conv/MIGraphX
  compile+run, ORT's own 3828-test operator suite) already clears the
  *Sanity Tested* bar this repo could claim for itself, which is more
  than gfx900 currently has -- worth stating explicitly in any upstream
  pitch, since it's the strongest asset this repo has that a bare patch
  diff wouldn't communicate.

## What "upstream" actually means here, concretely

Two different upstream targets, not one:

1. **TheRock** (`ROCm/TheRock`) -- the build-orchestration repo. This is
   where `SUPPORTED_GPUS.md`'s "Build Passing" column comes from, and
   it's the one lucbruni-amd explicitly pointed at. It doesn't carry
   library source itself; it clones/pins `rocm-systems`, `rocm-libraries`,
   MIGraphX, etc. and applies `patches/<tag>/` trees keyed by
   `--patch-tag`. Getting gfx803 accepted here first, at the
   HIP/OpenCL-enablement layer, is the low-risk on-ramp: it's additive
   (a new `therock_add_amdgpu_target(gfx803 ...)` line + a new patch
   directory), doesn't touch any other architecture's code path, and has
   a working precedent already on an AMD engineer's own fork.
2. **The component repos themselves**
   (`ROCm/rocm-systems`/`rocm-libraries` monorepos, standalone rocBLAS/
   MIOpen/MIGraphX history before the monorepo restructure) -- this is
   where the correctness patches (WGM, GSU zeroing, small-GEMM assembly,
   Winograd, reduce-prod identity, MLIR stub symbols) would ultimately
   need to land, since they touch Tensile logic / MIOpen solvers /
   MIGraphX passes that TheRock only orchestrates, doesn't own. These are
   harder sells: they're bug fixes in shared code that historically
   *was* built for gfx803 and (per this repo's own investigation
   history) sometimes affects other GCN-family architectures too, which
   raises the review bar from "does this help Polaris" to "does this
   regress anything else on the same kernel."

Sending Layer 2 patches at TheRock directly (as lucbruni-amd's advice
could be read) is the wrong repo for most of them -- TheRock has no
Tensile/MIOpen solver source to patch. The actual routing should be:
TheRock PR for target enablement (Layer 1), separate PRs against
`rocm-systems`/`rocm-libraries` for each correctness fix (Layer 2), once
each is re-scoped per architecture-generality below.

## Layer 1: HSA/OpenCL enablement -- ready now

Diff comparison, this repo vs. lucbruni-amd's fork (both against the
same `therock-7.14`-family pin):

| This repo | lucbruni-amd/TheRock | Verdict |
|---|---|---|
| `hsa-agent-rejects-legacy-doorbell.patch` (255 lines, `rocm-systems`) | `0001-rocr-Restore-legacy-doorbell-support-for-gfx8-Polari.patch` | Same bug, same fix target (`amd_gpu_agent.cpp` doorbell-type gate). Independently converged. |
| `opencl-gfx8-hardcoded-rejection.patch` (257 lines, `rocm-systems`) | `0002-clr-Enable-OpenCL-support-for-gfx8-Polaris-GPUs.patch` | Same `runtimeRocSupported()` GFX8 hardcode, same image-query-abort fix. |
| `va-reuse-defer.patch` (368 lines, `rocm-systems`) | *(not present)* | This repo's own addition -- lucbruni-amd's branch gets HSA enumeration + OpenCL working but does not carry the VA-reuse fix this repo found necessary for MIOpen CK reduction-kernel correctness under sustained use. Worth offering as a follow-up once the base two are through review, not bundled with them (it's a different, less obviously-safe class of fix -- see below). |

lucbruni-amd's branch also adds the missing piece this repo doesn't need
to (because this repo doesn't build through TheRock): a
`therock_add_amdgpu_target(gfx803 "Polaris / RX 550-580" FAMILY dgpu-all
gfx803-dgpu ...)` line in `cmake/therock_amdgpu_targets.cmake`, one line
below the existing `gfx900` entry. That's the actual "make gfx803 a real
TheRock target" step -- this repo's Dockerfile never needed it because it
builds each component's `cmake`/`rmake.py` directly with `-a gfx803`,
bypassing TheRock's target-family machinery entirely.

**Recommendation**: this repo's two doorbell/OpenCL patches are already
functionally redundant with an AMD engineer's own branch. Rather than
PRing a third, slightly-different version, the highest-leverage move is
to (a) validate this repo's hardware-verified numbers *against
lucbruni-amd's exact branch* (does his version, unmodified, also survive
real dispatch/matmul/conv on the RX 470? if yes, that's independent
confirmation from a second card, which is exactly the kind of evidence
that unblocks a merge) and (b) offer the `va-reuse-defer` patch as an
addition to his branch/PR rather than a competing one, with the
"NOT AUTOMATICALLY FIXED, confirmed against pinned source" framing this
repo's patch headers already have.

## Layer 2: correctness patches -- root-caused, not yet upstream-shaped

These correctness patches (WGM, small-GEMM assembly, Winograd,
reduce-prod-wrong-identity, MLIR-stub) share real strengths that most
upstream contributions don't arrive with:

- Every one states hardware measurements (dose-response tests, N/N
  solution-matrix sweeps, before/after correctness counts on real
  silicon), not just "this fixed it for me."
- Every one identifies the exact broken mechanism (a specific Tensile
  codegen path, a specific memory-reuse assumption, a specific compiler
  stub), not a symptom-level workaround.
- Several explicitly re-verified against the new pin rather than
  assuming the 6.4.4-era finding still holds (`va-reuse-defer`'s
  "RE-DIFF NOTE" / "NOT AUTOMATICALLY FIXED" section is exactly the kind
  of diligence a reviewer would otherwise have to redo themselves).

What would block a clean merge as-is:

1. **Scope is stated as gfx803-only where the underlying bug may not
   be.** `wgm-miscompute.sh`'s own header says the WGM swizzle assembly
   bug is a Tensile codegen defect ("broken assembly, not a gfx803
   hardware quirk"), and `small-gemm-assembly-miscompute` reads the same
   way. AGENTS.md's own workflow step 3 ("differential test against a
   different architecture before assuming something is gfx803-specific")
   is exactly the missing step before these two are upstream-shaped --
   if the same WGM=8 assembly path exists and is wrong on, say, gfx900 or
   gfx906 (same GCN family, same Tensile assembly backend generation),
   this is a Tensile bug report + fix, full stop, unscoped to any one
   architecture, and a much easier sell than a gfx803-only patch. If it
   turns out gfx803's specific register/ISA quirk is the actual cause
   (plausible -- GCN3 has real ISA differences from GCN5), that's the
   thing to state explicitly in the patch header before sending it, the
   same way `hsa-agent-rejects-legacy-doorbell.patch` explicitly frames
   itself as gfx803-specific doorbell hardware behavior.
2. **The sed-based WGM patch (`wgm-miscompute.sh`, no `.patch` file) is
   not a *diff* upstream can review line-by-line** -- it's a
   whole-tree rewrite driven by a script, which is the right call for
   *this repo's* build (documented well in its own header: unbounded
   file count, no fixed insertion point) but is the wrong shape for a
   PR. Upstream will want either a targeted Tensile-generator-level fix
   (change whatever emits WGM8 solutions in the first place, one file)
   or, if the yaml-rewrite approach is genuinely necessary, a proper
   diff generated from actually rewriting the files rather than a sed
   invocation described in prose.
3. **No CI-visible regression evidence for other architectures.** A
   Tensile/MIOpen maintainer's first question on any of these will be
   "does this change behavior for gfx900/gfx906/gfx942/etc," since these
   are shared logic files, not gfx803-gated code paths the way the CLR/
   ROCR doorbell patches are (those are behind an explicit doorbell-type
   check that only ever matches Polaris hardware). Before sending Layer
   2 upstream, each patch needs either (a) a build/test showing it's a
   no-op for solutions selected on other architectures, or (b) same-day
   differential numbers on at least one other GCN-family or RDNA card
   confirming the bug (and the fix) generalizes, following this repo's
   own AGENTS.md investigation workflow step 3.

**Recommendation**: don't bundle Layer 2 with the Layer 1 PR. Pick WGM
and small-GEMM-assembly first (most likely to be genuinely
architecture-general, highest payoff if confirmed), do the cross-arch
differential test AGENTS.md already prescribes, and only then open
issues/PRs against `rocm-libraries` framed as "Tensile assembly codegen
bug, reproduces on gfx803, may affect others" rather than "gfx803 fix."

## Should this repo move (more) of its build onto TheRock?

Partially, and there's a real distinction to draw here between two
different things "using TheRock" could mean:

- **Already true**: this repo pins to TheRock's own release branches
  (`release/therock-7.14` for `rocm-systems`/`rocm-libraries`,
  `release/rocm-rel-7.14` for MIGraphX) -- same source TheRock itself
  would build. The Dockerfile's own comments are explicit about
  following the same branch-pin convention `rocm-migraphx-ort-builder`'s
  `release.yml` uses for `MIGRAPHX_REF`. So the *source* is already
  TheRock-aligned.
- **Not true yet**: the *build orchestration* is this repo's own
  hand-rolled multi-stage Dockerfile (manual `git clone` + `cmake`/
  `rmake.py` invocations per component with `-a gfx803`), not TheRock's
  `fetch_sources.py --patch-tag gfx803` + `cmake
  -DTHEROCK_AMDGPU_FAMILIES=gfx803-dgpu` + `ninja` pipeline. These are
  not guaranteed equivalent -- TheRock's build applies its own patch
  sets, its own dependency ordering, its own packaging step, none of
  which this repo currently exercises.

The concrete value of switching (or adding a second, TheRock-native
build path) is exactly the question in the prompt: **it's the only way
to find out whether this repo's patches, and the resulting correctness
guarantees, survive being built the way AMD would actually build them**,
rather than the way this repo happens to build them. Two options, not
mutually exclusive:

1. **Minimum-effort validation**: clone `lucbruni-amd/TheRock@lb/gfx803-polaris-support`
   as-is, run it through TheRock's actual `fetch_sources.py` /
   `cmake --DTHEROCK_AMDGPU_FAMILIES=gfx803-dgpu` / `ninja` pipeline on
   the same hardware this repo already validates against
   (192.168.1.214), and diff the result against this repo's own
   `rocm7` image at the HSA/OpenCL layer only (rocminfo, clinfo,
   a HIP hello-world). This confirms Layer 1 without touching rocBLAS/
   MIOpen/MIGraphX at all, and answers "does AMD's own draft actually
   work on my card" directly rather than by inference from this repo's
   independently-converged patch.
2. **Full convergence** (bigger lift): restructure this repo's patch
   application to actually route through TheRock's `patches/<tag>/`
   mechanism instead of the current per-component Dockerfile RUN steps,
   for the base-enablement layer at minimum. This is the only way to
   make the eventual upstream PR a genuine "here's the branch, here's
   the diff against TheRock main, here's the hardware log" submission
   rather than "here's a patch file we've verified applies to source
   pinned the way TheRock also pins it." It's more work and not
   necessary to get useful signal from AMD (option 1 already does that),
   so treat it as the step that follows a maintainer showing real
   interest, not a prerequisite to asking.

## Is "gfx900-equivalent" support actually achievable?

Yes, and gfx803's starting position for the *first* milestone (Build
Passing) is arguably better than "achievable" -- it's close to done:

- gfx900 (GCN5.0, Vega 10) already has ✅ Build Passing on both Linux
  and Windows in `SUPPORTED_GPUS.md`, with no ✅ in Sanity Tested or
  Release Ready. That is the entirety of what AMD currently guarantees
  for a restored legacy Instinct/Radeon architecture -- a wheel/tarball
  gets produced, nothing about runtime correctness is promised or
  checked.
- gfx803 doesn't appear in the table at all -- Polaris (GCN4) has no
  row on either the Instinct or Radeon side. The actual ask is "add a
  row and get it to the same single checkmark gfx900 has," which is
  materially smaller than "achieve parity with a currently-supported
  architecture" -- it's parity with the *bar for a freshly-restored*
  one, and lucbruni-amd's 3-file branch is most of the way to clearing
  that bar for HIP/OpenCL specifically (once someone runs TheRock's
  actual `ninja` build against it and files the PR).
- This repo can honestly claim to already be past gfx900's own bar in
  one respect: **Sanity Tested**, per `SUPPORTED_GPUS.md`'s own
  definition ("either in CI or some light form of manual QA has been
  performed"), is something this repo has real evidence for that
  gfx900 currently lacks on the public roadmap page -- real
  `KERNEL_DISPATCH` agent enumeration, real GPU dispatch across rocBLAS/
  MIOpen/MIGraphX/ORT, and a 3828-test ORT operator-suite run diffed
  against both the 6.4.4 line and a gfx1201 image. That's a genuinely
  strong opening data point for a TheRock issue/PR, worth leading with
  rather than burying in a patch description.
- The realistic ceiling to be honest about: **Release Ready** (AMD's
  own release process, CI-gated) is a different and much higher bar --
  it implies AMD taking on long-term maintenance of an architecture they
  explicitly dropped, which is a business/roadmap decision this repo's
  patch quality can't unilaterally earn no matter how clean the diffs
  are. Build Passing + community-maintained Sanity Testing (this repo's
  own correctness-suite, kept running against each new `therock-7.x`
  pin) is the realistic, honestly-scoped target -- the same tier gfx900
  currently sits at, not full parity with an actively-supported arch
  like gfx906.

## Concrete next steps, in order

1. Pull `lucbruni-amd/TheRock@lb/gfx803-polaris-support`, build it
   through TheRock's own `fetch_sources.py`/`cmake`/`ninja` pipeline
   (not this repo's Dockerfile), validate on the RX 470 the same way
   `MIGRATION_NOTES.md`'s 2026-08-09 entry already validated this
   repo's own image (rocminfo enumeration, real dispatch). Report back
   on that GitHub issue/thread with the result -- second-card
   confirmation is the single highest-value thing this repo can offer
   right now.
2. Offer `va-reuse-defer.patch` (re-diffed, already confirmed
   NOT-auto-fixed against the current pin) as a fourth patch alongside
   lucbruni-amd's two, since his branch gets enumeration+OpenCL working
   but doesn't yet carry the fix this repo found necessary for
   MIOpen CK correctness under sustained use.
3. Cross-arch differential test WGM and small-GEMM-assembly against a
   second GCN-family or RDNA card before framing either as a Tensile bug
   report; only then open issues against `rocm-libraries`.
4. Treat Winograd/reduce-prod/MLIR-stub patches as lower priority for
   upstream outreach until 1-4 land -- not because they're less correct,
   but because a first submission succeeding builds the credibility
   (and maintainer relationship) the later, harder-to-review ones will
   need.
