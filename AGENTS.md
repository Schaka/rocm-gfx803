# Agent instructions for rocm-gfx803

Read `README.md` first for repo layout and status. This file is
process/judgment guidance for working in this repo specifically -- the
parts that aren't obvious from the code or the patch headers alone.

## Standing philosophy

- **Build locally, never push.** All work in this repo builds locally (the
  host or the box at 192.168.1.214) -- never trigger remote CI, never push,
  never commit, unless the user explicitly instructed that specific commit
  or push. Committing and pushing are the user's tasks. A remote workflow
  run builds whatever is on the remote branch at that moment, so pushing is
  a prerequisite to any remote build anyway -- but the rule is simpler than
  that: don't. Build locally, verify locally, hand the user the diff or the
  local build, and let them decide about commit/push. This includes `gh
  workflow run` -- dispatching a remote run without being told is the same
  violation as pushing.
- **Fix at the source. No workarounds.** A gfx803 bug gets fixed where it
  actually lives -- the broken Tensile logic, the broken MIOpen solver, the
  broken MIGraphX pass -- not papered over with a retry loop, a narrower
  input-shape gate that avoids triggering it, a fallback to a slower-but-
  working path chosen defensively without knowing if the fast path is
  actually broken, or a try/catch that swallows the failure. If you can't
  fix it at the source (upstream code you don't control, or a genuine
  hardware limitation), say so explicitly and stop -- don't ship a
  workaround dressed up as a fix. `wgm-miscompute.sh` is the model:
  root-caused down to the exact broken instruction sequence, fixed by
  correcting the actual parameter Tensile computes wrong, not by avoiding
  the shapes that expose it.
- **Correctness costs performance sometimes -- that's fine, up to a point.**
  5-10% throughput lost to a correctness fix is an acceptable, expected
  trade, not something to negotiate around. Past that, flag it explicitly
  and ask rather than silently accepting a large regression or silently
  picking a workaround to avoid it. Never trade correctness for speed
  without saying so out loud first.
- **Comments say WHY, never WHAT.** Code should be legible enough that the
  WHAT doesn't need restating in prose next to it -- a comment that
  describes what the next three lines do is dead weight the moment the
  code changes and stops matching it. Write a comment only when there's a
  non-obvious reason behind the code: why this bound, why this order, why
  this workaround-shaped thing is actually correct here, what broke last
  time someone tried the obvious approach. This repo's patch headers are
  the reference model -- WHY (how the bug was found, what it looks like)
  before WHAT (the actual diff).
- **Comments are not commit history.** Don't write "changed X to Y",
  "added this for the Z fix", "removed unused W" -- that's what `git log`
  and `git blame` are for, and it rots the moment the described change is
  no longer the most recent one. A comment should make sense read cold, by
  someone with no idea what the last edit was.
- **Never comment on absence.** Don't write a comment explaining that
  something *isn't* there, *used to be* there, or *isn't being done* --
  "no libomp-dev here", "removed the X workaround", "we don't do Y
  anymore". A reader sees only the code that exists; a note about code
  that doesn't exist is unverifiable noise the moment they check, and dead
  weight forever after. If a line was dropped, dropping it needs no
  comment -- the absence speaks for itself. Only write a comment when it
  justifies something *present*.
- **Comments don't describe cross-component architecture.** If a comment
  needs to explain how this file relates to three other files, why a
  particular Dockerfile stage exists in the overall pipeline, or how the
  patch fits into the broader gfx803-vs-mainline story, that belongs in
  `README.md`/`MIGRATION_NOTES.md`/a patch header -- not in an inline code
  comment. A code comment's job is to explain the code immediately around
  it, not to teach the reader the whole system. If you're tempted to write
  three paragraphs above a function about how the pipeline works, that's a
  docs edit, not a code comment.

## Why this repo exists, and why it matters for how you work here

gfx803 (Polaris) is unsupported upstream since ROCm 6.0. Every fix here is
local -- nothing gets upstreamed to AMD, nothing gets fixed by a ROCm
version bump unless you go check. That has two consequences for how to
approach work in this repo:

1. **Assume nothing is fixed until you've checked the current pinned
   source.** A patch's own header saying "confirmed broken as of ROCm X"
   is a snapshot, not a permanent fact. Before re-diffing or investigating
   further, grep the current pinned commit for the target function/struct
   and read what's actually there now -- this repo's own history has
   examples both ways: patches that turned out to already be obsolete
   (the fix landed upstream on its own) and patches whose target code was
   *replaced* by something with unknown, unverified behavior on the same
   bug class (not fixed, not necessarily still broken -- genuinely
   unknown until checked).
2. **A patch that applies clean has confirmed nothing about correctness.**
   The recurring bug class on this hardware is silent miscompute --
   `rocblas_status_success` with wrong numbers, a kernel that dispatches
   fine and returns garbage. Re-diffing a patch so it compiles again is
   necessary but not sufficient; say so explicitly ("NOT YET RE-VERIFIED
   ON REAL HARDWARE") in the patch/notes until someone actually re-runs
   the original repro against the new binaries, not just confirms the
   diff applies.

## Investigation workflow (what's worked repeatedly in this repo's history)

1. **Trace before you patch.** `MIOPEN_ENABLE_LOGGING_CMD=1
   MIOPEN_LOG_LEVEL=6` and `MIGRAPHX_TRACE_COMPILE=1` reveal the actual
   dispatched solver/op graph, not what you'd guess from reading the
   Dockerfile or the op name. Several "obvious" hypotheses in this repo's
   history turned out wrong once traced (a ConvTranspose investigation
   assumed a specific MIOpen solver was responsible; the trace showed a
   completely different code path -- MIGraphX's own graph rewrite, not
   MIOpen at all).
2. **Isolate before you conclude.** If a bug reproduces through the full
   stack (ORT -> MIGraphX -> MIOpen -> rocBLAS), test each layer standalone
   before assuming which one is broken -- `MIOpenDriver <op> ... -V 1` runs
   MIOpen's own GPU-vs-CPU-reference check with zero ORT/MIGraphX
   involvement and has repeatedly cleared MIOpen as a suspect in favor of a
   layer above it (or vice versa).
3. **Differential test against a different architecture before assuming
   something is gfx803-specific.** A failure that also reproduces on a
   modern arch (gfx900+/gfx12x) under the exact same source pin is an
   upstream/generic bug, not gfx803's to fix here -- report it upstream
   instead. A failure that's unique to gfx803 under identical source is
   fair game for a local patch. Get this distinction right *before*
   writing a patch, not after -- a gfx803-only patch for a generic bug
   fixes nothing for anyone else hitting the same code path, and isn't
   this repo's job.
4. **Ablation-test before crediting a patch.** If several patches are
   candidates for fixing an observed symptom, build a variant with each
   disabled (keep everything else identical) and compare -- this repo's
   history has a case where two rocBLAS patches were confirmed *partially*
   responsible for a symptom (removing them made it much worse, not just
   "no different"), which is a different and more useful finding than
   either "yes" or "no."
5. **When something works on one line but not the other, check whether
   it's actually comparable before calling it a regression.** New ORT/
   ROCm/MIGraphX versions add real new functionality (new ONNX opsets, new
   parser features) that the older line literally could not have
   exercised -- a test failing on the new line but not existing/loadable
   on the old one isn't a regression, it's new-and-still-buggy. Only count
   something as "broken here, worked there" once you've confirmed the
   older line actually ran the same code path and got it right.

## Patch conventions

- Every patch header states WHY (how the bug was found, what it looks
  like, hardware measurements if applicable) before WHAT. If you can't
  write the WHY convincingly, you haven't finished the investigation.
- Two apply dialects, on purpose: `git apply` for anything cloned as a
  real git repo (`rocm-systems`); `patch -p1` for anything that's a
  sparse-checked-out monorepo subdirectory, because `git apply --check`
  silently no-ops ("Skipped patch", exit 0, nothing modified) on those
  in this box's git version. Match whichever dialect the sibling patches
  in that directory already use.
- Every `.sh` driver verifies its own result after applying -- greps for a
  marker string, fails loudly (`exit 1`) if it's missing. Don't add a
  driver that trusts the patch tool's exit code alone.
- The two lines (`rocm7`/root and `rocm6.4.4/`) are independent copies on
  purpose. Do not make one reference the other's `patches/`/`tools/` --
  both are under active development and a shared file risks a
  bug-in-progress on one reaching the hardware-verified state of the
  other. Copy, don't link, until the deliberate convergence step (see
  README's "Convergence" section) is actually happening.

## Component ref pinning -- branches, not commit SHAs, no nightlies

Every upstream component this repo builds from source (`ROCM_SYSTEMS_REF`,
`ROCM_LIBRARIES_REF`, `MIGRAPHX_REF`, `PYTORCH_REF`, etc. in the Dockerfile)
is pinned to a named release *branch* -- `release/therock-7.14`,
`release/rocm-rel-7.14`, and so on -- not a frozen commit SHA and not a
per-run resolution of `develop`/`main`. Same convention the mainline
(`rocm-migraphx-ort-builder`) repo's `release.yml` uses.

This works only because CI here is manual-dispatch only, with no schedule --
there's no nightly job re-running against a moving branch tip unattended. A
branch pin means "build whatever's on that branch the day someone runs the
workflow"; two manual runs weeks apart can legitimately land different
commits if upstream pushed a cherry-pick to the branch in between. That's
expected, not drift to chase down -- if a build starts failing that
previously passed and nothing in this repo changed, check the branch's
current tip against what the last successful build actually used before
assuming a local regression.

If you ever add a component that has no such release branch (rocBLAS/MIOpen
before the `rocm-libraries` monorepo restructure, or a component whose
upstream only tags releases rather than branching them), pin that one to an
exact commit SHA instead and say so explicitly in the Dockerfile comment --
don't default to `develop`/`main` to avoid the question.

## Hardware access

Real-hardware validation requires the actual gfx803 card -- this cannot be
emulated or approximated on a different GPU. If you don't have access to
one, say so explicitly rather than reporting a patch as "verified" based on
a clean build/apply alone. Container needs `--device=/dev/kfd
--device=/dev/dri --group-add video` passed through; `rocminfo` inside the
container should enumerate the card as a real `KERNEL_DISPATCH` agent
before trusting anything else it reports.

Known device-side pitfalls when instrumenting on real hardware: device-side
`printf` can hang the kernel rather than just being slow; adding *any* extra
device-side write for debugging can itself mask a race you're trying to
observe (changes timing); always verify you're actually running against the
patched binary you think you are (a stale image/container is a recurring
false lead), not just that the patch file on disk looks right.

## What doesn't need the hardware

Whether a Dockerfile builds, whether a patch applies against a given pin,
source-level tracing of where a bug lives (reading the actual pinned
source, diffing across ROCm versions, following a compiler pass through
its actual invoking code rather than assuming), and setting up cross-arch
differential tests (the test itself needs a card of *some* kind, but not
necessarily gfx803, to establish "is this generic or gfx803-specific"
before spending real-hardware time on the gfx803 side of that comparison).
