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
- The three lines (`rocm10`/root, `rocm7.14/`, `rocm6.4.4/`) are
  independent copies on purpose. Do not make one line reference another's
  `patches/` -- the root, `rocm7.14/` and `rocm6.4.4/` each carry their own
  patch set, all under active development, and a shared file risks a
  bug-in-progress on one reaching the hardware-verified state of the
  other. Copy, don't link, until the deliberate convergence step (see
  README's "Convergence" section) is actually happening. `tools/` is the
  deliberate exception: it's hardware/arch-level tooling (host-setup,
  correctness-suite), not version-specific, so it lives once at the root
  and all three lines use it from there.

## Component ref pinning -- branches, not commit SHAs, no nightlies

Every upstream component this repo builds from source (`ROCM_SYSTEMS_REF`,
`ROCM_LIBRARIES_REF`, `MIGRAPHX_REF`, `PYTORCH_REF`, etc. in the Dockerfile)
is pinned to a named release *branch* -- `release/therock-10.0`,
`release/rocm-rel-10.0`, and so on -- not a frozen commit SHA and not a
per-run resolution of `develop`/`main`. Same convention the mainline
(`rocm-migraphx-ort-builder`) repo's `release.yml` uses. (The `rocm7.14/`
line uses the corresponding `release/therock-7.14` / `release/rocm-rel-7.14`
branches.)

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

## vLLM support lives in vllm/ as a hard fork -- not patch files, not a submodule

`vllm/` in this repo is a **hard fork**: upstream gfx906 vLLM
support, taken and adjusted for gfx803, tracked directly by this repo's own
git history as of 2026-08-29. It started life as a separate checkout with its
own `.git` (remote `ai-infos/vllm-gfx906-mobydick`) and was deliberately
un-forked from that -- `.git` removed, no `.gitmodules`, no gitlink, no
independent history, no relationship to that upstream repo going forward.
This was an explicit, discussed decision (not a submodule, not "local
only either way" -- fully folded into `rocm-gfx803`'s own tracking) made
on the reasoning that this fork will never be upstreamed, so there's no
value in maintaining a separate history to eventually reconcile. Don't
`git submodule add` it or try to reconnect it to that remote without
being told to.

The fork lives at the **10.0 root line** (copied there when the 7.14 line
was archived under `rocm7.14/`, 2026-08-29; the pristine fork copy stays
under `rocm7.14/vllm/`, untouched, for archival) and **targets the 10.0
stack by assumption**: the hand-written gfx803 kernels
(`vllm/vllm/gfx803_kernels/*.hip`) are version-agnostic source compiled
once with the stack's own `hipcc --offload-arch=gfx803` (see each loader's
docstring for the exact invocation), and `librocblas.so` resolves through
the stack's `LD_LIBRARY_PATH` (`/opt/rocm/core-10.0/lib` on 10.0). The
compiled `.so` files are built on the box next to their loaders and never
committed, so nothing stack-specific is pinned in this repo. Hardware
validation of vLLM on the 10.0 stack is DONE (2026-09-02): the two
crashes that blocked it were both in the ROCm 10.0 stack, not vLLM, and
are fixed by `patches/rocm-systems/va-reuse-defer-noremap.patch` (the
va-reuse-defer park-branch `_fmm_map_to_gpu` re-map left a kernel GPUVM
mapping behind that libhsakmt's aperture allocator then re-handed out, so
every code-object load collided with it -- kernel EINVAL ->
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` -> the HIP launch fallback's
reinterpret_cast turned that into a SIGSEGV) and
`patches/rocm-systems/d2h-null-dsthost.patch` (a D2H copy into a
host-accessible *device* allocation handed `readBuffer` a NULL host
destination because `getHostMem()` is unset for device-origin memory).
Verified with `qwen35_2b_bench_v3.py` on the box: EXIT=0, prefill 311.0
tok/s, decode 30.2 tok/s (2101-token prompt / 128 decode tokens; the
7.14 record for the same bench: 331.7 / 24.4).

**It is not built from this repo's Dockerfile** (vLLM runs as a box-only
editable install), and it is **not documented via `.patch.md` files under
`patches/vllm/`** -- that convention existed once, for a version of this
repo that didn't vendor vLLM at all, and was deliberately removed
(commit `a8485b4`, "[Build] Bring working vLLM in"). It must not be
revived now either. Whatever gets figured out for gfx803 in vLLM goes
directly into the real source files in `vllm/`, the same way any
other fix in this repo lands in real code, not a doc describing a
hypothetical patch.

**Gotcha: `vllm/` is double-nested.** `vllm/` itself is the fork's
own root (has its own `AGENTS.md`, `README.md`, `benchmarks/`, etc. --
`vllm/AGENTS.md` is upstream's *own* contribution-policy doc, about
submitting PRs to `vllm-project/vllm`; irrelevant here, ignore it, we are
never upstreaming); the actual importable `vllm` package is one level
deeper, at `vllm/vllm/` (`vllm/vllm/model_executor/
layers/...`, `vllm/vllm/v1/attention/ops/...`). Writing to
`vllm/model_executor/...` instead of
`vllm/vllm/model_executor/...` silently lands nothing (or the
wrong thing) and is easy to do by accident if the shell's cwd has drifted
(this tool's working directory persists between commands in a session) --
verify with an absolute path, not a relative guess, when in doubt.

**Hand-written gfx803 HIP kernel source lives in `vllm/vllm/
gfx803_kernels/`** (`gfx803_gemm_lib.hip`, `gfx803_attn_split.hip`), a
dedicated folder, not alongside the Python loaders that `ctypes`-load their
compiled output. `vllm/.gitignore` blanket-excludes `*.hip`
repo-wide with the comment "hip files generated by PyTorch" -- true for
upstream's actual hipify output (auto-translated from `.cu` CUDA source as a
build step) but wrong for these two, which are hand-written originals with
no `.cu` to regenerate from. `gfx803_kernels/` is explicitly whitelisted
back (`!/vllm/gfx803_kernels/*.hip` in `vllm/.gitignore`) instead of
routinely force-adding files past the blanket rule -- put any new
hand-written gfx803 `.hip` kernel source there and it tracks normally. The
compiled `.so` still needs to land next to its Python loader (e.g.
`vllm/vllm/model_executor/layers/libgfx803gemm.so`) for that
loader's `__file__`-relative ctypes path to find it -- only the source
moved, the `hipcc -o` output path did not; see the loader's own docstring
for the exact compile invocation.

**Syncing box-only work back**: real gfx803 vLLM fixes get iterated and
verified live on the box at 192.168.1.214 (`/data/vllm-mobydick/`, see
"Hardware access" below), then the actual changed/new files get copied
back into this local `vllm/vllm/...` checkout so the real diff
lands here -- not summarized into a separate document. The usual
commit/push rule applies same as everywhere else in this repo: build and
stage locally, never commit or push without being told to for that
specific commit.

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
