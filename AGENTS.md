# Agent instructions for rocm-gfx803

Read `README.md` first. It gives the layout and the status of each line. This
file gives the process rules and the judgment calls. You cannot get either one
from the code or from a patch header alone.

## Standing philosophy

### Build locally. Never push.

All work in this repo builds locally. You build on the host or on the box at
192.168.1.214. Do not trigger remote CI. Do not push. Do not commit. Do these
only when the user asks for that one commit or push. Committing and pushing are
the user's jobs. A remote workflow builds whatever the remote branch holds at
that moment, so a push is a prerequisite for it anyway. The rule is
shorter than that: do not. Build locally, test locally, and give the user the
diff or the local build. Then they decide. This rule also covers
`gh workflow run`. Starting a remote run without being asked is the same
violation as a push.

### Fix at the source. Do not use workarounds.

Fix a gfx803 bug where the bug lives. That can be the broken Tensile logic, the
broken MIOpen solver, or the broken MIGraphX pass. Do not hide it with a retry
loop. Do not hide it with a narrower input-shape rule that avoids the trigger.
Do not hide it with a fallback to a slower path that you picked because you were
afraid, without proof that the fast path is broken. Do not hide it with a
try/catch that swallows the failure. Some bugs have no source fix. The code is
upstream and you do not control it, or the hardware cannot do it. Say that in
plain words and stop. Do not ship a workaround that looks like a fix.
`wgm-miscompute.sh` is the model. It found the exact broken instruction
sequence, and it fixed the one parameter that Tensile computes wrong. It did not
avoid the shapes that show the bug.

### Correctness costs speed sometimes. That is fine, to a point.

A correctness fix that costs 5 to 10 percent of throughput is an acceptable and
expected trade. Do not negotiate around it. Beyond that, say so and ask. Do not
accept a large regression in silence, and do not pick a workaround in silence to
avoid one. Never trade correctness for speed without saying it out loud first.

### Comments say WHY. They never say WHAT.

Write code that is clear enough to read on its own. A comment that repeats what
the next three lines do is dead weight, and it becomes wrong the first time the
code changes. Write a comment only for a reason you cannot see in the code: why
this bound, why this order, why a thing that looks like a workaround is actually
correct here, and what broke the last time someone tried the obvious approach.
The patch headers in this repo are the model. Each one states WHY (how the bug
was found and what it looks like) before WHAT (the diff itself).

### A comment is not commit history.

Do not write "changed X to Y", "added this for the Z fix", or "removed unused
W". `git log` and `git blame` do that job, and such a comment becomes wrong as
soon as a later edit lands. A comment must make sense to a reader who has no
idea what the last edit was.

### Never write a comment about something that is absent.

Do not explain that something *is not* there, *used to be* there, or *is not
done*. Examples of bad comments: "no libomp-dev here", "removed the X
workaround", "we do not do Y anymore". A reader sees only the code that exists.
A reader cannot test a note about code that does not exist, so it is dead weight
forever. A dropped line needs no comment, because its absence is visible. Write
a comment only to justify something that is present.

### A comment does not describe the whole system.

Some comments need to explain how one file relates to three others, why a
Dockerfile stage exists in the larger pipeline, or how a patch fits the
gfx803-versus-mainline story. That text belongs in `README.md`, in
`MIGRATION_NOTES.md`, or in a patch header. It does not belong in an inline
code comment. A code comment explains the code next to it. If you want to write
three paragraphs above a function about how the pipeline works, that is a docs
change, not a comment.

## Why this repo exists, and what that means for your work

AMD dropped support for gfx803 (Polaris) in ROCm 6.0. Every fix here is local.
Nothing goes upstream to AMD, and a ROCm version bump fixes nothing unless you
go and look. Two results follow.

1. Assume nothing is fixed until you read the pinned source. A patch header
   that says "confirmed broken as of ROCm X" is one moment in time, not a
   permanent fact. Before you re-diff a patch or investigate again, grep the
   pinned commit for the target function or struct and read what is there now.
   This repo's history has both outcomes. Some patches were already obsolete,
   because the fix landed upstream on its own. In others the target code was
   replaced by something new whose behavior on the same bug class is unknown
   and unverified. That case is not fixed and not proven broken. It is unknown
   until you read the pinned source.
2. A patch that applies cleanly proves nothing about correctness. The common
   bug class on this hardware is a silent wrong answer. `rocblas_status_success`
   returns with bad numbers, or a kernel dispatches fine and returns garbage.
   Re-diffing a patch so it compiles again is necessary and not enough. Say so
   in the patch or the notes: "NOT YET RE-VERIFIED ON REAL HARDWARE". Keep that
   note until someone runs the original repro against the new binaries, instead
   of only confirming that the diff applies.

## Investigation workflow

These steps worked many times in this repo's history.

1. Trace before you patch. `MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6` and
   `MIGRAPHX_TRACE_COMPILE=1` show the solver and op graph that really ran. They
   do not show what you guess from the Dockerfile or the op name. Several
   "obvious" explanations here were wrong after a trace. One ConvTranspose
   investigation blamed a specific MIOpen solver. The trace showed a different
   code path: MIGraphX's own graph rewrite, and MIOpen was not involved.
2. Isolate before you conclude. When a bug shows through the whole stack (ORT to
   MIGraphX to MIOpen to rocBLAS), test each layer alone before you pick a
   suspect. `MIOpenDriver <op> ... -V 1` runs MIOpen's own GPU-versus-CPU
   reference comparison with no ORT or MIGraphX in the process. It cleared MIOpen
   in favor of a layer above it, and the other way around, more than once.
3. Compare against another architecture before you call a bug gfx803-specific.
   A failure that also happens on a newer arch (gfx900+ or gfx12x) with the same
   source pin is an upstream bug, and it is not gfx803's to fix here. Report it
   upstream. A failure only on gfx803 with identical source is a fair target for
   a local patch. Make this distinction before you write the patch, not after. A
   gfx803-only patch for a generic bug helps nobody else who hits the same code
   path, and that work is not this repo's job.
4. Turn patches off one at a time before you credit one. When several patches
   can explain a symptom, build one variant per disabled patch and keep
   everything else the same. History: two rocBLAS patches were each *partly*
   responsible. Removing them made the symptom much worse, which is a different
   and more useful result than "no difference" or "yes".
5. When one line works and the other does not, first ask whether the two are
   comparable. Newer ORT, ROCm, and MIGraphX add real new features (new ONNX
   opsets, new parser paths) that the older line cannot run at all. A test
   that fails on the new line and does not even load on the old one is not a
   regression. It is new and still broken. Count something as "worked there,
   broken here" only after you make sure that the older line ran the same code
   path and got it right.

## Patch conventions

- Each patch header states WHY before WHAT: how the bug was found, what it looks
  like, and the hardware measurements where they apply. If you cannot write a
  convincing WHY, your investigation is not finished.
- Two apply styles exist on purpose. Use `git apply` for a tree that is a real
  git repo, such as `rocm-systems`. Use `patch -p1` for a sparse checkout of a
  monorepo subdirectory, because `git apply --check` there reports success and
  changes nothing ("Skipped patch", exit 0) on this box's git version. Match the
  style that the other patches in that directory already use.
- Every `.sh` driver makes sure that its own apply worked. It greps for a marker
  string, and it stops with `exit 1` when the marker is absent. Do not add a driver
  that trusts only the exit code of the patch tool.
- The three lines (`rocm10` at the root, `rocm7.14/`, and `rocm6.4.4/`) are
  separate copies by design. Do not make one line point at another line's
  `patches/`. Each one carries its own patch set, and a shared file risks a
  half-fixed bug on one line reaching the hardware-tested state of another.
  Copy, do not link, until the planned merge in README's "Convergence" section
  actually happens. `tools/` is the one exception. It holds hardware and arch
  level tooling (host-setup, correctness-suite) rather than version-specific
  code, so it lives once at the root and every line uses it from there.

## Component pins: branches, not commit SHAs, and no nightlies

Every upstream component that this repo builds from source is pinned to a named
release branch. The Dockerfile args are `ROCM_SYSTEMS_REF`,
`ROCM_LIBRARIES_REF`, `MIGRAPHX_REF`, and `PYTORCH_REF`, and the values are
`release/therock-10.0`, `release/rocm-rel-10.0`, and similar. Do not pin a
frozen commit SHA, and do not resolve `develop` or `main` per run. The mainline
repo (`rocm-migraphx-ort-builder`) uses the same convention in its
`release.yml`. The archived `rocm7.14/` line uses `release/therock-7.14` and
`release/rocm-rel-7.14`.

This is safe only because CI here runs on manual dispatch and has no schedule.
No nightly job re-runs against a moving tip while nobody watches. A branch pin
means: build what that branch holds on the day a person runs the workflow. Two
manual runs weeks apart can land different commits, because upstream can push a
cherry-pick to the branch in between. That is expected. It is not drift to chase. If
a build starts to fail and nothing in this repo changed, compare the branch tip
with the commit that the last good build used, before you suspect a local
regression.

Two additions to that rule matter for how the build finds the tip. CI resolves
each branch to its commit once per run and passes it as a `*_SHA` build-arg, so
a moved tip changes the layer cache key instead of being reused invisibly. See
"Component images, pins and line provenance" in `README.md`. Nobody sets these
values by hand, and the Dockerfile keeps the branch names as the readable pin.

If you add a component that has no release branch, pin that one to an exact
commit SHA and say why in the Dockerfile comment. This happens for a component
whose upstream only tags releases, or for rocBLAS and MIOpen before the
`rocm-libraries` monorepo restructure. Do not answer the question by defaulting
to `develop` or `main`.

## vLLM lives in vllm/ as a hard fork, not as patch files and not as a submodule

`vllm/` in this repo is a hard fork. It carries upstream gfx906 vLLM support,
adjusted for gfx803, and this repo's own git history tracks it directly as of
2026-08-29. It began as a separate checkout with its own `.git` and a remote
named `ai-infos/vllm-gfx906-mobydick`. That link was removed on purpose: no
`.git`, no `.gitmodules`, no gitlink, and no independent history. The decision
was discussed and explicit. The fork will never be upstreamed, so a separate
history to reconcile later has no value. Do not run `git submodule add` on it,
and do not reconnect it to that remote unless you are told to.

The fork lives on the 10.0 root line. It was copied there when the 7.14 line was
archived under `rocm7.14/` on 2026-08-29. The untouched fork copy stays under
`rocm7.14/vllm/` for the record. The fork targets the 10.0 stack by assumption.
Its hand-written gfx803 kernels in `vllm/vllm/gfx803_kernels/*.hip` are
version-agnostic source. They compile once with the stack's own
`hipcc --offload-arch=gfx803`, and each loader's docstring gives the exact
command. `librocblas.so` resolves through the stack's `LD_LIBRARY_PATH`, which
is `/opt/rocm/core-10.0/lib` on 10.0. The compiled `.so` files are built on the
box next to their loaders and are never committed, so this repo pins nothing
stack-specific.

Hardware validation of vLLM on the 10.0 stack is done, on 2026-09-02. Two
crashes blocked it, and both were in the ROCm 10.0 stack rather than in vLLM.
`patches/rocm-systems/va-reuse-defer-noremap.patch` fixes the first one. The
va-reuse-defer park branch re-mapped a buffer in `_fmm_map_to_gpu` and left a
kernel GPUVM mapping behind. libhsakmt's aperture allocator handed that range
out again, so every code-object load collided with it. The kernel load returned
EINVAL, that became `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, and the HIP launch
fallback's reinterpret_cast turned it into a SIGSEGV.
`patches/rocm-systems/d2h-null-dsthost.patch` fixes the second one. A D2H copy
into a host-accessible *device* allocation passed `readBuffer` a NULL host
destination, because `getHostMem()` is not set for memory that came from the
device. Verified on the box with `qwen35_2b_bench_v3.py`: EXIT=0, prefill 311.0
tok/s, decode 30.2 tok/s, on a 2101-token prompt and 128 decode tokens. The
7.14 record for the same bench is 331.7 and 24.4.

The fork is not built by this repo's Dockerfile. vLLM runs as a box-only editable
install. It is also not documented with `.patch.md` files under `patches/vllm/`.
That convention belonged to an older version of this repo that did not vendor
vLLM at all, and commit `a8485b4` ("[Build] Bring working vLLM in") removed it on
purpose. Do not bring it back. Every gfx803 finding in vLLM goes into the real
source files under `vllm/`, the same way every other fix in this repo lands in
real code rather than in a document about a hypothetical patch.

One trap: `vllm/` is nested twice. `vllm/` is the fork's own root, with its own
`AGENTS.md`, `README.md`, and `benchmarks/`. That `vllm/AGENTS.md` is upstream's
contribution policy, about pull requests to `vllm-project/vllm`. It does not
apply here, because this fork is never upstreamed. The importable package is one
level deeper, at `vllm/vllm/`. Real examples are
`vllm/vllm/model_executor/layers/...` and `vllm/vllm/v1/attention/ops/...`. A
write to `vllm/model_executor/...` lands nowhere, or in the wrong file. This is
easy to do when your shell's working directory has drifted, because the tool's
working directory persists between commands in a session. Use an absolute path
when you are unsure.

Hand-written gfx803 HIP kernel source goes in `vllm/vllm/gfx803_kernels/`
(`gfx803_gemm_lib.hip`, `gfx803_attn_split.hip`), a folder of its own, not next
to the Python loaders that `ctypes`-load the compiled output. `vllm/.gitignore`
excludes `*.hip` repo-wide with the comment "hip files generated by PyTorch".
That is true of upstream's hipify output, which is translated from `.cu` CUDA
source during the build, and it is wrong for these two files, which are
hand-written originals with no `.cu` to regenerate them from. The folder is
whitelisted back with `!/vllm/gfx803_kernels/*.hip` in `vllm/.gitignore`, rather
than force-added past the rule each time. Put any new hand-written gfx803 `.hip`
source there, and it tracks normally. The compiled `.so` must still sit next to
its Python loader, for example
`vllm/vllm/model_executor/layers/libgfx803gemm.so`, because the loader builds its
`ctypes` path from `__file__`. Only the source moved. The `hipcc -o` output path
did not move. The loader's docstring gives the exact command.

Box-only work flows back like this. Iterate on a real gfx803 vLLM fix live on the
box at 192.168.1.214 under `/data/vllm-mobydick/` (see "Hardware access" below),
test it there, then copy the changed and new files back into this local
`vllm/vllm/...` checkout, so the real diff lands here. Do not summarize it into
a separate document. The usual rule still applies: build and stage locally, and
do not commit or push unless you are told to for that specific commit.

## Hardware access

Real-hardware validation needs a real gfx803 card. You cannot emulate it, and
another GPU will not do. If you do not have one, say so. Do not report a patch as
"verified on hardware" because it only built and applied cleanly. The container needs
`--device=/dev/kfd --device=/dev/dri --group-add video` passed through. Inside
the container, `rocminfo` must list the card as a `KERNEL_DISPATCH` agent before
you trust anything else it reports.

Known traps when you add instrumentation on real hardware:

- Device-side `printf` can hang the kernel instead of only slowing it down.
- Any extra device-side write for debugging changes timing, and it can hide the
  race you are looking for.
- Always make sure that you run against the patched binary you think you are
  testing. A stale image or container is a repeated false lead here. Make sure
  of that, and not only that the patch file on disk looks right.

## What does not need the card

You can answer these without hardware. Does the Dockerfile build. Does a patch
apply against a given pin. Where does a bug live, read from the actual pinned
source, diffed across ROCm versions, or followed through the code that really
calls a compiler pass instead of the code you assume calls it. You can also set
up a cross-arch comparison test without gfx803, because that step needs a card of
some kind and not necessarily this one. Do that first, to learn whether a bug is
generic, before you spend hardware time on the gfx803 side of the comparison.
