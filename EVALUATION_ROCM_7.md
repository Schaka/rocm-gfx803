# Evaluating gfx803 on ROCm 7.x: is it feasible?

Question: could this build's gfx803 stack move from the pinned ROCm 6.4.4 base
to ROCm 7.x (7.1.x / current 7.13.x dev line), instead of staying on the last
ROCm-6 release forever? This is a source-level evaluation done by reading the
current upstream trees (ROCm/rocm-libraries monorepo, ROCm/ROCR-Runtime,
ROCm/clr) as of 2026-08-08, cross-referenced against every patch in
`rocm6.4.4/patches/`. No ROCm 7 build was actually attempted -- this is "is the
ground still there to stand on," not "we built it and it works."

**Short answer: the wall everyone hits first (ROCR-Runtime rejecting the
GPU at enumeration) has a small, proven fix. Getting past that wall reveals
the library-level ground is mostly still there too -- gfx803 device maps and
Tensile logic have not been ripped out. But nothing here is upstream-official,
none of it is tested by AMD on this hardware, and this build's own patch set
would need re-verification line by line against whatever 7.x tag is actually
targeted, not just against current `develop`. Feasible as a sustained,
from-source effort. Not a small lift, and it does not get smaller over time --
every 7.x point release is a fresh chance for an unrelated refactor to shift
a line our patches key off of.**

## Why this isn't in scope for the pinned/nightly split already

The existing gfx803 Dockerfile pins ROCm 6.4.4 specifically because ROCR-Runtime
started rejecting Polaris (see the file's own header comment and
[ROCm/clr#269](https://github.com/ROCm/clr/issues/269)). That issue was closed
2026-05-01 as "not planned" by AMD (gfx803 isn't a supported target, use
6.4.2/6.4.4 or patch it yourself). So the premise of this evaluation is: what
would "patch it yourself" actually require, end to end, not just at the
runtime layer.

## Layer 1: HSA runtime enumeration (ROCR-Runtime)

This is the actual reason ROCm 7 rejects the card at all, confirmed by reading
`runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp` in ROCm/ROCR-Runtime
directly:

```cpp
if (node_props.Capability.ui32.DoorbellType != 2)
    throw AMD::hsa_exception(HSA_STATUS_ERROR,
        "Agent creation failed.\nThe GPU node uses a deprecated doorbell type\n");
```

Polaris reports `DoorbellType` 0/1 ("legacy" doorbell). This one `throw`,
in the `GpuAgent` constructor, is the entire enumeration block -- confirmed
by an AMD engineer (`lucbruni-amd`) on the clr issue thread, who got
`rocminfo`/`clinfo` working on a real RX550 via a source build (TheRock) with
this relaxed. Their fork branch
(`lucbruni-amd/TheRock@lb/gfx803-polaris-support`) claims "core HIP, OpenCL
and other basic compute working," not just enumeration -- i.e. legacy-doorbell
hardware still functions once the runtime stops refusing to create the agent
object for it. That's a materially different (better) signal than "rocminfo
prints a name and everything past that is unknown."

**What this means for feasibility:** the layer that looked like the hard
blocker turns out to be a single guarded exception, already prototyped and
verified by someone at AMD on real Polaris hardware. This is the easy part of
the plan, not the hard part -- assuming the rest of the doorbell-dependent
packet/signal path (queue write pointers, interrupt-based vs polling
completion) doesn't have other DoorbellType==2 assumptions elsewhere that
lucbruni's fork didn't need to touch for basic HIP/OpenCL but a full
MIOpen/rocBLAS workload would exercise harder (many concurrent queues,
sustained dispatch -- exactly the conditions under which this build already
found a *hardware*-adjacent race in Tensile's GSU CAS path). Building on
TheRock (a from-source, all-in-one build system) rather than this repo's
current model (patch a stock `rocm/dev-ubuntu` image) would also be a real
process change, not just a patch -- worth flagging now, not discovering later.

## Layer 2: does the compiler still emit gfx803 code objects?

Not independently verified here (would require actually invoking `clang
--target=amdgcn-amd-amdhsa -mcpu=gfx803` against ROCm 7's LLVM, which this
evaluation didn't do), but the indirect evidence is reassuring: rocBLAS's
build (see Layer 3) routes every explicit `-a gfx803` request through
`rocm_check_target_ids()`, which fails hard (`FATAL_ERROR "Unsupported
target ... by compiler"`) if the installed compiler doesn't recognize the
target ID string at all -- that's a distinct failure mode from "target ID
recognized but not in the default list," and nothing in the current rocBLAS
CMake or LLVM's public AMDGPU backend documentation suggests gfx8 codegen
itself (as opposed to library-side tuning) has been dropped. AMDGPU is a
shared LLVM backend used well beyond ROCm (Mesa, other consumers still target
GCN3); removing gfx8 codegen from LLVM itself would be a much bigger, more
visible change than anything found in the library repos. Treat this as
"probably fine" pending a real compile test, not "confirmed."

## Layer 3: rocBLAS

Checked `projects/rocblas/CMakeLists.txt` in ROCm/rocm-libraries `develop`
directly (rocBLAS was folded into this monorepo; the standalone `ROCm/rocBLAS`
repo is now deprecated/archived-in-spirit, pointing here).

**Default target list, by ROCm version, straight from the file:**

```
TARGET_LIST_ROCM_6.0  = gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a...   (gfx803 dropped here)
TARGET_LIST_ROCM_7.0  = same shape, still no gfx803
TARGET_LIST_ROCM_7.1  = same shape, still no gfx803
TARGET_LIST_ROCM_7.13 = same shape, still no gfx803, adds gfx1250
```

Identical story to 6.4.4 today: gfx803 has been out of the *default* list
since 6.0 and stays out through the entire 7.x line inspected. That's not new
information -- it's confirmation the situation hasn't gotten worse. What
matters is whether the escape hatch this build already relies on
(`rmake.py -a gfx803`, which sets `GPU_TARGETS` explicitly, bypassing the
default list) still reaches working Tensile logic. It does:

* `shared/tensile/Tensile/Common.py` still maps `'gfx803': 'r9nano'`
* `Logic/asm_full/r9nano/*.yaml` and `Logic/asm_lite/r9nano_*.yaml` are still
  present and populated in the current tree, not stubs
* `Tensile/Source/lib/include/Tensile/AMDGPU.hpp` still enumerates
  `Processor::gfx803 = 803` with a real name mapping

So the Tensile *logic* this build's patches operate on (WGM,
GlobalSplitU/CAS accumulation, the small-shape assembly dispatch fix) is
still there in principle. The concrete risk is version drift in the
*files our patches key off of*, not the disappearance of gfx803 support:

* `TENSILE_VERSION` is pinned to `4.47.0` in current `develop` -- almost
  certainly a different Tensile version than what 6.4.4 shipped. The GSU
  workspace bug (`gsu-workspace-not-zeroed.sh`, targeting
  `library/src/include/handle.hpp` and `tensile_host.cpp`) and the small-shape
  dispatch fix are exactly the kind of patch that survives a Tensile version
  bump by luck, not by design -- they'd need a real `git apply --check`
  against whatever tag gets pinned, not an assumption of compatibility.
* **Present in 7.x CMake, same as the 6.4.4-era version this build already
  handles:** `option(BUILD_WITH_HIPBLASLT "Build with HipBLASLt" ON)`, and
  when it's on, `find_package(hipblaslt ... REQUIRED)` -- a hard CMake
  dependency when enabled. **Resolved, not a new risk**: `rmake.py` in
  current `develop` still has the exact `--no_hipblaslt` flag
  (`build_hipblaslt`, default `True`, `--no_hipblaslt` flips it to `False` ->
  `-DBUILD_WITH_HIPBLASLT=OFF`) this build already passes today on 6.4.4 (see
  `rocm6.4.4/Dockerfile`'s `rmake.py -i -a "${ROCM_ARCH}" ... --no_hipblaslt`).
  With that flag, the `find_package(hipblaslt REQUIRED)` block is never
  reached (`if(BUILD_WITH_HIPBLASLT)` guards it) -- rocBLAS builds pure-Tensile,
  exactly like it does today. No change needed for 7.x; the flag this build
  already relies on survives unchanged. See the dedicated hipBLASLt section
  below for why this flag exists in the first place and what patching
  hipBLASLt itself (instead of just disabling it) would actually take.

### Could hipBLASLt be patched to support gfx803 instead of just disabled?

Checked directly, since "rocBLAS can substitute" only closes the build-time
question -- worth knowing whether patching hipBLASLt itself is even on the
table as a future option. It isn't, and not because of a missing device-map
entry. hipBLASLt bundles its own fork of Tensile (`tensilelite`,
`projects/hipblaslt/tensilelite/`), separate from rocBLAS's classic Tensile,
and the two have diverged in exactly the dimension that matters here:

* **The ISA gate that actually controls what gets built**,
  `Tensile/Common/Architectures.py`'s `SUPPORTED_ISA` list, starts at
  `IsaVersion(9, 0, 0)` (gfx900). `IsaVersion(8, 0, 3)` (gfx803) is not in it.
  `detectGlobalCurrentISA`/`detectHostGfxArchs` filter every enumerated
  device through this list, so gfx803 is invisible to autodetection even
  before any kernel-generation question comes up.
* **A same-named `architectureMap` entry (`"gfx803": "r9nano"`) and old
  C++ runtime scaffolding (`AMDGPU.hpp`'s `Processor::gfx803`,
  `PlaceholderLibrary.hpp`, a `tensile_host.cpp` string match returning
  `LazyLoadingInit::gfx803`) still exist** -- but this is leftover from
  `tensilelite` being a fork of the same shared Tensile ancestor rocBLAS's
  classic Tensile comes from, not evidence of a maintained gfx803 path. It
  lets the C++ host recognize the device-name string and look for a
  `TensileLibrary_*_gfx803` file; since `TensileCreateLibrary` never
  generates one (gated out by `SUPPORTED_ISA`), that lookup just fails
  cleanly at runtime. This is the actual mechanism behind the Dockerfile's
  existing "hipBLASLt has no gfx803 kernels" comment -- confirmed, not
  just asserted.
* **Re-adding the ISA entry would not be enough.** hipBLASLt's whole reason
  to exist alongside rocBLAS is matrix-core (MFMA/WMMA) GEMM kernels with
  fused epilogues -- that's the product, not a feature toggle. Its solution
  space (`SolutionStructs/Validators/MatrixInstruction.py` and everything
  downstream in `KernelWriterAssembly.py`) is built around selecting and
  emitting `MatrixInstruction` configurations for a target ISA. GCN3
  (gfx803) has **no MFMA instruction and no WMMA instruction at all** --
  this isn't a missing tuning entry the way "gfx803 isn't in the default
  rocBLAS TARGET_LIST" is, it's an absence of the hardware feature the
  entire kernel-generation pipeline exists to target.

**What patching this for real would take:** not a device-map edit, but
writing an entirely new non-matrix-core ("VALU-only") kernel generation
backend inside `tensilelite` -- new `KernelWriter` logic, new solution
parameters, new library-logic YAML schema entries, a new tuning pass -- for
one architecture that no other `tensilelite` consumer needs, in a codebase
whose every abstraction above the instruction-emission layer assumes matrix
cores exist. That is not a bug fix, it's building a second GEMM engine from
scratch inside the wrong codebase -- and the engine it would be duplicating
(source/assembly VALU-only GEMM kernels for exactly this chip, already
correctness-audited) **already exists and already ships**, as rocBLAS's
classic Tensile `r9nano` logic plus this build's own
`patches/rocblas/sgemm-shim/` hand-written kernel for the cases Tensile
still gets wrong. hipBLASLt would add zero capability gfx803 doesn't already
have through rocBLAS, at the cost of building a kernel backend that doesn't
exist anywhere today, for hardware with no matrix-core throughput advantage
to expose in the first place.

**Verdict: not worth attempting, and "virtually impossible" is the honest
characterization of the amount of new code it would take, not a hedge.**
Nothing here suggests it's physically impossible -- GCN3 can still do plain
VALU FMA GEMM, which is exactly what rocBLAS's fallback already proves out --
but doing it *inside hipBLASLt specifically* means recreating rocBLAS's
Tensile from a standing start inside a library architected around a
hardware feature this chip doesn't have, to duplicate something that already
works. Keep using `--no_hipblaslt` and treat this as permanently closed
unless AMD's own `tensilelite` grows a VALU-only kernel path for reasons
unrelated to gfx803 (it won't -- there's no other target that would benefit).

## Layer 4: MIOpen

Checked `projects/miopen/` in the same monorepo. This is the strongest
positive signal in the whole evaluation:

* `CMakeLists.txt`: `set(ALL_GPU_DATABASES gfx803 gfx900 gfx906 gfx908 gfx90a
  gfx942 gfx950 gfx1030 gfx1100 gfx1102 gfx1201)` -- gfx803 is still first in
  the list, current `develop`, not a stale historical branch.
* `src/target_properties.cpp` still maps `Ellesmere`/`Baffin`/`RacerX`/
  `Polaris10` -> `gfx803`, same device-name table this build's Dockerfile
  comment already describes relying on.
* Every solver file this build's patches touch is still present with the
  same gfx803-specific conditionals:
  - `src/solver/conv_bin_winoRxS_fused.cpp`: `if(name != "gfx803")` -- the
    exact function context `winograd-fused-conv-miscompute.patch` targets.
  - `src/solver/conv/conv_bin_winoRxS.cpp` and `conv_bin_wino3x3U.cpp`:
    `if(!(name == "gfx803" || name == "gfx900" || ...))`
  - `src/solver/conv/conv_asm_5x10u2v2f1.cpp`,
    `conv_asm_5x10u2v2b1.cpp`, `conv_asm_7x7c3h224w224k64u2v2p3q3f1.cpp`:
    gfx8-family asm kernel solvers, still gated the same way.
  - `MIOpenReduceCalculation.cpp` (the file `reduce-prod-wrong-identity.sh`
    patches) -- exists, unrelated to arch gating so no reason to expect it
    dropped gfx803 relevance.

None of this proves the patches apply cleanly -- MIOpen is an actively
developed project and these files will have moved lines even if the logic
they contain is unchanged -- but it proves the *target still exists to patch*.
This is a materially different situation than CK or rocMLIR (next section),
where the floor genuinely excludes gfx8 at the architecture level and no
patch could add support back without inventing missing matrix-core codegen
from scratch.

## Layer 5: what's still structurally absent, unchanged from today

Re-confirmed directly against current source, not just carried over from the
existing Dockerfile comment:

* **Composable Kernel** (`projects/composablekernel/CMakeLists.txt`):
  `set(CK_UNSUPPORTED_GPU_TARGETS "gfx900;gfx906;gfx90c")` -- CK's floor is
  gfx908 (MFMA/XDLOPs-based), and gfx803 isn't even in the *unsupported* list
  because it was never a candidate to begin with -- there's no MFMA on GCN3
  hardware for CK to target. This isn't a missing-patch situation, it's a
  missing-instruction-set situation. No amount of patching CK adds gfx803
  support; it would mean writing a new non-MFMA kernel backend for CK, which
  is a different, much larger project than anything in `rocm6.4.4/patches/`.
* **hipBLASLt**: gfx90a+, same reasoning (matrix-core-only library).
* **rocMLIR**: separate repo (`ROCm/rocMLIR`), still active, not checked line
  by line here because it's already switched off in this build's Dockerfile
  and was never gfx8-capable on any branch -- no new information changes that.

## Layer 6: ONNX Runtime patches

`patches/onnxruntime/mha-basic-mode-no-viable-op.patch` and
`topk-radix-tiebreak-nondeterministic.patch` patch ORT's own C++ source
(`GemmSoftmaxGemmPermuteGenericPipeline`, `topk_impl.cuh`), not
ROCm/rocBLAS/MIOpen. These are essentially decoupled from the ROCm-version
question -- they'd port to any ORT version with the same code structure,
independent of whether the ROCm side is 6.4 or 7.x. The real constraint here
is the one already on record ([[mha-basic-mode-ort-bug]] /
[[topk-radix-tiebreak-ort-bug]] in memory, and noted in the Dockerfile): ORT
deleted `onnxruntime/core/providers/rocm/` entirely after v1.22.2 (PR
#25181), so the ORT version ceiling is v1.22.2 regardless of which ROCm this
build targets. That ceiling doesn't get worse or better by moving to ROCm 7 --
it's an independent constraint that would need its own separate evaluation if
it ever needs revisiting (e.g. whether MIGraphX EP alone, without ROCMExecutionProvider,
is viable on newer ORT -- out of scope here).

## What a real ROCm 7 attempt would actually require, in order

1. **Pick and pin an exact ROCm 7.x release tag** (not `develop` -- everything
   above was read against `develop`/HEAD, which moves daily; a real attempt
   needs a stable tag the same way 6.4.4 is pinned today).
2. **Patch and build ROCR-Runtime** to relax the `DoorbellType != 2` check
   (Layer 1). Cross-check against lucbruni-amd's fork for anything beyond the
   one-line check that turned out to matter for real compute, not just
   enumeration -- their branch is the fastest way to find out if there's more
   to it, since apparently there wasn't for basic HIP/OpenCL.
3. **Verify the compiler still emits gfx803 code objects** -- a five-minute
   `clang -mcpu=gfx803` smoke test, trivially checkable early and cheaply,
   should be step one of any hands-on attempt rather than assumed.
4. **Rebuild rocBLAS with `-a gfx803 --no_hipblaslt`** (same flags this build
   already uses on 6.4.4 -- confirmed unchanged in current `rmake.py`), then
   re-run every patch in `patches/rocblas/` against the actual pinned tag and
   re-verify with the existing correctness tooling (`tools/rocblas_sweep.cpp`
   etc.) -- assume every patch needs re-diffing, not just re-applying.
5. **Rebuild MIOpen**, re-diff `patches/miopen/*.patch` against the pinned
   tag, re-run the grouped-conv OOB repro and the Winograd/reduce-prod repros
   that motivated those patches.
6. **Full correctness pass**: everything in `KERNEL_BUGS.md`'s methodology
   section applies unchanged -- there is no reason to assume ROCm 7's Tensile
   codegen for gfx803 doesn't have its own new miscompiles distinct from the
   ones already found on 6.4.4. The GSU CAS race in particular is a
   hardware/codegen interaction (no float atomics on GCN3, software CAS loop)
   that a Tensile version bump could easily change the shape of without
   fixing.
7. **Re-verify PyTorch/MIGraphX/ORT on top of the new base** -- these already
   have their own gfx803-specific caveats (PyTorch 2.8 vs the
   hardware-verified 2.6 fallback) that are orthogonal to the ROCm version but
   compound the total verification surface.

## Feasibility judgment

Getting *something* running on ROCm 7 looks genuinely possible, not a dead
end -- the runtime block has a known small fix, and the library-level ground
(device maps, Tensile logic, solver gating) is still present in source, not
excised. That's a more optimistic finding than "MIOpen/rocBLAS completely
lack the code," which was the pessimistic hypothesis in the question.

But "possible to get compiling" and "worth maintaining" are different
questions, and the honest answer to the second is: **this would roughly
double the maintenance surface, permanently, for a card AMD has explicitly
declined to support** ("not planned," per the clr issue). Every item in
`KERNEL_BUGS.md` was found by original, hands-on investigation against 6.4.4
specifically -- there is no guarantee any of it transfers cleanly, and the
methodology that found them (sweep every solution, trust nothing, verify with
100+ reps) would need to run again in full against 7.x, because a Tensile
version bump can plausibly change *which* shapes/solutions are broken without
changing *whether* something is broken. None of this is a one-time cost that
amortizes -- ROCm 7.x is a moving target (the `develop` tree checked here
changes daily; a real pin would need re-validation against every subsequent
7.x point release the same way 6.4.4 needed its own dedicated investigation),
and gfx803 is exactly the kind of platform where upstream refactors are least
likely to be tested before landing.

Concretely: two Dockerfiles, two full patch sets, two independent correctness
investigations to keep current, indefinitely, for hardware that gets slower
relative to everything else every year. The 6.4.4 pin isn't a stepping stone
to a ROCm-7 gfx803 build that then replaces it -- if this gets attempted, it
would be a *third* build target alongside the existing nightly (full float)
and manual-release (full pin) lines, not a replacement for either, since
6.4.4 is the only ROCm-6 release actually verified against this hardware and
there's no reason to give that up even if 7.x starts working.

**Recommendation: don't commit to it as an ongoing target.** If there's
appetite to find out empirically, the cheap, decisive first move is Layer 1
(patch ROCR-Runtime's doorbell check) plus a `rmake.py -a gfx803
--no_hipblaslt` rocBLAS build against one pinned 7.x tag on real
hardware -- rocminfo enumerates the card and rocBLAS either builds clean or
doesn't. Both are answerable in under an hour
of hands-on time and would convert most of this document's "probably fine" /
"needs re-checking" hedges into confirmed facts before investing in a second
full correctness investigation. If either fails outright, that's the answer,
cheaply. If both pass, the real cost is Layer 6/7's repeat of the entire
`KERNEL_BUGS.md` investigation against new binaries -- which is the actual
multi-week commitment this evaluation can't shortcut, and the point past
which "can we" turns into "should we keep doing this every release."
