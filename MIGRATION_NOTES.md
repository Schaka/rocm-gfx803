# gfx803 on ROCm 7.14 -- migration notes

**2026-08-09: full end-to-end `docker build` succeeded for the first
time** -- `gfx803-rocm7:full-poc`, 23.5GB, all stages clean (ROCR-Runtime
+ CLR, rocBLAS, MIOpen, MIGraphX, PyTorch/torchvision/torchaudio, ORT
v1.28.0, final assembly), all final-stage import sanity checks passed:
ORT providers include `MIGraphXExecutionProvider`, `torch 2.13.0+gitbdbef9c`
built against `HIP 7.14.60850`, `torchvision 0.28.0`, `torchaudio 2.11.0.2`,
`migraphx python module OK`. See "PyTorch build: host OOM on
`TensorTopK.hip`" and the MIGraphX MLIR-stub section below for the two
build-blocking bugs that had to be fixed to get here; both are genuine
upstream gaps, not gfx803-specific, worth a look at whether they're
already fixed on a newer `rocm-7.x`/`therock-7.x` tag before assuming
they'll always be needed.

**Same day, real-hardware validation on the gfx803 box (192.168.1.214,
`/data/rocm7`, podman not docker there): the Layer-1 doorbell fix is
confirmed for real, not just in theory.** Image transferred via `docker
save | ssh | podman load` (podman reads docker's tar format fine).
`ortrun` (an unrelated 6.4.4-era container using the same card) was
stopped first to avoid two ROCm stacks touching the GPU concurrently --
left stopped, needs a manual restart by whoever depends on it.

- `rocminfo` inside the container: **`gfx803` enumerates as a real
  `KERNEL_DISPATCH` HSA agent** ("AMD Radeon RX 470 Graphics", Node 1) --
  not rejected at agent creation, which is what ROCm 7 does *without*
  this line's ROCR-Runtime/CLR patch. This was the single biggest open
  question in `EVALUATION_ROCM_7.md` (Layer 1) and the whole reason this
  directory exists.
- Actual dispatch, not just enumeration: ran a real 256x256 `torch`
  matmul on the GPU (`torch.cuda.is_available() == True`, real numeric
  result, not NaN/garbage). Confirms the doorbell fix enables genuine
  kernel dispatch (the thing the fix specifically targets) and that
  rocBLAS's re-diffed patches (`wgm-miscompute`,
  `small-gemm-assembly-miscompute`) work correctly on real Polaris
  silicon under 7.14, not just "apply cleanly."
- MIOpen: `torch.nn.functional.conv2d` on the GPU produced a real result
  (not NaN/garbage). Two benign warnings, both expected and already
  documented as accepted gaps: no perf-db shipped for gfx803
  (`gfx803_32.HIP.fdb.txt` unreadable -- falls back to untuned kernel
  selection) and no Composable Kernel grouped-conv library for gfx803
  (`libMIOpenCKGroupedConv_gfx803.so` missing -- CK was never built for
  this arch on this line, matches `EVALUATION_ROCM_7.md` Layer 5).
- MIGraphX: built a tiny `dot`+`relu` program by hand via the python
  API, `m.compile(migraphx.get_target('gpu'))` then `m.run(params)` --
  real GPU compile and execution, correct output shape.
- ONNX Runtime: same graph via a real ONNX model (`onnx` package isn't
  in the final image -- not an ORT runtime dependency, only needed here
  to author the test model -- installed ad hoc in the container for this
  one test), `InferenceSession(..., providers=['MIGraphXExecutionProvider'])`
  -- MIGraphX EP compiled and ran it, `get_providers()` confirms it (not
  a silent CPU-EP fallback).

**All four subsystems this build touches (ROCR-Runtime/CLR doorbell fix,
rocBLAS, MIOpen, MIGraphX+ORT) are now confirmed doing real GPU work on
actual gfx803 hardware, same day as the first successful build.** This
satisfies the original "validate loosely on gfx803" ask. Not yet run:
the correctness-suite (broader coverage, not yet adapted for 7.14 --
see "What's NOT done yet" in `README.md`) and anything sustained/
performance-related (today's tests were all single-shot, tiny tensors,
proving correctness of the happy path, not stability under load or
comparable performance to the 6.4.4 line).

`ortrun` (the unrelated container stopped before this session to free
the GPU) was restarted afterward.

## ORT's own ONNX backend test series -- run on real hardware

Ran `onnxruntime/test/python/onnx_backend_test_series.py` (ORT's actual
upstream operator-correctness suite, run through
`onnxruntime.backend`, which uses `get_available_providers()` --
`MIGraphXExecutionProvider` first -- with CPU fallback explicitly
disabled by the test harness itself, so a node MIGraphX EP can't run
shows up as a real failure here even though production usage would
silently fall back to CPU and never notice) against `v1.28.0` source
(matching the build), on the actual gfx803 card. Full log saved next to
this file: `ort-backend-test-full-2026-08-09.log` (18MB, not meant for
git -- kept locally so the raw tracebacks aren't lost, same reasoning as
this file itself).

**3828 tests, 2578 skipped** (ORT's own `current_failing_tests_MIGRAPHX`
exclusion list in `onnx_backend_test_series_filters.jsonc`, applied
automatically since `backend.supports_device("MIGRAPHX")` is true --
this is upstream's own known-broken list for this EP, not something
this session excluded), **leaving 1250 actually exercised: 1067 pass,
13 fail, 170 error.**

### 13 failures (wrong numeric output, not a crash)

**8 of 13 are ConvTranspose in some form**:
`test_convtranspose_autopad_same`, `test_convtranspose_kernel_shape`,
`test_convtranspose_output_shape`, `test_convtranspose_pad`,
`test_convtranspose_pads`, `test_ConvTranspose2d`,
`test_ConvTranspose2d_no_bias`, `test_operator_convtranspose` -- this is
a real, systematic bug, not one flaky case (confirms the suspicion that
ConvTranspose2d "should" work but doesn't). Not yet root-caused --
next: bisect whether this is MIOpen's transposed-conv path, a
MIGraphX lowering bug, or gfx803-specific, the same way the 6.4.4 line's
`KERNEL_BUGS.md` bugs were originally found (extract the op, compare
ROCm EP-equivalent vs CPU EP output on a minimal repro).

The other 5: `test_attention_3d_gqa_with_past_and_present_expanded`,
`test_attention_4d_gqa_with_past_and_present_expanded`,
`test_averagepool_2d_ceil_last_window_starts_on_pad`,
`test_convinteger_without_padding`,
`test_rotary_embedding_with_interleaved_rotary_dim_expanded`. Not yet
individually root-caused.

### 170 "errors" -- three distinct buckets, not one problem

Unittest calls anything that raises an exception (rather than an
`assert_allclose` mismatch) an "error." Splitting by *where* the
exception happens matters a lot here -- it separates "MIGraphX won't
even compile this graph" from "MIGraphX compiled it but the kernel
crashed on this actual card":

- **141: compile-time, at `InferenceSession.__init__`.** Exception is a
  generic `RUNTIME_EXCEPTION: Exception during initialization: Failed
  to call function` -- MIGraphX rejects the graph outright before
  anything runs. Almost entirely two op families: the `Attention` node
  family (~48 tests -- `test_attention_3d_*`, `test_attention_4d_*`;
  this is a very new ONNX op, opset 23) and virtually **every** `Reduce*`
  variant (~80 tests -- L1, L2, LogSum, LogSumExp, Max, Mean, Min, Prod,
  SumSquare, across keepdims/negative-axes/empty-set variants), plus
  most `rotary_embedding` tests. Given the breadth across *all*
  Reduce-family ops, this smells like a generic MIGraphX-EP ONNX-parser
  gap for this ORT/onnx version combination, not something gfx803-
  specific -- worth running this exact same suite against the main
  repo's gfx900+ image to confirm before assuming it's this line's
  problem to fix.
- **23: runtime, at `session.run()`.** Exception names the actual fused
  MIGraphX kernel and says `Status Message: Failed to call function` --
  these graphs *compiled fine* (MIGraphX generated and launched a real
  kernel for them) and the failure is the kernel itself erroring on the
  real card. All 23 fall into exactly four op families: `tril`/`triu`
  (13), `cumsum` (9), `einsum_scalar` (1). **This bucket is the one most
  likely to be a genuine gfx803-specific hardware/codegen issue** --
  same shape as the historical class of gfx803 dispatch bugs in
  [[gfx803-asm-kernels-break-on-tiny-shapes]] -- compiles clean,
  fails only at actual dispatch on this specific old arch. Worth
  root-causing before the compile-time bucket, since it's the one this
  effort can actually fix with a patch.
- **4 (of 6 `sequence_map` tests): a real ORT+MIGraphX-EP software bug,
  not hardware.** `onnxruntime.capi.onnxruntime_pybind11_state.Fail: ...
  Op (Loop) [TypeInferenceError] ... com.microsoft:MGXKernel_subgraph_
  seq_map_body_<hash>_0(-1) is not a registered function/op` -- MIGraphX
  EP fuses a `Loop`'s subgraph body into a synthetic custom op name, but
  ORT's own type inference doesn't know that name. This is an EP/runtime
  interaction bug in the subgraph-fusion registration path, independent
  of gfx803 -- would very likely reproduce on any MIGraphX EP build.
  (The other 2 `sequence_map` tests landed in the 141-bucket instead --
  not yet checked why they differ.)

**Bottom line for this session**: the ConvTranspose failures and the
tril/triu/cumsum runtime-dispatch errors are the two things worth
digging into next -- both look like real, fixable bugs rather than
"MIGraphX just doesn't support this op." The 141-bucket compile-time
errors and the sequence_map Loop bug look like upstream MIGraphX-EP
gaps this line inherits rather than causes.

Running log of the ROCm 6.4.4 -> 7.14 port. This is the working record;
`rocm6.4.4/EVALUATION_ROCM_7.md` is the earlier, pre-implementation feasibility
read (kept as-is, historical). `README.md` in this directory is the
user-facing status summary. This file is where findings get written down
*as they're found*, so nothing gets re-discovered from scratch later.

## Version pin resolution

ROCm 7.14 has no per-component release tags the classic way 6.4.4 did.
Confirmed via `git ls-remote --tags`:

- `ROCm/ROCR-Runtime`, `ROCm/rocm-libraries`, `ROCm/clr`: tags stop at
  `rocm-7.2.4`. No `rocm-7.14` tag exists anywhere.
- `ROCm/TheRock`: has `therock-7.14`. `version.json` at that tag reads
  `"rocm-version": "7.14.0"` -- **this is what "ROCm 7.14" actually refers
  to now**: a TheRock meta-release, not a per-repo tag.
- Docker Hub: `rocm/dev-ubuntu-26.04` has a `7.14.0-full` tag (and
  `7.14.0-full` only for that version -- no `-complete` variant exists for
  7.14; `-complete` was the old per-repo-tag-build naming, `-full` is
  TheRock's). This is AMD's own pinned, signed, TheRock-built image --
  confirmed to have the classic `/opt/rocm/{bin,lib,include,llvm}` symlink
  layout (via `/etc/alternatives/`), so it's a drop-in for every path this
  build's stages already hardcode. No reshaping needed, unlike the
  nightly-only `rocm-base.sh` path (which installs bare `amdrocm-*` .debs
  onto plain ubuntu and has to reconstruct those symlinks itself).
- TheRock's `.gitmodules` at `therock-7.14` pins `rocm-libraries` and
  `rocm-systems` by exact commit SHA (not branch), since both track
  `develop` in the module URL. Resolved via `git ls-tree HEAD
  rocm-libraries rocm-systems`:
  - `rocm-libraries` @ `cd9574023093742434e8c992d13b89ab9a6c1cf8`
  - `rocm-systems` @ `2b22ab0195cc1461cd9abf3b969e9dd7c10af350`

  Both used directly (not via TheRock's submodule machinery -- too heavy
  for what this build needs; a blobless sparse clone straight from the
  upstream repo at the pinned SHA gets the same source at a fraction of
  the size/time).
- MIGraphX stayed a standalone repo (not folded into rocm-libraries) and
  does publish its own tags: `rocm-7.14` exists on `ROCm/AMDMIGraphX`.
- These pins match the main (gfx900+) repo's own manual release track:
  `.github/workflows/release.yml`'s default `rocm_version` is already
  `7.14.0`, `base-image` is computed as
  `rocm/dev-ubuntu-26.04:${rocm_version}-full`, and `pytorch_version`
  defaults to `2.13.0` -- this rocm6.4.4/rocm7 build deliberately reuses the
  same targets rather than picking independently.

**Re-pinning later**: read `therock-7.14`'s `.gitmodules`-pinned commits
again (or whatever the next `therock-7.1X` tag is), don't just bump a
version string -- the whole point of pinning by commit is that "ROCm
7.14.0" the docker tag and the exact rocm-libraries/rocm-systems state it
was built from can drift independently until you re-resolve them together.

## Layer 1: ROCR-Runtime + CLR (new for this line, no 6.4.4 equivalent)

`EVALUATION_ROCM_7.md`'s Layer 1 said the wall was "a small, proven fix."
That undersold it: getting real dispatch working, not just enumeration,
needs a 4-file, ~76-line change, not the single guarded exception in
`amd_gpu_agent.cpp` alone.

**Source of truth for the fix**: not original investigation here. Restored
verbatim from an AMD engineer's (lucbruni-amd) fork,
`lucbruni-amd/TheRock@lb/gfx803-polaris-support`, commit `3d4ad60`, which
the engineer verified on real Polaris hardware (RX 550, both `rocminfo`
and `clinfo`). That commit has two patches (`patches/rocm6.4.4/rocm-systems/
0001-...` and `0002-...`), both re-diffed and validated here against this
build's own pinned `rocm-systems` commit (not lucbruni's fork's, which is
at a different point in history) -- `git apply --check` / `patch --dry-run`
both pass clean, no fuzz, against `2b22ab0195...`.

**Semantic validation done, not just apply-clean** (per explicit
instruction not to trust apply-success alone): the struct fields the fix
wires up --  `amd_queue_t::max_legacy_doorbell_dispatch_id_plus_1`,
`amd_queue_t::legacy_doorbell_lock`, `AMD_SIGNAL_KIND_LEGACY_DOORBELL` --
**already exist, unused, in the pinned 7.14 source** (confirmed via grep
against `runtime/hsa-runtime/inc/amd_hsa_queue.h` and
`amd_hsa_signal.h`). This is AMD's own dormant ABI scaffolding for legacy
doorbells, not something the patch invents -- the fix reactivates disabled
logic, it doesn't retrofit incompatible layout onto a struct that never
had it.

Files: `patches/rocm-systems/hsa-agent-rejects-legacy-doorbell.{patch,sh}`,
`patches/rocm-systems/opencl-gfx8-hardcoded-rejection.{patch,sh}`.

**Not yet done**: nothing beyond this has been run against real hardware.
The engineer's own verification was "core HIP, OpenCL and other basic
compute working" -- not a full MIOpen/rocBLAS workload under sustained
concurrent dispatch, which is exactly the regime this repo's own
6.4.4-era investigation found a *separate* hardware-adjacent race in
(Tensile's GSU CAS path, see `KERNEL_BUGS.md`). No reason to assume that
race, or a new one, doesn't also show up here -- untested.

The Dockerfile builds HIP only (`CLR_BUILD_OCL=OFF`); the OpenCL patch is
applied (harmless, validated) but the OCL runtime itself isn't built,
since this stack doesn't use it.

## Layer 3: rocBLAS

Two patches carried over from the 6.4.4 line; one intentionally NOT
carried over.

### wgm-miscompute.sh -- unchanged, no port needed

Path/pattern-based (rewrites every `WorkGroupMapping:` entry in the
Tensile `Logic/` YAML tree via `sed`, not a line-numbered diff), and
self-verifying (fails the build if its target pattern disappears). Tested
directly against the pinned 7.14 rocBLAS source: 103,741 non-1
`WorkGroupMapping` entries found, so the self-check passes unchanged.
Copied as-is.

### small-gemm-assembly-miscompute.patch -- re-diffed, two real issues found

`git apply --check` reported "Skipped patch" with **exit 0** and **zero
files modified** -- looked like a clean no-op success, was not. Root-caused
by cross-checking with `patch --dry-run`, which correctly reported "Hunk
#2 FAILED". This is a `git apply`-specific defect/quirk on this box's git
(2.55.0) -- reproduced even for a hand-verified-correct, re-diffed patch
later in the same investigation. **Lesson: don't trust `git apply`'s exit
code alone on this git version -- cross-check with `patch --dry-run`, or
just use `patch` throughout (which every `.sh` script in this directory
now does).**

Two real drifts found by actually reading the current source, not
assumed:

1. **Reindentation** (cosmetic): `tensile_host.cpp`'s
   `runContractionProblem()` gained an `is_device_memory_size_query()`
   branch that pushed the patched block one level deeper (16-space body
   indent vs. 12-space in 6.4.4). Same control flow, just shifted -- this
   is the same class of drift `gsu-workspace-not-zeroed.patch` (below)
   also hit.
2. **Real API change** (not cosmetic): `runContractionProblem`'s template
   signature was `template <typename TiA, typename To, typename Tc,
   typename TiB, typename TcA, typename TcB>` in 6.4.4 (confirmed by
   fetching that exact tag's source) -- separate input types for the A and
   B GEMM operands. Collapsed to `template <typename Ti, typename To,
   typename Tc>` in 7.14 -- one input type now covers both operands. The
   patch's `std::is_same<TiA, ...> && std::is_same<TiB, ...>` gate would
   not compile against 7.14 at all (no `TiA`/`TiB` in scope); rewritten to
   a single `std::is_same<Ti, float>::value` check.

Re-diffed patch applies clean (`patch -p1 --dry-run`, exact line match, no
offset) against the pinned commit. **Not yet re-verified on real
hardware** -- the 6.4.4 measurements (onnxruntime_test_all failure counts,
the 1x1x1 isolation) haven't been re-run against 7.14 binaries.

### gsu-workspace-not-zeroed.patch -- re-diffed but deliberately NOT wired

Re-diffed for completeness (same reindentation-drift class as the
small-GEMM patch above; kept in `patches/rocblas/` for reference), but
**this patch was never actually wired into `rocm6.4.4/Dockerfile` for
6.4.4 either** -- checked directly: `rocm6.4.4/Dockerfile`'s
`rocblas-builder` stage only calls `wgm-miscompute.sh` and
`small-gemm-assembly-miscompute.sh`. `gsu-workspace-not-zeroed.sh` exists
in `rocm6.4.4/patches/rocblas/` but nothing invokes it -- superseded by a
better approach at some point in the 6.4.4 line's own history, per direct
user correction during this port. **Do not wire it into
`Dockerfile` either.**

## Layer 4: MIOpen

Three patches from the 6.4.4 line evaluated; two ported clean, one
blocked outright, one (opt-in/experimental) not attempted.

### winograd-fused-conv-miscompute.patch, reduce-prod-wrong-identity.patch -- ported unchanged

Both apply clean against the pinned 7.14 source with only small line
offsets (6 and 1 lines respectively), no content drift. Semantically
spot-checked: `ConvBinWinogradRxSFused::IsApplicable` and
`calculationfwdcontiguous`/`calculationparallelfwdcontiguous` (the exact
functions/classes these patches target) are present with matching
signatures. Copied as-is into `patches/miopen/`, `.sh` wrappers rewritten
to use `patch` instead of `git apply` (same quirk as above).

### conv-direct-fwd-grouped-oob.patch -- BLOCKED, target solver removed upstream

**`patch`/`git apply` couldn't even find the target file**:
`src/solver/conv/conv_ocl_dir2Dfwd.cpp` does not exist in the pinned 7.14
source. Confirmed via `grep -rln "ConvOclDirectFwd" .` across the whole
MIOpen tree -- the only hits are `CHANGELOG.md` and `src/solver.cpp`,
where the latter literally has:

```cpp
++id; // removed ConvOclDirectFwdGen
++id; // removed ConvOclDirectFwd3x3
++id; // removed ConvOclDirectFwd
++id; // removed ConvOclDirectFwdFused
++id; // removed ConvOclDirectFwd1x1
```

The whole `ConvOclDirectFwd*` solver family (OpenCL-source direct
convolution) was deleted from MIOpen between 6.4.4 and 7.14. A
replacement exists -- `ConvHipDirectFwd`
(`src/solver/conv/conv_hip_dir2Dfwd.cpp`, HIP-source, not OpenCL-source)
-- but it is **not the same code**, and whether it has the same
grouped-convolution out-of-bounds weights-buffer read the original patch
fixed is genuinely unknown. Porting the old fix (reject
`GroupCount() != 1` in `IsApplicable`) onto the new solver without
re-running the original repro would be inventing an unverified
assumption, not restoring a known fix.

**Status: not applied, not ported.** If the same *symptom* (a fault or
miscompute on grouped/depthwise convs, especially under
`AMD_SERIALIZE_KERNEL=3` attribution) resurfaces during real-hardware
validation on 7.14, that's the trigger to open a fresh investigation
against `ConvHipDirectFwd` specifically -- start from
`rocm6.4.4/KERNEL_BUGS.md`'s methodology, not from assuming this old patch
just needs its class name swapped.

### reduce-program-bound-eviction.patch -- not attempted

Opt-in/experimental even on the 6.4.4 line (gated behind
`--build-arg ENABLE_REDUCE_BOUND=1`, off by default, never validated
against the real ORT test suite per its own header). `patch --dry-run`
against the pinned 7.14 `src/reducetensor.cpp` shows hunk 1 failing and
the other 4 succeeding with a large (-288 line) offset -- real drift, not
just noise, but not investigated further given this patch's own
not-production-ready status on 6.4.4. Not ported, not wired. Lower
priority than the two blocking items above; revisit only if/when the
ReduceSum investigation (`KERNEL_BUGS.md`, "The ReduceSum kernel-cache
mystery") resumes.

### reduce-kernel-cache-eviction.patch -- not applicable

Already abandoned/not-shipped on the 6.4.4 line (see its own
`*** ABANDONED ***` header). Not evaluated for porting; no reason to
revive an already-abandoned approach as a first move on a new ROCm line.

## MIGraphX

`parse-resize-fixes.patch` (two ONNX-parser backports from 6.4.4's
`release/rocm-rel-6.4` pin) is **fully obsolete against `rocm-7.14`** --
checked directly against the actual current source, not assumed from the
patch's own "these are already merged to develop" note:

- Fix 1 (`keep_aspect_ratio_policy="stretch"` acceptance): already present
  verbatim in `rocm-7.14`'s `src/onnx/parse_resize.cpp` (lines ~273-288,
  logically inverted but equivalent: `if(not contains(...) or ... ==
  "stretch")`).
- Fix 2 (`.as_standard()` before `calc_neighbor_points`): the target
  function `calc_neighbor_points` no longer exists at all in
  `rocm-7.14` -- the whole linear-mode resize lowering it patched was
  replaced by a JIT-backed resize op upstream (the original patch's own
  header predicted this might happen: "upstream `develop` has since
  replaced the whole approach with a JIT-backed resize op, PR #4553").

**No port of the 6.4.4-era patch needed for this line** -- but two new,
7.14-specific build/runtime patches were needed instead (both in
`patches/migraphx/`):

### gfx-default-rocblas-hipblaslt-off-build-failure.patch

`gpu::gfx_default_rocblas()` is declared behind `#if
MIGRAPHX_USE_HIPBLASLT` in `device_name.hpp` but called unconditionally
from `lowering.cpp`, a real upstream gap (not gfx803-specific) that only
shows up when hipBLASLt is disabled -- true for gfx803 since hipBLASLt
has never had gfx8 kernels. Fails the build itself (undefined reference
at link time). Fix moves the declaration outside the `#if` and adds a
stub `#else` definition returning `true` (dead code at runtime --
`hipblaslt_supported()` always returns `false` when disabled, so the
`or gfx_default_rocblas()` short-circuits away; the fix only needs the
symbol to exist, not to do anything reachable).

### mlir-stub-missing-symbols.patch

Same class of bug, different function, caught much later -- at final
image assembly, not migraphx-builder. `is_module_fusible`,
`dump_mlir_to_file`, `dump_mlir_to_mxr`, and `adjust_param_shapes` are
declared unconditionally in `mlir.hpp` and called unconditionally from
`jit/mlir.cpp` (globbed via `file(GLOB JIT_GPU_SRCS .../jit/*.cpp)`,
always compiled regardless of `MIGRAPHX_ENABLE_MLIR`). Their only
definitions live inside `#ifdef MIGRAPHX_MLIR` in `mlir.cpp`; unlike the
sibling functions in that same file (`compile_mlir`, `insert_mlir`,
`get_tuning_config_mlir`), these four have **no stub in the `#else`
branch** -- an upstream oversight, not something gfx803-specific either
(MLIR is disabled on every non-MLIR-target build, not just Polaris).
`adjust_param_shapes` was caught proactively by grepping every other
`mlir.hpp` declaration for unconditional callers in `jit/mlir.cpp`
before rebuilding again, rather than waiting for a fourth pipeline
stage to surface it individually.

Doesn't fail the build: `libmigraphx_gpu.so` links fine because nothing
*else* in that shared object references the missing symbols to trip the
linker. Surfaced instead as `ImportError:
.../libmigraphx_gpu.so.2016000: undefined symbol:
_ZN8migraphx14version_2_16_03gpu17is_module_fusibleERKNS0_6moduleERKNS1_7contextERKNS0_5valueE`
the first time anything `dlopen`s it -- specifically caught by this
Dockerfile's own final-stage `import migraphx` sanity check, not by
`import onnxruntime` (which loaded and registered
`MIGraphXExecutionProvider` successfully just before it, since ORT's
provider registration never touches this code path). Consistent with
this toolchain defaulting to eager/`BIND_NOW` symbol resolution --
otherwise this would only surface as a crash on first real *use* of
MLIR fusion, not on import. Fix: added the missing stubs to `mlir.cpp`'s
`#else` branch, `is_module_fusible` returning `false` (correct: no MLIR
means nothing is MLIR-fusible), the two dump functions as no-ops,
matching the existing stub style exactly.

**Lesson for this whole feature-flag-gated-declaration bug class**
(now hit three times across this port: `gfx_default_rocblas`,
`is_module_fusible`+friends, and structurally similar to the PyTorch
`TensorTopK.hip` directory-scope bug below in spirit if not mechanism):
disabling a feature via `-DMIGRAPHX_USE_X=Off` in this codebase doesn't
reliably mean "no dead code paths remain" -- it means "grep every
always-compiled caller of that feature's public API for one that
forgot to check whether a stub exists." Worth doing that grep
proactively for any *other* MLIR-gated or CK-gated function before
assuming the rest of `mlir.cpp`/similar files are clean, rather than
waiting for each one to surface individually at a different pipeline
stage.

## ONNX Runtime -- switched to v1.28.0, MIGraphX-only (dropped ROCm EP)

Originally pinned to `v1.22.2` with `--use_rocm` kept alive, matching the
6.4.4 line's reasoning (`EVALUATION_ROCM_7.md` Layer 6: ORT deleted the
ROCm EP entirely after v1.22.2, and MIGraphX EP alone has no CK/MLIR to
fuse with on gfx803, so the ROCm EP fallback matters more here than on
gfx900+). Also carried over both 6.4.4-era ORT patches
(`mha-basic-mode-no-viable-op`, `topk-radix-tiebreak-nondeterministic`)
unchanged on that assumption.

**Reverted after a real build failure, not preemptively.** v1.22.2's ROCm
EP source does not compile against ROCm 7.14's HIP headers:
`onnxruntime/core/providers/rocm/cu_inc/common.cuh`'s
`constexpr int GPU_WARP_SIZE = warpSize;` fails because `warpSize` changed
from a directly-constexpr-usable value to a non-constexpr accessor object
(`hip/amd_detail/amd_warp_functions.h`'s `operator int()`, not marked
`constexpr`) -- a HIP API change orthogonal to gfx803, confirmed via
direct compile error, not assumed. Patching multi-year-old,
upstream-deleted EP code (PR #25181) to work with a HIP version it was
never built against was judged not worth it -- the fix found here was
one instance among what's very likely several, in code nobody upstream
maintains anymore.

**Resolution (user-confirmed choice between two options -- see chat):
switched to `v1.28.0`, `--use_migraphx` only, matching the main
(gfx900+) build's `docker/ort.Dockerfile` / `scripts/build/ort.sh`
exactly.** `--use_rocm`/`--rocm_home` are gone from ORT's own build
flags as of 1.28 regardless (ROCm EP folded away upstream,
https://github.com/microsoft/onnxruntime/issues/26801) -- `--rocm_home`
only survives as a deprecated no-op under the MIGraphX flag group, per
that script's own comment.

Both 6.4.4-era ORT patches **dropped, not ported**: both patch
ROCm-EP-exclusive source
(`contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh`
and `core/providers/cuda/math/topk_impl.cuh`, the latter only
compiled into the build when hipified for `--use_rocm`) -- neither file
is reached at all once `--use_rocm` isn't passed, so applying them would
be dead weight, not a correctness fix. `patches/onnxruntime/`
is intentionally empty.

**Accepted cost**: the CK/MLIR-fusion gap MIGraphX has always had on
gfx803 (no Composable Kernel, no MLIR -- see `EVALUATION_ROCM_7.md`
Layer 5) is no longer softened by a ROCm EP fallback on this line, unlike
6.4.4. Not yet measured how much this actually costs in practice (no
real-hardware run has happened yet) -- revisit if/when correctness-suite
results on 7.14 show MIGraphX-only underperforming the 6.4.4 line's
ROCm-EP-assisted results on the same workloads.

## PyTorch build: host OOM on `TensorTopK.hip`

Full-Dockerfile builds repeatedly died on the host, not in a container --
the Linux OOM killer, not a compiler bug. First symptom:
`clang++: error: unable to execute command: Killed` /
`Aborted (core dumped)` while compiling
`caffe2/CMakeFiles/torch_hip.dir/__/aten/src/ATen/native/hip/TensorTopK.hip.o`.
Recurred at `BUILD_PARALLEL_LEVEL=4` and again at `=2` -- lowering ninja's
job count didn't help, because it's not a concurrency problem: `ps`
during the second recurrence showed the single `clang-23 -cc1` process
compiling that one file at **19.6GB RSS**, alone, with the host's 8GB
zram swap already saturated. `TensorTopK.hip`'s CUB radix-sort template
instantiations are just that expensive for LLVM's AMDGPU backend to
register-allocate at `-O3`. No `-j` value fixes a single file's own peak
footprint.

Two independent fixes, both applied:
- **Host-side (this session, ad hoc, not repo state)**: added a real
  disk-backed swapfile (`/swapfile-pytorch-build`, 24G, priority -1 so
  zram is preferred first) as overflow headroom beyond zram -- zram is
  RAM-backed compression, not real extra capacity, so it was already
  useless once full. On btrfs a swapfile needs `chattr +C` (NOCOW) set
  *before* it has any data, or `swapon` fails with `Invalid argument`.
  Remove this file after the build stabilizes; it's a host workaround,
  not something the build depends on.
- **Real fix, in-repo**:
  `patches/pytorch/tensortopk-hip-build-oom.patch` --
  `set_source_files_properties()` on `TensorTopK.hip`, forcing `-O1` for
  that one file only (trailing flag wins over the default `-O3`). Wired
  into `pytorch-builder` the same way every other stage applies its
  patches (`COPY patches/pytorch/`, `sh .../*.sh /pytorch`).

  **First attempt was silently ineffective** -- patched
  `aten/src/ATen/CMakeLists.txt` right after `ATen_HIP_SRCS` is
  assembled (~line 663), applied cleanly, `grep` marker check passed,
  but the live compile still showed `-O3` and OOM'd again at ~10 min
  in (22GB RSS, host down to 680MB available). Root cause:
  `set_source_files_properties()` is directory-scoped in CMake. That
  file only *assembles* the `ATen_HIP_SRCS`/`Caffe2_HIP_SRCS` variable
  list and hands it up via `PARENT_SCOPE`; the actual
  `add_library(torch_hip ${Caffe2_HIP_SRCS})` call lives in
  `caffe2/CMakeLists.txt` (a sibling directory, not a child of
  `aten/src/ATen`) -- so the property never reached the target that
  compiles the file. No error either way; CMake just silently drops a
  source property nothing in-scope ever reads. Confirmed by
  `add_library(torch_hip` grep landing in `caffe2/CMakeLists.txt:943`.
  Fixed by moving the override there instead, right next to upstream's
  own `set_source_files_properties(... PROPERTIES LANGUAGE HIP)` call
  at line ~936 (same directory scope as the target, same pattern
  upstream already uses for exactly this reason) -- referenced via
  `${TORCH_ROOT}` (this file's own established convention for
  referencing repo-root paths, not a path I introduced) rather than
  `${CMAKE_CURRENT_SOURCE_DIR}`, since `caffe2/` isn't `aten/`'s parent.
  Lesson: when overriding a per-source compile property in PyTorch's
  (or generally any multi-directory) CMake tree, always confirm the
  property is set in the same directory scope as the `add_library()`/
  `add_executable()` call that actually consumes the source list --
  not wherever the file happens to get glob'd.

## Tooling note: `git apply` vs. `patch` on this box

Every `.sh` wrapper in `patches/` uses `patch -p1`, not
`git apply`, unlike the 6.4.4-era scripts. Reason: `git apply --check`
(git 2.55.0) reproducibly printed `Skipped patch '<file>'.` and exited
**0** without modifying anything, for at least one patch later
independently confirmed correct (content and placement verified by hand,
applies cleanly via `patch -p1 --dry-run` with an exact line match). This
looks like a git-apply-specific defect or overly-lenient heuristic on
this git version, not a defect in the affected patches. `patch` was
reliable throughout this investigation and is used consistently for that
reason, not case-by-case.

## Layer 2: compiler smoke test -- confirmed

`EVALUATION_ROCM_7.md` Layer 2 flagged this as "probably fine, pending a
real compile test" and didn't run one. Run here, against the actual
`rocm/dev-ubuntu-26.04:7.14.0-full` image's LLVM: compiled a trivial HIP
kernel (`__global__ void kern(float*)`) with
`clang++ -x hip --offload-arch=gfx803`, succeeded (57KB object), extracted
its `.hip_fatbin` section (49KB, non-empty) via `objcopy`, and confirmed
the string `gfx803` appears in the embedded device code. ROCm 7.14's LLVM
still emits real gfx803 code objects -- upgraded from "probably fine" to
confirmed.

## Correction (2026-08-09): the 170 "errors" are NOT gfx803-specific

Traced all four op families in the 23-runtime-error bucket by re-running
their exact `onnx/backend/test/data/node/*/model.onnx` files directly
against `MIGraphXExecutionProvider` with `AMD_LOG_LEVEL=3` +
`MIGRAPHX_TRACE_COMPILE=1`, instead of trusting ORT's generic wrapper
message (`Status Message: Failed to call function`, which looks like a
dispatch/hardware failure but isn't one -- it's ORT's own catch-all for
"the EP threw during its lazy per-subgraph compile-or-run call", which
covers parse-time rejections too, not just kernel launch failures). The
real error is one line above it in MIGraphX's own log, and it changes the
conclusion for this whole bucket:

- `tril`/`triu` (13 tests): `PARSE_TRILU: dynamic k not supported`
- `cumsum` (9 tests): `PARSE_PREFIX_SCAN: axis - dynamic shape not supported`
- `einsum_scalar` (1 test): `parse_equation: einsum op_builder: No term
  specified before '->' symbol` (malformed-equation parsing edge case)

All three are genuine ONNX-spec compliance gaps in MIGraphX's parser: the
ONNX spec allows `Trilu`'s `k` and `CumSum`'s `axis` to be passed as plain
graph *inputs* (not attributes), and these particular test models do
exactly that (confirmed directly: `test_cumsum_1d`'s `axis` has zero
matching entry in `graph.initializer` -- it's a bare, non-constant graph
input). MIGraphX's parser categorically requires these to be
constant-foldable at parse time and rejects the model outright otherwise
-- same rejection on any GPU architecture running this MIGraphX build,
gfx803 never enters into it. **Not something a gfx803 patch can fix** --
fixing it means teaching MIGraphX's ONNX parser to handle non-constant
`k`/`axis` (e.g. lower to a dynamic op instead of requiring a literal),
which is upstream, cross-arch scope, out of this line's mandate.

Also spot-checked one 141-bucket (compile-time) case from the Attention
family: `test_attention_3d_scaled` fails the same way --
`PARSE_ATTENTION: num_heads attribute required`, a real gap/strictness bug
in MIGraphX's own (brand-new, opset-23) Attention parser, unrelated to
gfx803. Reinforces the existing 141-bucket hypothesis (generic
MIGraphX-EP ONNX-parser gap for this version combo) rather than
overturning it.

**Bottom line, corrected**: of the 170 errors, none traced so far are
gfx803-specific or fixable via a patch in this repo. The
`sequence_map`/`Loop` bug (4 tests) was already known to be an ORT/EP
registration bug, independent of gfx803.

## Cross-arch differential (2026-08-09): gfx803 vs. gfx1201, same suite

Ran the identical `onnx_backend_test_series.py` (same v1.28.0 pin, same
`onnx==1.22.0`, same filters/overrides jsonc from the same tag) against
`ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx1201` on real
gfx1201 hardware, to separate "generic MIGraphX-EP gap, any arch" from
"this line broke something in the 6.4.4->7.14 port." Same total (3828),
same skip count (2578, same exclusion list) -- **7 failures, 178 errors**
vs. gfx803's 13/170. Diffed by exact test name (`comm` on sorted FAIL:/
ERROR: lines from both full logs):

- **176 of gfx803's 183 failing/erroring tests also fail on gfx1201** --
  confirms the parser-gap conclusion above at a much larger scale than
  the 4 cases individually traced: the overwhelming majority of both
  buckets (the 141+23+4 error split and 10 of the 13 failures) are
  generic MIGraphX-EP gaps for this ORT/onnx/MIGraphX version
  combination, not this line's problem.
- **7 fail ONLY on gfx803 -- real discrepancies from this port, need
  fixing**:
  - `test_convtranspose_autopad_same_cpu`, `test_convtranspose_kernel_shape_cpu`,
    `test_convtranspose_output_shape_cpu`, `test_convtranspose_pad_cpu`,
    `test_convtranspose_pads_cpu` -- 5 of the 8 original ConvTranspose
    failures. The other 3 (`test_ConvTranspose2d_cpu`,
    `test_ConvTranspose2d_no_bias_cpu`, `test_operator_convtranspose_cpu`)
    fail on gfx1201 too -- confirmed generic, not gfx803's problem. So the
    original "8 of 13 are ConvTranspose, systematic bug" framing was too
    broad: it's a real systematic bug, but only for the 5 explicit-`Trilu`-
    style-parameterized node tests, not the PyTorch-operator-derived ones.
  - `test_attention_3d_gqa_with_past_and_present_expanded_cpu`,
    `test_attention_4d_gqa_with_past_and_present_expanded_cpu` -- note the
    *non*-`_expanded` GQA variants (`test_attention_3d_gqa_with_past_and_present_cpu`
    etc., 13 tests total) error identically on both archs (generic,
    already-known compile-time parser gap), but these two `_expanded`
    decomposed-subgraph variants specifically produce wrong numeric output
    on gfx803 only.
- 9 fail/error ONLY on gfx1201 (`test_strnormalizer_*`/`test_strnorm_model_*`,
  `test_qlinearconv_cpu`) -- gfx1201-side quirks, irrelevant to gfx803
  validation, not investigated further here.

### The 7, in detail (exact mismatch data from `ort-backend-test-full-2026-08-09.log`)

**5 ConvTranspose node tests** -- all four failing shapes show the *same*
error signature: small integer test data (values 0-8, not floats), a
consistent count of mismatched elements (28-29% of the tensor), and
mismatches that are structural, not numerical-precision drift -- e.g.
`test_convtranspose_pads`: `[0, 0, 5, 1]: 0.0 (ACTUAL), 7.0 (DESIRED)` --
an output position that should have accumulated a real contribution came
back exactly zero. That shape (a position silently getting *no*
contribution instead of a wrong-but-nonzero one) reads like a
scatter-add/col2im step in the deconvolution lowering either skipping a
write or writing to the wrong offset, not an activation/rounding error.
`kernel_shape`, `output_shape` and `pad` are byte-identical in both
mismatch count (45/160) and the exact ACTUAL/DESIRED arrays -- strongly
suggests these three ONNX test variants compile down to the *same*
MIGraphX/MIOpen kernel dispatch, differing only in how the ONNX graph
expresses padding (explicit `pads` attribute vs `output_shape` vs
autopad), consistent with one shared broken code path underneath all of
them rather than three separate bugs.

**2 Attention GQA `_expanded` tests**: `test_attention_3d_gqa_with_past_and_present_expanded`
and the `4d` variant both show the *first N elements exactly correct*,
then diverge from a fixed offset onward (3d: elements 0-47 exact match,
diverges at index 48 out of 576; 4d: exact match through most of the
tensor, diverges starting at `[0, 4, 3, 0]`). A clean prefix-match/
suffix-diverge split like this is the classic signature of a GEMM
tile-boundary bug -- correct within one tile/block, wrong past it -- not
a softmax or scale/mask logic error (which would show mismatches
scattered non-contiguously, tracking wherever the mask/scale applies).

### Patch culpability -- which existing rocm7 patches could cause this

Went back through every patch currently shipped in
`patches/` (the four ported *correctness* fixes from 6.4.4:
`rocblas/wgm-miscompute.sh`, `rocblas/small-gemm-assembly-miscompute.patch`,
`miopen/winograd-fused-conv-miscompute.patch`,
`miopen/reduce-prod-wrong-identity.patch`) with these two failure
signatures specifically in mind. None of the four have been re-verified
against real 7.14 hardware for the *specific* bug they claim to fix (each
patch's own header says so) -- re-diffing only proved they compile, not
that the underlying assembly/logic they patch around still behaves the
same way on 7.14's regenerated Tensile kernel set. Two stand out as
plausible:

- **`rocblas/small-gemm-assembly-miscompute.patch`** -- fires specifically
  when every GEMM dimension is <=8. ConvTranspose's official ONNX test
  shapes are tiny (the test data above tops out at value 8, and these are
  1-2 channel, 3x3-kernel toy tensors) -- if MIGraphX's deconvolution
  lowering for gfx803 goes through an im2col-style GEMM (rocBLAS is the
  only GEMM backend in play here: MLIR is disabled for gfx803, see the
  MIGraphX section above), the resulting problem dimensions plausibly
  land in this patch's all-dims-<=8 trigger zone. Attention's GQA matmuls
  (Q@K^T, softmax@V) at the test's small head/seq-len sizes are similarly
  tiny. **Most likely single culprit for both clusters if the underlying
  cause is rocBLAS, not MIOpen/MIGraphX.**
- **`rocblas/wgm-miscompute.sh`** -- a workgroup-tile-swizzle bug that is
  "correct or wrong per-tile," independent of overall problem size --
  matches the attention test's clean tile-boundary divergence pattern
  (first tile right, next tile wrong) better than the small-gemm patch
  does. Forces `WorkGroupMapping: 1` tree-wide at build time and its own
  self-check (before/after non-1-WGM count) passed during this build, so
  the substitution *did* happen -- but whether WGM=1 is still a universal
  fix (vs. 6.4.4's Tensile logic tree, which may have picked different
  solutions/kernels for these exact shapes under 7.14's regenerated
  library) is unverified.
- **ConvTranspose specifically may not be a rocBLAS problem at all** --
  this line's own `conv-direct-fwd-grouped-oob` note (see "Layer 4:
  MIOpen" above) already flagged that the `ConvOclDirectFwd*` solver
  family (which had a known grouped/OOB weights-read bug on 6.4.4) was
  deleted upstream and replaced by `ConvHipDirectFwd`
  (`src/solver/conv/conv_hip_dir2Dfwd.cpp`) with **"whether it has the
  same grouped-convolution out-of-bounds weights-buffer read the original
  patch fixed is genuinely unknown"** and explicitly named the trigger
  condition for investigating it: *"If the same symptom (a fault or
  miscompute on grouped/depthwise convs...) resurfaces during
  real-hardware validation on 7.14, that's the trigger."* ConvTranspose
  commonly lowers to a grouped/weight-transposed convolution internally --
  this may be exactly that predicted resurfacing, in which case the fix
  isn't porting an existing patch at all, it's a fresh investigation
  against `ConvHipDirectFwd`, same as the note already anticipated.
- **`winograd-fused-conv-miscompute.patch`** and
  **`reduce-prod-wrong-identity.patch`** don't fit either failure: the
  Winograd fix only applies to `ConvBinWinogradRxSFused`
  (`ConvActivationFusion` fusion-plan path specifically, not a plain
  conv/deconv or GEMM path), and reduce-prod is unrelated to both
  ConvTranspose and Attention's arithmetic. Low-probability culprits,
  not ruled out formally.

### va-reuse-defer: real signal from the actual 52-shape harness (2026-08-10)

The ORT backend suite showed zero difference with/without this patch
(see below) -- expected, since the bug it fixes needs sustained
back-to-back dispatch reusing freed device addresses, not one-shot
independent small models. Went to the actual tool this bug was found and
fixed with: `rocm6.4.4/tools/reduce-harness/` (52-shape MIOpen-CK-kernel
VA-reuse sweep, the same harness `rocm6.4.4/patches/rocr/va-reuse-defer.patch`
cites for its 6.4.4 measurements). Built it against the rocm7 image's own
`hipcc`/headers; the *committed* `.hsaco` code objects (built for 6.4.4)
loaded and dispatched fine under 7.14's ROCr without any rebuild -- HSA
code objects are ABI-stable across this ROCr version gap for the same
target ISA.

- **`gfx803-rocm7:va-reuse-defer` (patched), 100 runs: 0/52 failed, every
  run.** 5200 total shape-checks, zero bad.
- **`gfx803-rocm7:full-poc` (unpatched control), 30 runs: 29 runs at
  10/52 failed, 1 run at 9/52.** Matches the 6.4.4-era stock failure rate
  (~6-10/52) almost exactly.

This is a clean, direct, dispositive result on the actual mechanism the
patch targets, not an inference from a suite that wasn't built to catch
it: **the VA-reuse race is confirmed still present on stock ROCm 7.14
gfx803, at the same rate as 6.4.4, and the re-diffed patch eliminates it
completely.** Recommendation: wire it in -- the case for it is now direct
evidence, not just "carried over from 6.4.4 defensively."

### ORT suite comparison, patched vs. unpatched (2026-08-09/10)

Ran the identical full `onnx_backend_test_series.py` (3828 tests) against
`gfx803-rocm7:va-reuse-defer` and diffed the exact FAIL:/ERROR: test-name
set against the `full-poc` baseline log: **byte-for-byte identical** --
same 13 failures, same 170 errors, same 2578 skips, same test names.
Confirms the prediction above: this suite's workload (independent
small-model single-shot runs) doesn't exercise the sustained
allocate/free-reuse pattern the bug needs, so it can't see this fix (or
its absence) either way. Not a sign the patch does nothing -- the 52-shape
harness above is the tool that actually measures this bug; this suite
was never going to.

### ConvTranspose routing -- confirmed by trace, `ConvHipDirectFwd` hypothesis wrong (2026-08-10)

Wired `va-reuse-defer` into the real `Dockerfile` (see above)
-- confirmed effective, no longer an experiment. Then traced
`test_convtranspose_pads` directly with
`MIGRAPHX_TRACE_COMPILE=1 MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6`
to see the real dispatch path instead of guessing from the solver-gap
list above.

**MIGraphX lowers ConvTranspose to a `reverse_kernel` (flips the kernel
spatially) + `layout_kernel`, then a *plain forward convolution* with
adjusted padding -- the standard transposed-conv-as-flipped-conv
identity -- with several of these small convs `concat_kernel`'d back
together for some of the test shapes.** The forward conv itself never
reaches `ConvHipDirectFwd` or any `ConvDirectNaiveConvFwd`/naive solver at
all for these shapes: MIOpen's own Find log shows
`[EvaluateInvokers] Selected: GemmFwdRest` (im2col via
`MIOpenIm2d2Col`/`Im2d2Col_v2` + `[CallGemm] rocBLAS`) and, for a 1x1-shaped
sub-piece, `GemmFwd1x1_0_1` (direct GEMM, no im2col) -- both explicitly
**skip** `ConvDirectNaiveConvFwd` once a faster GEMM solver is found
(`"Skipping Naive Solver"` in the log). **The `ConvHipDirectFwd`
hypothesis above was wrong** -- that solver family is never even in
contention for these shapes; no point pursuing it further for this bug.

This redirects the investigation to two real candidates instead, both
narrower than "rocBLAS GEMM" broadly (already cleared -- the WGM/small-gemm
ablation test above showed ConvTranspose's mismatch count is unaffected
by those two patches, so the rocBLAS GEMM math itself is very likely not
where this lives):

1. **`MIOpenIm2d2Col`/`Im2d2Col_v2`** -- the im2col packing kernel
   `GemmFwdRest` depends on to turn the conv into a GEMM in the first
   place. A packing bug here would feed a correct rocBLAS GEMM call
   wrong operands and look exactly like this: a real numeric result,
   structurally wrong, not garbage.
2. **MIGraphX's own `reverse_kernel`/`layout_kernel`/`concat_kernel`
   glue** -- the flip-and-stitch machinery around the forward conv, which
   is MIGraphX-side code, not MIOpen's. The `concat_kernel` stitching
   several small conv outputs back into one tensor is a plausible source
   of the "some output positions get exactly zero instead of a real
   contribution" pattern already recorded above.

**Isolated (2026-08-10): MIOpen's im2col+GEMM path is cleared, bug is in
MIGraphX's own glue.** MIOpen logs its own exact driver command for the
underlying forward conv
(`MIOpenDriver conv -n 1 -c 1 -H 3 -W 3 -k 2 -y 1 -x 2 -p 0 -q 1 -u 1 -v 1
-g 1 -F 1`) -- ran it standalone, with `-V 1` (verify against MIOpen's own
GPU-vs-CPU-reference check), completely bypassing MIGraphX/ORT. **Both
the asymmetric (`-p 0 -q 1`, the actual `test_convtranspose_pads` config)
and symmetric (`-p 1 -q 1`) padding variants verify clean**
(`Forward Convolution Verifies OK on GPU reference`, error ~1e-8, solver
`GemmFwdRest` both times -- same solver the real test uses). MIOpen's
im2col (`MIOpenIm2d2Col`) and the rocBLAS GEMM it feeds are correct in
isolation for this exact shape/algo/padding.

**This narrows it to MIGraphX's own `reverse_kernel` (weight flip),
`layout_kernel` (permutation), or `concat_kernel` (stitching multiple
small conv outputs back together) -- not MIOpen, not rocBLAS.** Next
step: read MIGraphX's ConvTranspose parsing/lowering source (likely
`src/onnx/parse_convolution.cpp` or a dedicated `parse_conv_transpose`
path, plus whichever `.cpp` implements the `reverse`/`concat` ops) at the
pinned `rocm-libraries` commit and check the padding/output-shape math
feeding into the flip-and-conv-and-stitch sequence against ONNX's own
ConvTranspose semantics -- this is where the actual defect lives, not a
kernel-correctness bug at all at this point, more likely a lowering/glue
logic bug (wrong flip axis, wrong stitch offset, or a padding
miscalculation feeding a technically-correct conv the wrong inputs).

### Source-level dig (2026-08-10): `test_convtranspose_autopad_same` isn't a regression; the real 4 need a deeper pass than expected

Cloned `AMDMIGraphX` at both `rocm-7.14` and `rocm-6.4.4` tags to diff the
actual ConvTranspose code paths instead of continuing to guess.

**`test_convtranspose_autopad_same` reclassified: not a regression.**
`src/onnx/parse_conv_transpose.cpp`'s `auto_pad` handling is **entirely
new in 7.14** -- on `rocm-6.4.4` that branch is just
`MIGRAPHX_THROW("PARSE_CONV_TRANSPOSE: auto padding not supported")`.
Since 6.4.4's ORT registers both MIGraphX and ROCm EP, a MIGraphX parse
exception there means ORT silently used the ROCm EP fallback for this
node -- 6.4.4 never actually exercised MIGraphX's conv-transpose
auto-pad path at all. Same category as the Attention tests: new
functionality with a bug, not lost ground. **Down to 4 real regressions,
not 5**: `kernel_shape`, `output_shape`, `pad`, `pads`.

For those 4: the parser (`parse_conv_transpose.cpp`) is **byte-identical**
between the two tags outside the new auto_pad block -- confirmed via
diff, not assumed. So the bug isn't in ONNX parsing; it's in GPU
execution, consistent with the isolation test above.

**Traced into `src/targets/gpu/`, ruled out the obvious places:**
- `lowering.cpp`'s registration of `"convolution_backwards"` (which
  `apply_map`-registers a handler; a duplicate/dead second registration
  for the same key exists later in the file, `add_convolution_backwards_op()`,
  a CPU-fallback path that copies through `hip::copy_from_gpu`/
  `hip::copy_to_gpu` and never actually runs since `emplace` doesn't
  overwrite -- confirmed harmless red herring, this ordering is identical
  on both tags, so it isn't a 6.4.4-vs-7.14 difference either way).
- `include/migraphx/gpu/miopen.hpp`'s `make_convolution_backwards()` --
  builds a real native MIOpen transpose descriptor (`miopenTranspose`
  mode), unchanged between tags. **But this isn't actually the path being
  taken**: the trace above shows the op MIOpen actually receives by the
  time it reaches `gpu::miopen_op` is already a plain `gpu::convolution`
  (forward), not `gpu::convolution_backwards` -- something rewrites
  ConvTranspose into reverse+forward-conv+concat *before* the native
  transpose path in `miopen.hpp` is ever reached, making that native path
  dead for this scenario. **Have not yet located the pass that does this
  rewrite** -- not in `lowering.cpp` or `miopen.hpp`; likely an
  optimization/simplify pass elsewhere in `src/` (target-independent, not
  under `src/targets/gpu/`) or inside `op::convolution_backwards`'s own
  compile-time logic. This is the next concrete thing to find and diff
  between the two tags -- that's where the actual defect almost certainly
  lives, since everything checked so far (parser, MIOpen's own conv,
  registration order, the native-transpose path) is either identical
  between versions or independently verified correct.

### The pass, found: `src/rewrite_convolution.cpp`, brand new, NOT gfx803-hardware-specific (2026-08-10)

Located the rewrite: `src/rewrite_convolution.cpp`, registered
unconditionally in `optimize_rewrite_pipeline()`
(`src/targets/gpu/target.cpp`). It decomposes `convolution_backwards`
(ConvTranspose) into MIOpen's implicit-GEMM backward-data "v4r1"
algorithm reimplemented as a MIGraphX graph: split the filter into
residues, run each as a stride-1 forward convolution (flip via
`reverse`, strided tap-slice via `slice`/`step`), then either
`reassemble_interleave` (concat + reshape/transpose, the no-dilation
fast path) or `reassemble_general` (zero-stuff + pad + add) back
together, then crop.

**Confirmed via direct diff: this file does not exist at all on
`rocm-6.4.4`** (`git show <6.4.4-tag>:src/rewrite_convolution.cpp` ->
"exists on disk, but not in" that commit). Entirely new code added
between the two lines, not something gfx803 has ever exercised before.

**Confirmed NOT hardware-gated to exclude gfx803 or include gfx1201 --
the reverse is true, and for an unrelated reason.** `fuse_mlir.cpp` has
this exact carve-out:

```cpp
// gfx12 lacks an accurate half version of MIOpen convolution_backwards path, so
// always route it through rocMLIR regardless of MIOpen availability or user
// env-var overrides.
mlir_mode conv_backwards_mode =
    get_mode("convolution_backwards", MIGRAPHX_USE_MIOPEN ? mlir_mode::none : mlir_mode::all);
if(starts_with(device_name, "gfx12")) { conv_backwards_mode = mlir_mode::all; }
```

gfx1201 (gfx12 family) **always** routes `convolution_backwards` through
rocMLIR -- a real hardware-fused kernel, bypassing `rewrite_convolution.cpp`
entirely -- because of a *different*, already-known MIOpen accuracy gap on
gfx12, unrelated to this bug. gfx803 (MIOpen available, not gfx12) always
gets `mlir_mode::none` and falls straight into the new software
decomposition. **This means the gfx1201 comparison run earlier cannot
tell us whether `rewrite_convolution.cpp`'s math is actually correct** --
gfx1201 never executes this code path at all, clean or not.

**Working conclusion, not yet certain**: this reads as a genuine defect
in MIGraphX's own new v4r1 decomposition logic -- not a gfx803 hardware
quirk -- that gfx803 is exposed to only because it's forced onto this
path (MIOpen-capable, non-gfx12) while gfx1201 is carved out onto a
different one for an unrelated reason. **This is the same class of bug
this whole repo already treats as in-scope** (see `wgm-miscompute.sh`,
`small-gemm-assembly-miscompute.patch` -- both are Tensile/rocBLAS logic
bugs "exposed" by gfx803's kernel selection, not narrow hardware defects,
and both are shipped patches) -- but unlike those, this bug plausibly
also affects every other non-gfx12 MIOpen architecture (gfx900, gfx906,
gfx908, gfx90a, gfx942...) using this exact MIGraphX build, since nothing
in the gating is gfx803-specific. **Before writing a patch**: this should
probably be reported upstream to `ROCm/AMDMIGraphX` regardless of what
this repo does locally, since a local gfx803-only patch wouldn't help any
other affected architecture and the fix belongs in the shared pass, not
a gfx803 patch tree.

**Not yet pinpointed to an exact line.** The `no_upsample`/`interleave`/
`reassemble_general` branch split, the residue reassembly's
reshape/transpose permutation math, and the final crop step (which
must correctly compose with the *parser's own separate* asymmetric-padding
slice in `parse_conv_transpose.cpp` for the asym-padding test cases) are
all candidates. Next step if this is pursued further: hand-derive
`reassemble_interleave`'s reshape/transpose sequence against a small
known-correct example (e.g. stride=2, kernel=2, dilation=1) and check
where it diverges from ONNX's own ConvTranspose reference math, or find
a minimal single-residue (stride=1, `no_upsample`) case among the 4
failing tests to rule that whole reassembly step out first if one
exists.

### Confirmed upstream, one level deeper (2026-08-10)

Checked MIGraphX's own git history and unit tests for this pass instead
of reasoning further from our own repro alone. `src/rewrite_convolution.cpp`
has exactly **one** commit touching it ever: `4bcfe75 "Convolution
backwards v4r1 (#4928)"`, merged 2026-07-10 by an AMD engineer ("Make for
better performance of convolution_backwards by changing them into
forward convolutions"), no fix/follow-up commits since on `develop`.

It shipped with its own dedicated unit test suite
(`test/rewrite_convolution_test.cpp`, 405 lines, ~15 cases + a 54-line
numeric multi-residue test) -- **every single test case uses symmetric
padding** (`{0}`, `{2,2}`, `{1,1}`, etc.). This isn't a coincidence: the
`convolution_backwards` op's own `padding` value is always symmetric by
construction from the parser's side -- for asymmetric ONNX padding,
`parse_conv_transpose.cpp` zeroes the op's padding entirely and bolts on
a **separate** `slice` instruction *after* whatever this pass produces
(see the parser diff/discussion above). **The exact combination
`test_convtranspose_pads` hits -- this pass's interleave reassembly,
followed by the parser's independently-inserted asymmetric-padding slice
-- is never exercised by any test in the commit that introduced this
code.** That's a real, confirmed gap in upstream test coverage, not
speculation.

**Conclusion: this is upstream's problem, not gfx803's, and not this
repo's to patch.** Brand new code (doesn't exist on 6.4.4), gated onto
gfx803 only because gfx803 lacks the gfx12-only MLIR carve-out (unrelated
reason), with a real, confirmed coverage hole in its own test suite
matching the exact failing case. A gfx803-only patch here would fix
nothing for the other non-gfx12 MIOpen architectures hitting the same
bug and isn't this project's job. **Decision: do not patch. Leave as a
known, tracked gap** (`kernel_shape`/`output_shape`/`pad`/`pads`
ConvTranspose tests will keep failing on this line until upstream fixes
`rewrite_convolution.cpp`, or MIOpen/MLIR coverage improves for gfx803).
Worth filing upstream if/when this project has bandwidth for that, but
not blocking.

**This is reasoning from code + failure-pattern shape, not confirmed** --
the actual next step is empirical: rebuild with these four correctness
patches removed (keeping the Layer-1 doorbell/dispatch fix, which is
load-bearing for the whole line, and the three build-only fix patches,
which fix compile failures unrelated to numeric correctness) and rerun
these 7 tests. See "Clean build, no ported correctness patches" below.

### Clean build, no ported correctness patches (2026-08-09) -- empirical result

Built a throwaway variant (`rocm6.4.4/rocm7-clean-experiment/`, not committed,
delete after use) of this Dockerfile with the four *correctness* patches
(`rocblas/wgm-miscompute.sh`, `rocblas/small-gemm-assembly-miscompute.patch`,
`miopen/winograd-fused-conv-miscompute.patch`,
`miopen/reduce-prod-wrong-identity.patch`) each swapped for `RUN true #
SKIPPED`, keeping everything else identical: the Layer-1 doorbell/dispatch
fix (`hsa-agent-rejects-legacy-doorbell`, `opencl-gfx8-hardcoded-rejection`
-- load-bearing for the whole line, not a correctness patch) and the three
build-only fix patches (`gfx-default-rocblas-hipblaslt-off-build-failure`,
`mlir-stub-missing-symbols`, `tensortopk-hip-build-oom` -- fix compile
failures, not runtime correctness). Full rebuild (cache miss on every
stage from a fresh directory), all stages succeeded, all final-stage
import/version sanity checks passed identically to the patched build.
Tagged `gfx803-rocm7:clean-noPatches`, transferred to the gfx803 box the
same way as before (`docker save | ssh | podman load`), ran the exact 7
gfx803-only tests directly (loading each `model.onnx` + comparing against
its `test_data_set_0` reference, same `rtol=1e-3, atol=1e-5` ORT itself
uses) against `MIGraphXExecutionProvider`.

**Result: all 7 still fail with the four correctness patches removed --
but the two clusters respond completely differently, which is the
useful part:**

| test | patched (mismatch/total) | unpatched (mismatch/total) |
|---|---|---|
| convtranspose_autopad_same | 15/72 | 18/72 |
| convtranspose_kernel_shape | 45/160 | 45/160 |
| convtranspose_output_shape | 45/160 | 45/160 |
| convtranspose_pad | 45/160 | 44/160 |
| convtranspose_pads | 12/42 | 9/42 |
| attention_3d_gqa_..._expanded (output[0]) | 40/576 | **575/576** |
| attention_4d_gqa_..._expanded (output[0]) | 40/576 | **575/576** |

- **ConvTranspose: the four patches have essentially zero effect** --
  mismatch counts are within noise of each other with or without them.
  **Ruled out**: `wgm-miscompute`, `small-gemm-assembly-miscompute`,
  `winograd-fused-conv-miscompute`, `reduce-prod-wrong-identity` are not
  the cause. This redirects the investigation squarely to the
  `conv-direct-fwd-grouped-oob`/`ConvHipDirectFwd` hypothesis already
  written up above (MIGraphX's deconvolution lowering itself, or a MIOpen
  solver neither this line's rocBLAS nor MIOpen correctness patches touch)
  -- next step should start there, not by tweaking rocBLAS further.
- **Attention GQA expanded: the opposite result** -- removing the patches
  made it dramatically worse (7% wrong -> 99.8% wrong, essentially the
  entire tensor). The rocBLAS patches (most likely `wgm-miscompute`, given
  the tile-boundary divergence shape noted above, though this run doesn't
  isolate which of the two rocBLAS patches specifically -- both were
  removed together) **are partially load-bearing for this bug**, but
  insufficient alone -- there is a second, still-unfixed defect in the
  same code path that they don't cover. Worth isolating which of the two
  rocBLAS patches is doing the work (re-run with only one removed at a
  time) before starting a fresh investigation into the residual 7%.

### Correction (2026-08-10): what "on par with 6.4.4" actually means here

6.4.4's ORT is built with **both** `--use_rocm` and `--use_migraphx`
(`rocm6.4.4/Dockerfile` around line 795-808) -- MIGraphX handles what it can,
**ROCm EP is a real fallback for everything else**. 7.14's ORT (v1.28.0)
has no ROCm EP at all -- deleted upstream, confirmed earlier in this file
-- MIGraphX is the only GPU path. This matters more than the gfx1201 diff
above captured, because gfx1201 is *also* MIGraphX-only under the same
ORT/rocm7 pin -- the gfx1201 comparison only separates "generic MIGraphX
gap, not this line's fault" from "this port broke something," it does
NOT answer "is this a regression from 6.4.4."

Ran the same 183 gfx803-rocm7 failing/erroring tests directly against
`gfx803-full-test:v13` (6.4.4, `onnxruntime==1.22.2`, default (full,
unforced) provider list so ROCm EP fallback is live) to check that
properly:

- **110 of 183 pass on 6.4.4** (0 numeric failures, 0 runtime exceptions
  among everything that loaded) -- these are real capability losses
  between 6.4.4 and 7.14, but **not gfx803-specific and not patchable
  here**: they're paid because ORT upstream deleted ROCm EP entirely in
  1.28, which costs every MIGraphX-only architecture identically
  (gfx1201 pays the same price -- it has no ROCm EP either, same as
  gfx803 on this ORT version). Nothing to fix in this repo; it's the
  price of the ORT version bump, out of scope for "gfx803 vs. mainline
  rocm7 parity."
- **64 fail to load at all on 6.4.4** (`Unsupported model IR version: 11,
  max supported: 10`) -- almost entirely the `Attention`-op family
  (opset 23, IR version 11), which didn't exist as ONNX-expressible
  functionality when 6.4.4's ORT was built. **Not a regression -- there
  is no 6.4.4 baseline for these at all**, new-and-still-buggy is a
  different category than broken-was-working.
- **The 5 ConvTranspose node tests (`autopad_same`, `kernel_shape`,
  `output_shape`, `pad`, `pads`) all pass cleanly on 6.4.4.** This is
  the one confirmed, real, in-scope regression from 6.4.4 -- gfx803 used
  to get these right and doesn't now. Everything else in the 183 is
  either an upstream-driven cost paid by every arch (110) or new
  functionality with no prior baseline (64, including both
  `attention_..._gqa_with_past_and_present_expanded` tests -- their
  models also fail to even load on 6.4.4's ORT, same IR-version wall).

**Bottom line**: "on par with 6.4.4" for this line, precisely, means
fixing the 5 ConvTranspose tests. The attention_gqa `_expanded` numeric
bug is real but is new functionality with a bug, not a parity gap --
lower priority than framed earlier in this file.

### ConvHipDirectFwd -- is this actually in scope?

The `conv-direct-fwd-grouped-oob` gap (see "Layer 4: MIOpen" above) is
**not** an "upstream problem, leave it alone" situation, despite the
"NOT YET RE-VERIFIED" language possibly reading that way. The original
6.4.4 bug (`ConvOclDirectFwd*` family, grouped-conv OOB weights read) was
always *our* patch, not an upstream fix -- gfx803 is unsupported upstream,
nobody else maintains correctness for it. What changed is that the
specific solver that bug lived in (`ConvOclDirectFwd*`, OpenCL-source)
was deleted and replaced by a **different implementation**
(`ConvHipDirectFwd`, HIP-source, `src/solver/conv/conv_hip_dir2Dfwd.cpp`)
-- "unverified" means nobody has run the original repro against the new
solver yet, not that this is somehow off-limits. If ConvTranspose's
regression traces there, fixing it is exactly the same kind of
gfx803-only patch as everything else in this repo -- squarely in scope,
same as the four correctness patches already carried over. It is *not*
yet confirmed that ConvTranspose actually routes through this solver
though -- that's the first thing to check before assuming this is the
right lead, not a hypothesis to patch against blind.

**Next actionable step**: two separate investigations now, not one --
(1) ConvTranspose via `ConvHipDirectFwd`/MIGraphX deconv lowering,
patches confirmed not responsible; (2) Attention GQA's residual 7%
wrongness with patches applied, via isolating `wgm-miscompute` vs.
`small-gemm-assembly-miscompute` individually, then the same
real-model-vs-CPU-EP bisection methodology as `KERNEL_BUGS.md`. Both use
the reproduction harness already written for this
(`repro_7tests.py`-style: load `model.onnx` from onnx's own test data
directly against `MIGraphXExecutionProvider`, compare to
`test_data_set_0`) rather than the full 3828-test suite, since these two
clusters are the only tests that matter here.

## correctness-suite -- run against `gfx803-rocm7:full-poc` on real hardware (2026-08-09)

All 23 sweeps pass, 0 real WRONG (`groupnorm_sweep`'s "2 WRONG" are its
own documented expected-rejection boundary cases, not a bug -- see its
log): 13 op-level sweeps (activ, pool, bn, softmax, layernorm, groupnorm,
tensorop, reduce, reduce_extreme, glu, cat, rope, kthvalue) + 10 conv
solver sweeps (ConvAsm1x1UV2, ConvAsm3x3U, ConvAsm5x10u2v2f1,
ConvAsm7x7c3h224w224k64u2v2p3q3f1, ConvBinWinograd3x3U, ConvBinWinogradRxS,
ConvDirectNaiveConvFwd, ConvOclDirectFwd11x11, ConvOclDirectFwdGen, fft).

**Caveat, confirmed not just speculated**: `ConvOclDirectFwd11x11` and
`ConvOclDirectFwdGen` "pass" but aren't testing what their name claims --
those solvers were deleted from MIOpen on 7.14 (see
`conv-direct-fwd-grouped-oob` above), so
`MIOPEN_DEBUG_FIND_ONLY_SOLVER=<that name>` can't find them and MIOpen
silently falls through to whatever solver it'd normally pick for that
shape instead -- no error, no indication in the sweep's own output that
the forced solver didn't apply. The PASS result is real (whatever solver
actually ran produced correct output) but doesn't exercise the specific
solver those two shape files were written to target. Not a correctness
bug in this build; a coverage gap in the suite's own solver-forcing
mechanism now that those solver names are stale on this line. Low
priority to fix (shapes/*.txt for solvers that still exist cover the
same math paths); noted here so it isn't mistaken for real ConvOclDirectFwd
coverage later.

## Real-model validation: faster-whisper/CTranslate2 (2026-08-10)

Cross-project context: `audiomuse-rocm-plugin` (sibling project, same author)
already did a deep gfx803 investigation for ASR backends against the 6.4.4
line, documented in its own `docs/ASR_BACKENDS.md`/`ARCH_NOTES.md`. On 6.4.4
gfx803 needed `arlo-phoenix/CTranslate2-rocm` (an unmaintained ROCm 6 fork)
because upstream CTranslate2's release wheel wants ROCm 7's `libhipblas.so.3`.
Found there: `float16` is the only correct compute type on that fork --
`float32` produces multilingual token-salad garbage ("the resurrected r9nano
Tensile logic" -- the exact same rocBLAS bug class this line's
`wgm-miscompute`/`small-gemm-assembly-miscompute` patches target) and
`int8_float32` silently returns empty text.

Since this line is now properly on ROCm 7.14, tried **upstream
`OpenNMT/CTranslate2` (v4.8.1, the same version/config the gfx900+ line
already uses) instead of the fork** -- no fork needed once the base ROCm
matches what upstream expects. Built against `gfx803-rocm7:va-reuse-defer`
with `-DWITH_HIP=ON -DWITH_CUDNN=OFF -DCMAKE_HIP_ARCHITECTURES=gfx803`
(identical to what `audiomuse-rocm-plugin/docker/Dockerfile` uses for
Vega/CDNA arches), compiled clean.

**Upstream's Conv1D is a completely different implementation than the
fork's** -- a custom im2col kernel + generic `gemm_batch_strided` (straight
rocBLAS), not MIOpen's convolution API at all. The fork's
`conv1d-workspace-cap.patch` (MIOpen `GetWorkSpaceSize` reporting a bogus
~1.4GB, capped to 256MB) **does not apply here** -- upstream never calls
that MIOpen API for Conv1D, so there's no equivalent bug to port.

Real transcription test (JFK sample, `tiny.en`, `beam_size=5`, matching
the plugin's own known-transcript methodology), on the real gfx803 box,
`CT2_CUDA_ALLOCATOR=cub_caching` set (same allocator-page-fault fix the
plugin found necessary for every arch, unrelated to gfx803 specifically):

- **`float16`: correct transcript.**
- **`float32`: correct transcript** -- same exact text. Slow (multi-minute
  for an 11-second clip -- confirmed via `AMD_LOG_LEVEL=3` trace to be real
  sustained Tensile GEMM kernel dispatch, ~950 kernel launches in 45s, not
  a hang/deadlock -- just genuinely low fp32 throughput on Polaris, a
  performance characteristic, not a bug), but **no longer garbage**. This
  is strong independent evidence (different codebase, different workload,
  found by a different investigation) that `wgm-miscompute.sh`/
  `small-gemm-assembly-miscompute.patch` fixed the real underlying rocBLAS
  defect the fork's docs pinned as "r9nano Tensile logic," not just the
  narrow repro those patches were originally verified against.
- **`int8_float32`: correct transcript** -- silently-empty on the old fork,
  works cleanly here.

**All three compute types are now correct on this line, via upstream
CTranslate2, no fork required.** Two Python-env issues fixed along the way
(unrelated to gfx803): `httpx`/`click`/`hf-xet` missing from
`huggingface_hub`'s modern dependency set, and an OpenMP double-init
(`KMP_DUPLICATE_LIB_OK=TRUE` needed -- CT2's bundled `libomp.so` collides
with torch/onnxruntime's own copy already in the image).

## Real-model validation: whisper.cpp and parakeet.cpp HIP (2026-08-10)

Same methodology as the CTranslate2 section above -- built standalone
against `gfx803-rocm7:va-reuse-defer`, tested with real audio on the
actual gfx803 box, compared against `audiomuse-rocm-plugin`'s documented
6.4.4 findings (`docs/ASR_BACKENDS.md`).

**`whisper.cpp` (`GGML_HIP=1 AMDGPU_TARGETS=gfx803`, `v1.9.1`, tiny.en,
JFK sample): correct transcript, real `ROCm0` GPU dispatch confirmed,
815ms total.** Matches the 6.4.4 finding exactly (already documented
there as "correct transcript, GPU used") -- no regression, nothing to
fix. (The plugin still ships Vulkan as the *default* for whisper.cpp on
gfx803 for unrelated reasons -- smaller images, one fewer backend to
maintain -- not because HIP doesn't work; the docs' own "Verdict" line
reads as slightly stale/inconsistent against its own detailed findings
two paragraphs above it.)

**`parakeet.cpp` (`PARAKEET_GGML_HIP=1`, `v0.4.0`, Parakeet-TDT-0.6B-v2,
JFK sample): correct transcript, real `ROCm0` GPU dispatch, confirmed
deterministic across two independent runs.** This directly **contradicts**
the 6.4.4-era finding of silent empty output on GPU (same doc: "silent
**empty** output on GPU, `exit 0`, no error... same input on CPU: perfect
transcript") -- that finding is exactly why `audiomuse-rocm-plugin`'s own
Dockerfile currently skips building parakeet.cpp HIP for gfx803 entirely
(`gfx803|gfx802|gfx805) echo "parakeet.cpp HIP skipped..."`). **On this
line it now just works.** The 6.4.4 doc itself flagged this result as
"needs re-testing" (written before the faster-whisper allocator/workspace
fixes landed, which may have shared root causes) -- this rocm7 result is
consistent with whatever was broken there having been fixed, either by
this port's own patches or by parakeet.cpp/ggml itself moving forward
since that probe ran. Not root-caused which -- this is a positive result
worth flagging back to `audiomuse-rocm-plugin` to re-enable parakeet.cpp
HIP for gfx803 there too, since nothing here explains *why* it's fixed,
only that it demonstrably is on this stack.

**Bottom line for ASR on rocm6.4.4/rocm7**: faster-whisper (upstream
CTranslate2, all 3 compute types), whisper.cpp HIP, and parakeet.cpp HIP
all produce correct real-model transcripts with real GPU dispatch on
this line. Full turnaround from every partial/broken result documented
for the 6.4.4 fork-based stack.

## Open items / not yet done

- [ ] Full local build of `Dockerfile` end-to-end (written,
      not yet built).
- [ ] Real-hardware validation on the gfx803 box (192.168.1.214,
      `/data/rocm7`) -- rocminfo enumeration first (the actual Layer-1
      claim under test), then the correctness-suite.
- [ ] `conv-direct-fwd-grouped-oob` gap above -- needs a fresh
      investigation against `ConvHipDirectFwd`, not a port.
- [ ] `reduce-program-bound-eviction` re-diff -- low priority, opt-in only.
- [ ] `TORCHVISION_REF`/`TORCHAUDIO_REF` (`v0.28.0`/`v2.13.0`) are
      extrapolated from the torch-fork version-offset pattern, not looked
      up against a real compatibility matrix -- confirm or correct once
      the PyTorch build stage actually runs.
