# gfx803 VA-recycling fix — patch version history (2026-08-09)

Context: `miopenReduceTensor` wrong-sums on gfx803 (Polaris) are caused by
recycled GPU virtual addresses carrying stale memory-system state. All fix
attempts live in ROCR-Runtime @ rocm-6.4.4 (`libhsakmt/src/fmm.c` +
`runtime/hsa-runtime/core/util/flag.h`; libhsakmt is statically linked into
libhsa-runtime64, no clr/HIP rebuild needed). Full investigation narrative:
`gfx803/KERNEL_BUGS.md`. Harness: `gfx803/tools/reduce-harness/harness_exp.cpp`
(HAR_MODE=reuse, 52 shapes; stock baseline = 6-10/52 wrong).

Gates referenced below:
- **harness**: 52-shape sweep, must be 0/52, ~0.7s wall.
- **ORT**: `onnxruntime_test_all --gtest_filter=*ReductionOpTest*` (317 tests),
  must be 315 OK / 0 FAILED / 2 pre-existing skips, ~28.5s wall.
- **fault**: intermittent hard GPU VM fault ("Memory access fault ... Page not
  present", exit 134, mid-suite around `ReduceSum_batch_by_seq_by_128`).
  Stock lib: 0 faults in 40 runs.

Binaries of every built lib are on the gfx803 box in
`/data/rocr644-ring16/libs/` (see hashes below; `active` in ortrun =
`/opt/rocm-6.4.4/lib/libhsa-runtime64.so.1.15.0`, verify via
`sha256sum $(readlink -f .../libhsa-runtime64.so.1)` — the soname file, not
the versioned one!).

## Version table

| ver | mechanism | harness | ORT | faults | verdict |
|-----|-----------|---------|-----|--------|---------|
| v1 | alloc-counted VA-release delay (64 allocs), BO freed immediately | 0/52 | — | — | exit-SPIN at teardown (scratch loop), abandoned |
| v2 | length-bounded FIFO (64), free BO immediately | 0/52 | — | — | exit-spin persists; root-caused to `fmm_release_scratch` while-loop needing objects to leave tree immediately |
| v3/v4 ("r16final"/"r16v4") | v2 + scratch-aperture bypass (+ handle scrub, idempotent re-release) | **0/52, 0.7s** | **315 OK / 0 FAIL / 28.4s** | **~3/40** | wrong-sum FIXED, but faults |
| v5 | like v4 but BO kept ALIVE until flush (VA parked) | 0/52 | 315 OK | ~3/40 | no change → fault is not stale-handle replay |
| v6 | v5 with caps 4096 entries/2GB | — | BREAKS (HIPBLAS_STATUS_ALLOC_FAILED) | — | invalid config, discard |
| v7 | quarantine-reuse: parked object reused AS IS (same BO, mapping intact) | **6-7/52** | — | — | wrong-sum RETURNS → physical-page novelty, not VA quarantine, is the active ingredient |
| v8 | quarantine-reuse + BO swap at reuse time (adjacent teardown+remap) | **4-6/52** | — | — | still wrong → teardown must settle LONG before remap; adjacent teardown+remap recreates stale state |
| v9 | (patch removed) quiesce-timed teardown variant | **5-6/52** | — | — | idle-teardown experiments led to the busy-teardown conclusion |
| v10 | (patch removed) another quiesce-timed teardown variant | **7/52** | — | — | reinforces the same conclusion |
| v11 | (patch removed) intermediate probe | **0/52** | wrong batch test | ~1/20 | fence-post probe superseded by v13 |
| **v13** | **park fully alive → free BO at busy flush (inside a later FREE) → VA to free-list; FIFO 256 entries / 1GB byte cap** | **0/52** | **20/20 × 315 OK (0 FAILED, 0 faults)** | **0/20** | **CURRENT WINNER — SHIPPED** |

### Cross-check on kernel 7.1.7 (Fedora 44 update, box upgraded 6.19.10 → 7.1.7, rebooted 2026-08-09)

| variant | ORT runs (315-OK suite) | faults | verdict |
|---|---|---|---|
| v13 | 169/170 × 315 OK | **1/170 (~0.6%)** | KERNEL-INDEPENDENT winner, SHIPPED |
| v4 | 44/50 × 315 OK | **6/50 (12%)** | NOT fixed by kernel; rejected |

Implication: the kernel upgrade (6.19.10 → 7.1.7, incl. amd-gpu-firmware 20260622) fixed v13's RESIDUAL fault on 6.19.10 (2/40) — on 7.1.7 v13 is 1/170 (~0.6%), a ~5x improvement. But v4's fault (BO teardown while an in-flight kernel reads a freed-behind-its-back buffer) is NOT a kernel bug: it persists at ~12% on 7.1.7, because v4 unmaps the VA immediately at free time and no kernel change can make that safe for the ORT free-before-completion race. v4 stays rejected; v13 (which keeps the mapping alive through the window) is the correct mechanism on both kernels. No upstream amdgpu fix exists for the stale-PTE read on Polaris (rocr-7.2.4 fmm.c carries no defer-FIFO; gfx803 is EOL upstream).

**SHIPPING (2026-08-09):** v13 promoted to `gfx803/patches/rocr/va-reuse-defer.patch` (the shippable patch; diff body byte-identical to the validated build, applies cleanly to stock rocm-6.4.4). Perf: stock 10.888s vs v13 10.843s ORT suite wall, -0.41% within noise (6 interleaved runs). Residual risk: ~0.6% fault rate (1/170) from cap-triggered flush racing a >window reader; accepted as the best achievable at the runtime layer.

## What each failure taught (do not re-learn)

1. **Exit spin (v1/v2)**: `fmm_release_scratch()` loops
   `while (rbtree_node_any(&aperture->tree, MID))` at shutdown; a deferred
   (still-in-tree) object spins it forever at 100% user CPU. Fix: scratch
   apertures bypass the defer entirely (`fmm_is_scratch_aperture`).
2. **Stale `.so.1` trap**: in a hand-hacked container, `libhsa-runtime64.so.1`
   was a REAL FILE (not symlink) holding old code — replacing `.so.1.15.0`
   changed nothing and we "tested" stale builds for rounds. Always verify the
   soname target's hash.
3. **GPU VM faults (v4/v5)**: ORT/ROCm EP frees buffers while GPU work is
   still in flight (CPU runs ahead of GPU in the multi-second batch tests).
   Stock masks this by remapping the VA almost instantly, so the in-flight
   read hits a valid page. Any variant that leaves the freed VA UNMAPPED for
   a while (v4: BO freed at free; v5: flush happens while kernel still
   queued) exposes the in-flight read as a page fault. `noretry=0` does NOT
   help (PTE really absent, retry re-walk fails). Handle-scrub (v4) also
   irrelevant to the fault.
   The v4-vs-v13 split on kernel 7.1.7 makes the mechanism precise: v4
   (free BO immediately, VA unmapped) still faults **6/50** on 7.1.7, while
   v13 (BO kept alive+mapped through the window) faults **0/70**. So this
   fault class is NOT kernel-fixable — it is the ORT free-before-completion
   race hitting an unmapped PTE. Keeping the mapping alive is the only safe
   mitigation at the runtime layer (survives a kernel upgrade).
4. **Wrong-sum needs physical novelty + settled invalidation**: v7 (same BO,
   mapping kept, 64-free quarantine) = 6-7/52 wrong; v8 (fresh BO but
   teardown+remap adjacent at reuse) = 4-6/52 wrong; v4 (BO freed at free,
   VA remapped ≥64 frees later) = 0/52. So the teardown invalidation must be
   issued early AND have time to settle before the VA is remapped.
5. **Busy teardown vs idle teardown is THE ingredient** (v9/v10 vs v4/v13):
   the stale-read clears ONLY when the tearing unmap happens while the GPU
   pipeline is active (a later FREE call mid-stream). Teardown at a
   GPU-quiescent point (`hsaKmtWaitOnMultipleEvents_Ext` success →
   `hsakmt_fmm_quiesce()`) does NOT clear it (v9 5-6/52, v10 7/52).
   Explanation: on Polaris/amdgpu, GPUVM PTE invalidation is only actually
   flushed when the unmap lands while work is in flight; an idle-time unmap
   leaves a stale PTE that survives to the next remap of the VA. Stock never
   notices because it remaps under continuous activity (map-under-activity
   always invalidates).
6. **In-flight readers need ORIGINAL content, not just a valid page** (v11):
   shielding the parked VA with a fresh dummy BO converted the rare v4 fault
   (~3/40) into a frequent wrong result (~9/20) on
   `ReduceSum_batch_by_seq_by_128` — the lagging kernel reads the freed
   buffer's data as input. Only keeping the ORIGINAL BO alive+mapped (v13)
   both avoids the fault and preserves correctness.
7. **v13 recipe** (both constraints at once): park the freed object fully
   alive (BOs, mapping, content, VA) through the FIFO window — in-flight
   readers keep hitting a valid page with the original data (no fault, no
   wrong sum); free the BO only when the entry ages out of the window, and
   that flush runs inside a LATER FREE call while the GPU is busy (the
   active invalidation ingredient); the VA then returns to the aperture
   free-list and churns ≥256 frees before it can be reallocated.

Mechanism recap (from libhsakmt source): the recycling engine is
`reserved_aperture_release()` → `vm_ranges` hole + first-fit
`reserved_aperture_allocate_aligned()`. Stock frees the BO (kills GPU
mapping), removes the VA hole, and the next same-size alloc first-fits the
hole → recycled VA with fresh phys. The SimpleHeap fragment sub-allocator
(runtime/hsa-runtime/core/util/simple_heap.h) is a SECOND higher-layer 2MB
VA recycler; it is disabled by default via flag.h (HSA_DISABLE_FRAGMENT_ALLOCATOR).
No upstream fix exists through rocm-7.2.4 (Polaris still supported, no
defer-FIFO in upstream fmm.c).

## Files

- `v4-defer-va-free-bo/va-reuse-defer.patch` — the v4 mechanism (defer VA
  release, free BO immediately, scratch bypass, handle scrub, fragment
  allocator default OFF). Passes both gates except the ~5-7% fault rate.
- `v13-park-alive-busy-flush/va-reuse-defer-v13.patch` — CURRENT WINNER: park
  fully alive (BO/mapping/content intact) + busy-flush teardown of aged
  entries (FIFO 256 entries / 1GB byte cap), scratch bypass, fragment
  allocator default OFF. Harness 0/52, ORT 20/20 × 315 OK, 0 faults. Lib:
  `v13-park-alive-busy-flush/libhsa-runtime64.so.1.15.0.v13`
  (sha256 f5a934ff4a09dd982ac0aaf0bf96718b401798541c943b75d375fdaa4a5f87b8).
  Binaries staged as `libhsa-runtime64.so.1.15.0.r16v13` in
  gfx803:/data/rocr644-ring16/libs/.
- v1/v2 and v5-v8 sources were incremental edits and were not individually
  preserved; their exact semantics are described in the table above and
  reconstructible from v4/v13. Binaries for all are on the gfx803 box:
  - `.ba` = stock baseline (94e627b5…)
  - `.r16` = v2 length-bounded FIFO, no scratch bypass (9de93dd4…)
  - `.r16v2` = v3 scratch fix, free-BO-immediately (pre-debug-strip)
  - `.r16v3dbg` = v3 + FMM_DEBUG trace prints
  - `.r16v4` = v4 hardened (c1ba87bf…)
  - `.r16v5` = v5 BO-alive-until-flush (cfb37e79…)
  - `.r16final` = v3/v4-lineage cleaned build (81c521cf…)
  - v6/v7dbg/v7/v8/v9 binaries staged in gfx803:/tmp/libhsa-*.so
  - `.r16v10`/`.r16v11` = dead-end probes (see table)

## Infra notes

- Remote: `ssh gfx803` (192.168.1.214), container `ortrun`, harness bundle +
  run scripts in `/data/rocr644-ring16/` (`run_harness.sh`,
  `run_ort_faultprobe.sh`, `run_ort_serial.sh`).
- Local quick-iteration build: `/tmp/opencode/Dockerfile.rocr644-ring16` →
  `docker build --no-cache` (plain builds have served a STALE patch from
  cache despite content change — always `--no-cache` or verify the lib hash).
- CI/shipped wiring (v13 live): `gfx803/Dockerfile` `rocr-builder` stage
  applies `patches/rocr/va-reuse-defer.patch` (regenerated from v13). To
  change the mechanism, update the wip patch, rebuild + re-validate both
  gates through `gfx803/tools/reduce-harness/`, then regenerate the
  shippable patch from the validated build input (the diff body must stay
  byte-identical to what built the tested lib).
