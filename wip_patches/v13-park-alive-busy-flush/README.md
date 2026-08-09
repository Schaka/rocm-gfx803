# v13 — park-alive + busy-flush teardown (CURRENT WINNER)

Patch: `va-reuse-defer-v13.patch` (applies to ROCR-Runtime @ rocm-6.4.4).
Lib: `libhsa-runtime64.so.1.15.0.v13` (sha256
`f5a934ff4a09dd982ac0aaf0bf96718b401798541c943b75d375fdaa4a5f87b8`).

## Mechanism

At `hsakmt_fmm_release` the object is parked on a per-aperture FIFO, fully
alive: BOs NOT freed, GPU mapping NOT touched, VA stays reserved, content
intact. When the entry ages out (FIFO > 256 entries or > 1GB parked bytes),
`fmm_do_release()` really frees the BOs inside a LATER FREE call -- i.e.,
while the GPU pipeline is typically active -- then returns the VA to the
aperture free-list. The VA cannot be reallocated until it has sat ≥256
frees, so the stale physical/VA pair churns before reuse.

## Why this shape

- **Both ORT gates at once**: harness 0/52; ORT `*ReductionOpTest*`
  315 OK / 0 FAILED in 20/20 runs (40-run validation running).
- **0 faults** (vs ~3/40 for the v4 family): the parked mapping stays alive,
  so an in-flight kernel that reads a buffer freed behind its back hits a
  valid page with the ORIGINAL content (proven required by v11: a fresh
  dummy page shields the fault but returns wrong data).
- **Busy teardown is the stale-clearing ingredient** (v9/v10 proved idle
  teardown at a GPU-quiescent point does NOT clear it): the aged BO free at
  a later, typically mid-stream FREE lands while engines have work, so the
  Polaris PTE/TLB invalidation actually executes.

## Config

- `FMM_REUSE_QUEUE_MAX 256` entries
- `FMM_REUSE_BYTES_MAX` 1GB byte cap (bounds the extra VRAM held; 4096/2GB
  caps from v6 broke real workloads with HIPBLAS_STATUS_ALLOC_FAILED)
- scratch apertures bypass the FIFO entirely (`fmm_is_scratch_aperture`),
  avoiding the v1/v2 exit-spin in `fmm_release_scratch`
- fragment sub-allocator default OFF in flag.h (second VRAM VA recycler the
  fmm fix cannot see)

## Residual risks

1. VRAM overhead: the window holds up to 1GB of "freed" VRAM alive (bounded
   by byte cap).
2. Busy assumption: if a workload frees buffers while the GPU is long idle,
   the aged teardown lands idle and the flush may not execute -- the same
   edge the quiesce variants hit, at a much lower probability since
   harness/ORT free mid-stream.
3. Fault at flush if a reader lags > 256 frees (cap-triggered flush of a
   still-read object); not observed in 20 runs.

## Performance (stock vs v13)

Interleaved ORT `*ReductionOpTest*` runs, kernel 7.1.7, 6 each, wall-clock:

| variant | mean | stdev | min/max |
|---|---|---|---|
| stock `.ba` | 10.888s | 0.100s | 10.72/10.99 |
| v13 `.r16v13` | 10.843s | 0.137s | 10.58/10.97 |

Δ = -0.41%, within noise → **no measurable perf regression**. The fix adds
O(1) FIFO push/pop on the free path and defers the BO-free ioctl; total
ioctl count over the workload is unchanged (frees happen, just later). The
only real cost is transient VRAM: up to 1GB of "freed" BOs held alive until
aged out, bounded by `FMM_REUSE_BYTES_MAX`. (The stock runs also exited 1
every time -- the wrong-sum bug reproduces on 7.1.7; v13 exited 0 always.)