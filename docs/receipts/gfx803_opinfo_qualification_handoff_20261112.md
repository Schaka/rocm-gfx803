# RX 580 / gfx803 qualification handoff — 2026-11-12

This is an **evidence report** attached to PR #8 for broader gfx803
compatibility triage. It does **not** broaden the TopK source patch,
claim that all HIP/ROCm functions work, or waive the large-input TopK
benchmark requested during review.

## Source-bound PyTorch OpInfo discovery

- PyTorch source: `d70fd61bab718a3d49b053efa8058efc0566b2b0`.
- Schaka base source: `8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78`.
- Original raw matrix path on RX 580 host:
  `/mnt/truenas/comet-net/experiments/gfx803_opinfo_full_20261111/matrix.json`.
- Matrix SHA-256:
  `47c7ed58d7b549abfc038b534faaaa9ecaf6ee2fc18b3ba55addebeed02b7e36`.
- All 1,774 generated `test_compare_cpu` and
  `test_variant_consistency_eager` CUDA/HIP node IDs attempted:
  **1,583 pass, 183 skipped/unsupported label, 3 ordinary fails,
  4 GPU fault candidates, 1 external interruption, zero unassayed**.

The original discovery runner mapped pytest skips to `unsupported`
without storing skip reasons. **These 183 cases are not evidence of
183 missing GPU operations.** The source-bound static-marker audit
collected 183/183 generated test items with zero attached skip/xfail
markers. On first 16 isolated original-body replays, 14 were genuine
upstream skips (undefined outputs or nondeterminism) and 2 were
expected failures with CPU/GPU numerical mismatches. These 16 had
zero new GPU faults/timeouts. Remaining 167 await replay
classification; no further numerical capabilities have been promoted.

## Fault and failure candidates

Four complex64 `test_variant_consistency_eager` fault candidates:
`acos`, `acosh`, `asin`, `asinh`. They were associated with the
same amdgpu VM fault: address 0, read from TC4,
`VM_CONTEXT1_PROTECTION_FAULT_STATUS=0x1004800C`.
Pinned PyTorch source has analogous complex dispatch and math
lowering for these operations. This correlation is **not proof**
of a common causal kernel/compiler defect. Isolated replay and
per-kernel attribution remain open.

Three ordinary failures: `sparse_sampled_addmm` (float32),
`topk` CPU comparison (float32), and `topk` eager variant
consistency (float32). The two original TopK OpInfo failures
must be analyzed against their exact sample/shape/tie/variant
contracts, notwithstanding the separately passing 120-case
TopK semantic assay.

## Original receipts and follow-up procedures

- Discovery:
  https://github.com/chboishabba/gfx803_compat_graph/blob/master/docs/receipts/gfx803_opinfo_discovery_20261112.json
- Complex unary fault replay:
  https://github.com/chboishabba/gfx803_compat_graph/blob/master/docs/GFX803_COMPLEX_UNARY_FAULT_REPLAY_20261112.md
- Skip classification:
  https://github.com/chboishabba/gfx803_compat_graph/blob/master/docs/GFX803_OPINFO_183_UNSUPPORTED_TRIAGE_20261112.md

Upstream TopK review remains independently gated on a **rebuilt
wheel from the reviewed helper/driver revision** and paired
unpatched/patched benchmarks, including 151,936-token and
1-million/16-million-element cases. No review-era binary
benchmarks are claimed here.

This report is deliberately compact and attributable; raw test
journals and kernel logs stay on the host, bound by original receipts.


## 14 November: independent backend and stability assay progress

These are **separate scoped RX 580/gfx803 hardware qualifications** on the
staged Schaka ROCm 10.1 closure; not evidence that the TopK patch itself
fixes MIOpen, HIP, KFD, or all PyTorch operations:

| Original assay | Hardware outcome | Unclosed scope |
| --- | --- | --- |
| Original Schaka FP16 GEMM/batched-GEMM/convolution | **27/27 source-verified numerical passes** | Broad BLAS, backend algorithms/dtypes |
| Original Schaka Triton dot `bf16/fp16/fp32/i8` | **4/4 executed**; max errors respectively `4.77e-7`, `9.02e-7`, `1.57e-6`, `0` | No upstream numerical tolerance defined; execution-only |
| Original Schaka MIOpen direct C++ suite | **23 entries**: 18 pass; 2 zero-applicable solver lists; pool GPU page-not-present; activation OOM; groupnorm intentionally rejected boundary shapes conservatively marked failed | Pool root cause, activation memory, groupnorm negative controls |
| Source-preserving pooling isolation | **42/42** individual cases pass; **10/10** in-process prefix lengths pass, including all 42 in one process | Original full-sweep page fault not reproduced; original exact binary/loader and intermittent/cross-workload state unresolved |
| ELF ABI inventory | All 10 expected shared-library families found (including MIOpen/rocSOLVER) | Symbol availability is not GPU function qualification |

Source-owned local original receipts on the investigator's RX 580 host
(prefix `/mnt/truenas/comet-net/experiments/`):
`gfx803_schaka_fp16_20261113/matrix.json`,
`gfx803_schaka_triton_20261113/matrix.json`,
`gfx803_schaka_miopen_20261113/matrix.json`,
`gfx803_pool_isolation_20261113_v3/matrix.json`,
`gfx803_pool_prefix_20261113/matrix.json`,
and `gfx803_rocm_abi_20261113_v2/matrix.json`.
Retain per-run hashes/environment in the original artifacts; do not
present unuploaded matrix paths as review-attached binaries.

**Existing stability/RCA baseline:** 2026-03-27/30 WhisperX
transcribe/align GPU hang and reset incident notes retain D2H copy →
host wait → H2D/copy activity → GPU hang, `ring gfx timeout`, BACO
reset and VRAM loss. Older gfx7/8 KFD SDMA doorbell callback and
queue-validation fixes are distinct from the November MIOpen GPU
VM fault. The running-kernel patch provenance and causal overlap remain open.

**Next highest-value work:** direct original HIP runtime, ROCr/HSA,
KFD/SDMA queue/copy/signal and memory-pressure tests, plus exact
original-binary pooling A/B and loader/journal correlation. The factory
plan is to reuse original component test suites, retain source/runtime
identities and separate function conformance from sustained host
reliability. Current roadmap:
https://github.com/chboishabba/gfx803_compat_graph/blob/master/docs/GFX803_PHASE_DELIVERABLES_RECONCILIATION_20261114.md

**PR scope stays unchanged:** source-level gfx803 TopK selection.
The reviewed-wheel rebuild, original OpInfo TopK failures,
and paired large-input before/after performance benchmarks remain
separate open review gates. No universal ROCm release or stability claim.
