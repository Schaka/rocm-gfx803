# tools/tc-staleness

These are hardware probes for the gfx803 cross-dispatch coherence bug. On CIK
(gfx7 and gfx8) the compute shader's TC (L1) is not maintained across a command
boundary, and the CP ignores the AQL `SCACQUIRE` and `SCRELEASE` scope bits, which
are ROCm's only expression of cross-dispatch coherence. Two failure directions
follow, and both are silent.
`patches/rocm-systems/gfx803-tc-invalidate-acquire-mem.patch` closes both with the
same PM4 `ACQUIRE_MEM` (`TC_ACTION_ENA|TC_WB_ACTION_ENA`) ring slot, ahead of each
kernel dispatch and inside `releaseGpuMemoryFence()`. That patch header holds the
mechanism, the evidence, the cost, and the two exposures that remain open.

How to read the results. In the reader direction the new data is already in VRAM and
only the kernel's view is stale, so it needs a recycled virtual address (torch's
caching allocator recycles a handful per iteration. A raw `hipMalloc` program that
never re-reads a cached VA does not reproduce it) and it survives `synchronize()`,
`AMD_SERIALIZE_KERNEL=3`, SYSTEM-scope fences and GPUVM TLB invalidation. In the
*copy-engine* direction the GPU's own answer is correct and only the host copy of
it is wrong, so probes there have to compare an on-device error against a
post-copy error -- that is exactly what `conbrace3.py` does.

## Running

You need a gfx803 card and a HIP stack, and nothing else.

```sh
python3 probe.py --trials 40                      # default shape 2x128x64x128
CLR_GFX8_TC_INVALIDATE=0 python3 probe.py         # bug must be visible
CLR_GFX8_TC_INVALIDATE=1 python3 probe.py         # bug must be gone
python3 probe.py --shape 4 256 128                # other bmm shapes also hit it
```

Exit code 0 = clean, 1 = stale reads present, so it drops into a script:

```sh
CLR_GFX8_TC_INVALIDATE=0 python3 probe.py && echo "patch is not in effect"
python3 probe.py || echo "regression: stale reads back"
```

## Harnesses here

| file | what it covers |
|---|---|
| `probe.py` | bmm, checked three ways (direct / frozen GPU->GPU copy / re-read) + repeat determinism |
| `soak.py` | 527 checks across bmm, conv+silu+groupnorm, linear+layernorm+softmax, `scaled_dot_product_attention`, reductions (sum/amax/std/cumsum/sort/median/logsumexp) and gather/scatter_add, in fp16 and fp32 |
| `train.py` | real training loop (4-conv + GroupNorm + SiLU + linear + LayerNorm, 60 Adam steps): loss vs an identical CPU run, bit-stability of the same seed, frozen-copy check on the gradients |
| `conbrace3.py` | suite-shaped conv loop that classifies each failure by site: device-side inputs, error recomputed on the GPU, error after the host copy. `copy_only: N` with `bad_gpu: 0` means the conv was right and the copy was wrong |
| `multistream2.py` | cross-stream producer to host copy. `SYNC=1` adds a full device sync, and that does not help, because the wait polls signals and never flushes |
| `xprobe2.py` | the cross-stream reader direction, with no copy engine involved |
| `pinnedhost.py` | a kernel writes pinned host memory and the CPU reads it after a stream sync |
| `wavesize_proof.hip` | rocSOLVER's wave-partial reduction pattern at `WarpSize` 32 and 64, see `patches/rocsolver/`. Build it with `hipcc --offload-arch=gfx803` |

Run every one of them twice, once with `CLR_GFX8_TC_INVALIDATE=0` as a positive
control. A clean result without that control proves nothing, because the reader
direction is intermittent, in about 15 to 50 percent of trials. The end-to-end gate is
`tools/correctness-suite/torch_op_suite.py`.

## Expected results (gfx803, one binary, knob toggled)

```
probe.py    (off) -> ~15/40 poisoned (stale first host read, stale bytes frozen
                     into a GPU->GPU copy, back-to-back reads of one tensor differ)
probe.py    (on)  -> 0/40
soak.py     (off) -> ~91/527 checks fail (bmm, conv-silu-groupnorm,
                     linear-layernorm-softmax, scaled_dot_product_attention)
soak.py     (on)  -> 0/527
train.py    (off) -> loss wrong vs CPU on most steps, gradient frozen-copy
                     mismatches, same seed not reproducible
train.py    (on)  -> loss bit-identical to the CPU trajectory, 0/60 mismatches,
                     reproducible
conbrace3   (off) -> ~90/240 anomalies, all classified copy_only
conbrace3   (on)  -> 0/240
op suite          -> 202/202 PASS, 0 BAD, 0 NONFINITE (13 failures with the knob
                     off, 5 of them NONFINITE)
multistream2      -> 5-11/240 with everything on: the cross-queue handoff is still
                     open. Read the patch header's KNOWN LIMITATION before you change
                     where the flush is emitted
pinnedhost        -> 0/60
```
