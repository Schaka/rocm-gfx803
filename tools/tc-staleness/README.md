# tools/tc-staleness

These are hardware probes for the gfx803 cross-dispatch coherence bug. On CIK
(gfx7 and gfx8) the compute shader's TC (L1) is not maintained across a command
boundary, and the CP ignores the AQL `SCACQUIRE` and `SCRELEASE` scope bits, which
are ROCm's only expression of cross-dispatch coherence. Three failure directions
follow, and all three are silent.
`patches/rocm-systems/gfx803-tc-invalidate-acquire-mem.patch` closes them with the
same PM4 `ACQUIRE_MEM` (`TC_ACTION_ENA|TC_WB_ACTION_ENA`) ring slot: ahead of each
kernel dispatch, inside `releaseGpuMemoryFence()` before copy-engine work, and ahead
of an event-record barrier for the benefit of the queue on the other side of that
event. That patch header holds the mechanism, the evidence, the cost, and the one
exposure that remains open (HIP graph replay).

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
| `multistream2.py` | cross-stream producer to host copy. `SYNC=1` puts a full device sync between the producer and the copy; that measures clean, but it is a serialisation, not a fix, and nothing in a real pipeline can afford it |
| `ms3.py` | `multistream2`'s shape with each anomaly classified by site, so a failure names its own cause instead of just counting |
| `ms4.py` | the same handoff with a pinned host destination, and with two producer streams fanning into one consumer |
| `graphprobe.py` | a captured `CUDAGraph` that reads what a copy engine, or a kernel on another queue, wrote. This is the probe for the one dispatch path the patch leaves uncovered |
| `xprobe2.py` | the cross-stream reader direction, with no copy engine involved |
| `pinnedhost.py` | a kernel writes pinned host memory and the CPU reads it after a stream sync |
| `wavesize_proof.hip` | rocSOLVER's wave-partial reduction pattern at `WarpSize` 32 and 64, see `patches/rocsolver/`. Build it with `hipcc --offload-arch=gfx803` |

Run every one of them twice, once with `CLR_GFX8_TC_INVALIDATE=0` as a positive
control, and the cross-stream ones (`multistream2`, `ms3`, `ms4`) also with
`CLR_GFX8_TC_RECORD_FENCE=0`, which turns off only the event-record site. A clean result without that control proves nothing, because the reader
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
multistream2      -> 0/1600 with the event-record site on, 5-11/240 with it off
ms4.py            -> 0/2400 with the site on
pinnedhost        -> 0/60
ms3.py            -> the cross-stream case with each anomaly classified. The kinds
                     are bad_gpu (the producer computed wrong), flush_too_early (a
                     kernel clone read the right bytes, so only the copy engine was
                     early), bad_copy and copy_race. Expected on gfx803: 0 anomalies
                     in 2000 checks with CLR_GFX8_TC_RECORD_FENCE=1, 21 in 960 with
                     it off. If you see flush_too_early, the writeback is not behind
                     the producer's retirement, which is REASON 3 of the patch
                     header; if you see bad_gpu, you are looking at a different bug
ms4.py            -> ms3's shape, but the consumer copies into pinned host memory
                     and a second arm fans two producer streams into one consumer
                     (one pinned buffer and one stream pair are reused, because
                     allocating a pinned tensor per iteration exhausts the pinned
                     pool and wedges an SDMA ring). Expected: 0 anomalies in 2400
                     checks with the site on
graphprobe.py     -> a captured torch.cuda.CUDAGraph whose only op reads a buffer
                     that something else wrote, replayed after (a) a copy-engine
                     write and (b) a kernel on another stream. This is the one
                     dispatch path with no ACQUIRE_MEM of its own
                     (dispatchAqlPacketBatchFlat), so it is the probe for the
                     patch header's remaining caveat
```
