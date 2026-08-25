NOT an auto-applied patch (vLLM isn't built from source in this repo's
Dockerfile -- see `gfx803-skinny-gemv.patch.md` for the full explanation
of why; same situation here).

WHY -- `prefix_prefill.py`'s `context_attention_fwd()` (the prefill/
chunked-prefill attention Triton kernel) already has gfx803-aware tile
sizing from an earlier fix (LDS-capacity-driven `BLOCK_M`/`BLOCK_N` drop
to 32x32, see the existing comment in that file), but its
`extra_kargs` block for ROCm was empty -- no `waves_per_eu` tuning at
all, unlike the decode-path attention kernel which already had this
(`chunked_prefill_paged_decode.py`, see `gfx803-skinny-gemv.patch.md`).
Same class of untried lever, found while reconciling prefill's real
comparison point against llama.cpp.

**Correction to a stale note**: SESSION_HANDOFF.md previously carried a
"163 t/s prefill (vulkan ceiling)" acceptance target from an *unrelated*
llama.cpp-ROCm-vs-Vulkan investigation earlier in this repo's history --
not vLLM's actual comparison point. The real number (from this repo's own
earlier direct `vllm bench latency` vs llama.cpp-Vulkan comparison,
SESSION_HANDOFF.md's original table) is **607.52 t/s** llama.cpp-Vulkan
prefill at 128 tokens. A clean isolated prefill-only benchmark (prefix
caching disabled -- the naive version of this benchmark that reuses the
same prompt every iteration silently measures near-nothing after the
first call hits the cache) puts vLLM at:

| prompt length | vLLM (before this fix) | vLLM (after) | llama.cpp-Vulkan | vLLM as % of ceiling |
|---|---|---|---|---|
| 128 tokens | 578.58 tok/s | 583.77-584.78 tok/s | 607.52 tok/s | ~96.1-96.3% (was ~95.2%) |
| 512 tokens | 608.83 tok/s | not re-measured | ~607.52 tok/s (128-tok number, used as approximation) | ~100.2%+ already |

Prefill was never actually "the real gap" (matches this repo's own
earlier "Prefill is already close -- not the problem" framing) -- it's
already at or above the llama.cpp-Vulkan ceiling at longer, more
realistic prompt lengths, and within ~4% at the shorter 128-token length
this repo's other benchmarks standardize on.

WHAT -- `prefix_prefill.py`, in `context_attention_fwd()`:

```python
# before:
    extra_kargs: dict[str, Any] = {}
    if current_platform.is_rocm():
        extra_kargs = {}

# after:
    extra_kargs: dict[str, Any] = {}
    if current_platform.is_rocm():
        extra_kargs = {"waves_per_eu": 1}
```

`num_warps` (2/8 tried, 4 already optimal) and `num_stages` (2 tried,
worse than 1) were also swept and left unchanged -- same diminishing-
returns pattern as the decode-attention launch-param sweep in
`gfx803-skinny-gemv.patch.md`.

STATUS: +1% at 128 tokens (578.58 -> 583.77-584.78 tok/s), correctness
verified via `sanity_gen.py` (unchanged output). Not yet promoted to an
auto-applied Dockerfile patch -- gated on the "vendored vLLM fork"
decision, same as the other `patches/vllm/*.patch.md` files.
