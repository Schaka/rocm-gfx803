# llama.cpp-on-gfx803 findings

Hardware-measured findings for making llama.cpp's HIP backend fast on AMD
Polaris (gfx803). The decision here was to pursue GCN-tuned kernels via
vLLM rather than inside llama.cpp; this folder preserves the work so anyone
picking llama.cpp back up has the measured facts plus one working proof of
concept.

| File | Purpose |
| --- | --- |
| `POSSIBLE_LLAMA_CPP_IMPROVEMENTS.md` | All findings, benchmark methodology, eliminated hypotheses, root causes, ISA gotchas. Read this first. |
| `gfx803-packed-dp4a.patch` | Verified +47.6% prefill fix: packed int8 dot for `ggml_cuda_dp4a()` on GCN3 (no `v_dot4`/`v_add3_u32`). |
| `apply-packed-dp4a.sh` | Applies the patch to a llama.cpp checkout and verifies it landed. |

The +47.6% prefill win (39.55 -> 58.38 t/s) is real and reproduced; Vulkan
on the same card does 163, so ~2.8x remains on the table in the mmq kernel
itself.