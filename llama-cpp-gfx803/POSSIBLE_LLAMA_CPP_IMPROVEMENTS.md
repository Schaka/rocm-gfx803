# Possible llama.cpp improvements for gfx803 (Polaris)

Findings from benchmarking llama.cpp's HIP backend on an RX 470 (gfx803,
Polaris) against its own Vulkan backend on the same GPU, same model, same
build commit. Everything here is hardware-measured on a real Polaris card —
none of it is guessed from reading code. If you want to make llama.cpp's
ROCm/HIP path faster on pre-RDNA GCN, start here.

The single most useful number to keep in mind: **Vulkan achieves 163 t/s
prefill and 24.9 t/s decode on this GPU; the HIP path achieves 58 and 17.8.
The gap is real and reproducible, and it lives in the kernels, not in the
launch path or the runtime.**

## Benchmark methodology

- Hardware: 1x AMD Radeon RX 470 (gfx803), 8 GiB VRAM, PCIe 3.0 x16.
- Software: llama.cpp pinned to `34af94cd9ab277632e27caeec2d41de2fd091b31`,
  built with `-DGGML_HIP=1 -DAMDGPU_TARGETS=gfx803` on a ROCm 7.14 gfx803
  patched base. Vulkan build from the same commit for comparison.
- Tool: `llama-bench` only. Server-side timing (llama-server) is noise;
  bench numbers are the ground truth. Never run parallel benchmarks on a
  8 GiB card — results become garbage.
- Model: Qwen3.5-9B-Q5_K_M (6.12 GiB), fully offloaded (`-ngl 99`), 12
  threads. Full run twice, take the stable value.
- Same GGUF, same commit, same machine. Only the backend differs.

## Measured baselines

Model: Qwen3.5-9B-Q5_K_M, fully offloaded, `-t 12`.

| Config | pp128 (t/s) | tg64 (t/s) |
| --- | ---: | ---: |
| HIP, stock llama.cpp | 39.55 | 18.21 |
| HIP + packed-dp4a patch | **58.38** (+47.6%) | 17.79 |
| Vulkan (stock) | 163.45 | 24.94 |
| HIP + packed-dp4a + HIP graphs | 58.31 | 17.73 |

pp512 on the patched HIP build is 58.72 — flat with pp128, which is another
clue that launch overhead is not the limiter (bigger batches should amortize
launch cost, and they don't).

## What the bottleneck is NOT (all eliminated experimentally)

These were each tested and proven NOT to be the cause of the prefill gap:

1. **Launch overhead.** HIP CUDA graphs were made to work on gfx803 (see
   below) and gave *zero* speedup. Graphs eliminate per-node host-side
   dispatch; if launches were the bottleneck, graphs would have helped.
2. **The sgemm shim** (`libgfx803_sgemm_shim.so`, an LD_PRELOAD shim in
   this repo that routes f32 GEMMs to a hand-written kernel). Setting
   `GFX803_SGEMM_SHIM_DISABLE=1` produced identical numbers. llama.cpp's
   quantized path uses its own mmq/mmvq kernels, not rocBLAS.
3. **DRAM bandwidth.** Prefill moves ~6.12 GiB per batch at ~2.8-5.3 GiB/s
   effective — about 2% of the card's ~238 GiB/s. The GPU is mostly idle.
4. **The va-reuse-defer ROCr patch** (a queue/VA-alloc change in this
   repo). An A/B test with a from-scratch ROCr build without the patch hung
   identically; the patch is exonerated.

## What the bottleneck IS

The mmq kernel's inner loop, on-chip. A rocprofv3 kernel trace of a pp64
run shows the mmq kernels dominate wall time by orders of magnitude:

- `mul_mat_q` for Q5_K (type 13): 262 calls x 5.16 ms
- `mul_mat_q` for Q6_K (type 14): 138 calls x 4.59 ms
- `mul_mat_q` for Q8_0 (type 8): 96 calls x 1.51 ms

Dispatcher-reported dispatch grid was 4096x4 workgroups of 256 threads on
36 CUs. The kernel is instruction-throughput-bound: the inner loop is doing
too much work per useful multiply, and gfx803's ISA makes that worse than on
any newer architecture (no `v_dot4`, no `v_add3_u32`).

## The packed-dp4a patch (implemented, verified, +47.6%)

gfx803 has no integer dot-product instruction. `ggml_cuda_dp4a()` in
`ggml/src/ggml-cuda/common.cuh` upstream falls back to a scalar 4x loop.
That fallback is catastrophically slow.

The fix in `gfx803-packed-dp4a.patch` adds a `GCN4` branch to
`ggml_cuda_dp4a()` that builds the packed 4-byte dot from GCN3's
`v_mul_i32_i24` (which supports byte-select `src0_sel:BYTE_n`) plus
`v_add_u32_e32`:

- 4x `v_mul_i32_i24` with `src0_sel:BYTE_0..3` = 4 instructions
- 3x `v_add_u32_e32` to accumulate = 3 instructions
- Total: 7 instructions per 4-wide dot (vs 4+ scalar for the fallback)

Critical GCN3 gotchas discovered the hard way (all verified on real
hardware / llvm-mc against `-mcpu=gfx803`):

1. **`v_add3_u32` does not exist on gfx803.** It appears in newer GCN/RDNA
   ISA. Do not use it.
2. **`v_add_co_u32` (the carry-out form) is also unusable here.** Same
   reason — not in the gfx803 ISA.
3. **Chained adds in a single `asm()` block corrupt results.** The adds use
   `vcc` as carry-in, and the compiler doesn't zero it between chained
   statements. The fix is one `asm()` statement per add; the compiler
   manages `vcc` carry-in (0 at entry) for each separately. This was
   verified: a single-block chain produces wrong sums (h=10 vs the correct
   h=-316 on the test vector), separate statements are correct.
4. **`v_mad_i32_i24` on gfx803 does NOT support byte-select.** Confirmed via
   llvm-mc (`src0_sel:BYTE_0` → "not a valid operand"). This kills the
   obvious "1 mul + 3 chained mads = 4 instructions" optimization: you can't
   feed byte-selected operands into the mad, so you'd have to pre-extract
   the bytes anyway, which costs the same as the 4 muls. The 7-instruction
   form is the floor for a packed dot on this ISA.
5. **`v_mul_u32_u24` with `src0_sel`/`src1_sel` byte selects IS supported**
   on gfx803 (the SDWA form), confirming the mul path is the right one.

The patch is `gfx803-packed-dp4a.patch` in this directory. It was verified
for correctness (coherent model output) and measured at +47.6% prefill.
`apply-packed-dp4a.sh` applies it to a llama.cpp checkout and verifies the
apply.

### Where the +47.6% lands, and what's left

The patch gets prefill from 39.55 to 58.38, still 2.8x behind Vulkan (163).
Decode is unaffected (17.79 vs 17.73) because decode uses the mmvq
mat-vec path, not mmq.

The remaining prefill gap is the mmq kernel itself. The compiler is doing
the right thing now — it's that the kernel structure (tiling, memory access
pattern, the per-thread Q5_K unpacking with `get_int_b4` nibble extraction)
was written for modern CUDA/AMD architectures and never tuned for GCN3.
gfx803 inherits the RDNA2 mmq config table (`nthreads=256, occupancy=2,
I=128, J=8`); there is no GCN-specific mmq config. The obvious experiment —
a GCN4 entry in `ggml_cuda_mmq_get_config` with different tiling/occupancy —
is the natural next step if anyone picks this up.

## HIP CUDA graphs on gfx803 (the launch path)

`hipGraphLaunch` hangs 100% reliably on gfx803 with llama.cpp's graphs. The
root cause is in the ROCm HIP runtime, not llama.cpp:

- The **segmented scheduling** path (`use_segment_scheduling_`, default ON)
  pre-builds AQL packets at graph-instantiate time and dispatches them as a
  burst at launch. It uses a per-graph signal pool (`GraphSignalManager` +
  `CreateHwEvents`/`ResetHwEvents`) whose completion callback
  (`OnLaunchComplete`) never fires on gfx803. The host then spins in
  `sched_yield` inside `hipGraphLaunch` forever.
- Disabling segment scheduling via the env var
  `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` (exact name, no `HIP_` prefix;
  defined in `projects/clr/rocclr/utils/flags.hpp:248`) makes graphs work.
  But it falls back to the **classic path** (`RunNodes()`), which calls
  `CreateCommand()` + `EnqueueCommands()` per node per launch — i.e. the
  same host-side work as no graph at all. Hence: graphs work, but give zero
  speedup. The env var *masks* the bug rather than fixing it.

This is a genuine ROCm runtime bug (upstreamable to ROCm/CLR). The graph
feature's entire performance advantage lives in the segmented path, and
that path is what's broken on gfx803. If graphs ever get fixed at the
source, re-test — the launch-overhead question was never definitively
closed because the fast path never ran.

## Summary of the wins that exist for gfx803

- **+47.6% prefill** (39.55 -> 58.38): the packed-dp4a patch. Verified,
  implemented, in this folder.
- **+0% from HIP graphs**: launch overhead proven not to be the bottleneck;
  the graph fast path is also broken on this arch (separate ROCm bug).
- **~2.8x prefill left on the table** (58 vs 163): lives in the mmq kernel
  internals. Needs a GCN3-tuned mmq config, or better, a hand-written GCN
  matmul like the Vulkan backend already has. This repo chose to pursue
  that via vLLM kernels on the gfx906 base rather than inside llama.cpp.
- **~1.4x decode left** (17.8 vs 24.9): the mmvq mat-vec path. Untested
  territory here — the config table has a GCN entry but its parameters were
  never measured on Polaris.

## Reproducing

```sh
# Benchmark (requires a real gfx803 card):
# HIP build
cmake -S . -B build-hip -DGGML_HIP=1 -DAMDGPU_TARGETS=gfx803
cmake --build build-hip --target llama-bench
./build-hip/bin/llama-bench -m model.gguf -ngl 99 -t 12 -r 2 -p 128 -n 64

# Apply the packed-dp4a patch, rebuild, re-run, compare.
./apply-packed-dp4a.sh /path/to/llama.cpp

# Vulkan build for the comparison ceiling
cmake -S . -B build-vk -DGGML_VULKAN=1
cmake --build build-vk --target llama-bench
```

## Transfer to vLLM

The useful knowledge here transfers to any GCN3-targeting kernel work:

- Packed-int8 dot must be built from `v_mul_i32_i24` byte-select + separate
  adds. Never assume `v_dot4`, `v_add3_u32`, or byte-select-on-mad.
- The 7-instruction floor for a packed dot is real; plan instruction
  budgets around it.
- GCN3's strength is fp16 and its L2; the mmq kernel's Q5_K unpacking
  (nibble `get_int_b4` extraction) is a compute tax that a well-designed
  kernel should avoid via a better weight layout or on-the-fly repacking.