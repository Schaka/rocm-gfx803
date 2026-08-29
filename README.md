# rocm-gfx803

AMD Polaris (gfx803: RX 460/470/480/560/570/580/590 and friends) support for
the MIGraphX + ONNX Runtime + PyTorch stack, split out from
[`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) into its own
repository.

## Prebuilt images -- don't build this yourself

CI already builds and pushes the final image to GHCR. You do not need to
build the Dockerfile locally unless you're patching something. Pull it:

```bash
# rocm7 (main line, ROCm 7.14) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx803

# rocm7 -- always the newest successful rocm7 build
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803

# rocm6.4.4 (older, hardware-verified line) -- versioned tag
docker pull ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm6.4.4-gfx803
```

See "Repository layout" and `.github/workflows/gfx803-component.yml` for
the full tag scheme (per-component images, cache tags, dated tags).

## Why a separate repo

gfx803 is unsupported upstream as of ROCm 6.0 -- AMD stopped building for it,
and ROCm 7 rejects the card outright at HSA agent creation. Everything that
makes gfx803 work at all is a local patch: a legacy-doorbell restore just to
get dispatch working on ROCm 7, plus a growing set of Tensile/MIOpen/MIGraphX
correctness patches for bugs that only manifest on this old GCN3 hardware.
That patch set changes independently of, and faster than, the mainline
nightly/release pipeline `rocm-migraphx-ort-builder` runs for every other
architecture -- keeping it there meant every gfx803-specific investigation
was noise in a repo that otherwise doesn't need any of it. This repo is
where that investigation and patch history actually lives now.

The two repos stay linked in one direction: `rocm-migraphx-ort-builder`'s
docs point here for gfx803; this repo doesn't try to track or duplicate the
mainline build's own per-arch matrix.

## Repository layout

```
rocm-gfx803/
├── Dockerfile              # ROCm 7.14 (TheRock), the primary/actively-developed line
├── .dockerignore
├── patches/                # gfx803 patches against the 7.14 pin
├── tools/                  # correctness-suite, reduce-harness (copies, not shared)
├── verify.py                # on-hardware smoke test
├── MIGRATION_NOTES.md      # running investigation log, written as findings happen
├── rocm6.4.4/               # the older, hardware-verified, classic-tag ROCm 6.4.4 line
│   ├── Dockerfile
│   ├── patches/
│   ├── tools/
│   ├── verify.py
│   ├── KERNEL_BUGS.md      # the original gfx803 bug-hunting methodology and record
│   └── wip_patches/        # rejected/superseded patch designs, kept for the record
└── .github/workflows/      # CI, same image tags/ghcr.io paths as before the split
```

**rocm7 (this repo's root) is the actively developed line.** ROCm 7.14 is a
TheRock meta-release, not a classic per-repo tag -- every component is pinned
to a release *branch*, not a frozen commit: ROCR-Runtime, rocBLAS and MIOpen
(now the `rocm-systems`/`rocm-libraries` monorepos) track
`release/therock-7.14`, MIGraphX tracks `release/rocm-rel-7.14`, same
`release/rocm-rel-<major.minor>` convention this repo's mainline
(`rocm-migraphx-ort-builder`) uses for its own manual releases -- no nightly
scheduling, no prebuilt wheels, just "build from the named release branch's
current tip when someone runs it." See `MIGRATION_NOTES.md` for how those refs
were resolved and why.

**`rocm6.4.4/` is the older, stable line** -- classic per-repo `rocm-6.4.4`
tags, hardware-verified over a longer period, actively maintained but not
where new investigation happens. It's kept because it's still the only line
with a full correctness-suite pass and extensive real-hardware mileage; the
rocm7 line inherits its patches and methodology but does not share code with
it -- **the two lines are deliberately independent copies**, not a shared
asset. Both are under active development; a shared file would mean a bug
found while chasing 7.14 could reach the hardware-verified 6.4.4 build. Once
7.14 is confirmed at least as solid as 6.4.4 across the board, the two get
diffed and consciously merged back together -- see "Convergence" below.

## Status

- **rocm7 (this repo's root)**: builds end-to-end and is hardware-verified.
  `rocminfo` enumerates the card as a real `KERNEL_DISPATCH` agent;
  rocBLAS/MIOpen/MIGraphX/PyTorch/ORT all do real GPU work on it. The full
  `tools/correctness-suite/` (23 MIOpen op/solver sweeps) passes clean.
  ORT's `onnx_backend_test_series.py` (3828 tests) has one open,
  non-gfx803-specific regression (`ConvTranspose`, an upstream MIGraphX
  bug -- reported upstream, not patched here). Real-model validation
  (faster-whisper/CTranslate2, whisper.cpp, parakeet.cpp) passes with
  correct transcripts on real audio.
- **rocm6.4.4**: hardware-verified, the older and longer-running of the two
  lines. See `rocm6.4.4/README.md` and `rocm6.4.4/KERNEL_BUGS.md`.
- **therock-experimental**: EXPERIMENTAL, not hardware-verified, a third
  independent line. Builds gfx803 through TheRock's own build
  orchestration (`fetch_sources.py --patch-tag` + `cmake
  -DTHEROCK_AMDGPU_FAMILIES` + `ninja`) instead of this repo's hand-rolled
  Dockerfile, to test whether this repo's patches survive AMD's own build
  system -- see `therock-experimental/README.md` and
  `ROCM_UPSTREAM_ANALYSIS.md`.

### Required host setup

- **VBIOS/VRAM clock**: this card's VRAM must run within its rated speed
  ceiling (1750MHz on the mining-tuned VBIOS this box shipped with,
  confirmed via core overdrive; a stock/correct-vendor VBIOS needs no
  overdrive). Running VRAM above spec causes real GPU VM faults under
  MIOpen (`pool_sweep`) and GPU hangs under vLLM -- both the same
  hardware-level root cause, not a software bug. See "Host VBIOS setting"
  below and `RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md` for the
  investigation.
- **Never set `ROCR_GFX8_EOP_MITIGATION=1` or
  `ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US`.** Both give up on a packet the
  CP has not consumed. Confirmed causing full unrecoverable GPU bus death
  (`device lost from bus`); the HIP-timeout variant hard-locked the whole
  box twice, needing physical power-cycles. This holds regardless of the
  VRAM-clock fix above.
- `patches/rocm-systems/aql-ring-queue-full-workaround.patch` restores the
  AQL ring's GFXIP 7/8 double mapping (64 -> 131072 packets, a 2048x
  increase over the unpatched cap) and is required for
  `graph-replay-batch-chunk-deadlock.patch` to be unnecessary. Needs a
  kernel WITHOUT `REFERENCE-amdkfd-gfx7-8-queue-size-writeback`, and must
  never be combined with `graph-replay-queue-size-cap.patch`.

### vLLM on gfx803

Working end-to-end, including hybrid Mamba/GDN-attention models
(Qwen3.5-2B). Two build-time fixes are required in a vLLM checkout beyond
the VBIOS setting above: the prebuilt `libgfx803gemm.so` needs an explicit
link to `libamdhip64.so` (recompile from `gfx803_gemm_lib_final.hip` with
the ROCm 7.14 `hipcc`, not the system one, if you hit `undefined symbol:
__hipPopCallConfiguration` at dlopen); and `librocblas.so` needs
`LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib`, since it isn't on the default
search path.

**First run against any new model or new prompt/batch shape is slow --
this is expected, not a hang.** Triton/LLVM's AMDGPU backend compiles
noticeably slower on gfx803 than on modern archs for some kernel configs
(`num_warps=4` autotune configs measured 19-67s each here, vs <2s for
`num_warps=2`); a hybrid model's first engine-init can take several
minutes of 0%-GPU-busy CPU compilation. Triton's on-disk kernel cache
(`~/.triton/cache`) makes this strictly one-time per unique
kernel/shape combination -- give it time rather than killing the process.
gfx803 has no matrix cores (MFMA), so attention always falls back to
vLLM's Triton paged-attention kernel rather than the ROCm custom one --
a real, understood cost on this hardware, not a bug. **Don't pass
`enforce_eager=True`** -- CUDA-graph capture works cleanly on this
hardware (confirmed: captures without error, output stays correct,
+7% decode throughput from cutting per-op launch overhead) and earlier
benchmarking in this repo's history used `enforce_eager=True` out of
habit from initial exploratory scripts, not because graphs don't work
here.

**gfx803 GEMM kernel** (`gfx803_prefill_gemm.py`/`gfx803_gemm_lib_final.hip`):
a hand-written, register-blocked HIP kernel used for prefill's linear
layers, alongside rocBLAS's hand-written GEMV kernel for decode's M=1
shape. Tuned to 57 VGPRs/thread (partial K-tile unroll, `#pragma unroll
8`) for double the occupancy of a naively fully-unrolled version. Its
transposed-weight cache is capped by `VLLM_GFX803_GEMM_CACHE_MB` (env
var, default 768 = ~10% of an 8GB card's VRAM) -- this cache is populated
during the model's first real forward pass, before vLLM's KV-cache memory
profiling settles, so a larger value buys more prefill-path coverage at
the direct cost of available KV cache. Measured on this box (Qwen3.5-2B,
RX 470 8GB, 2101-token prompt, `gpu_memory_utilization=0.98`,
single-request batch=1, metrics-based per-request timing):

| `VLLM_GFX803_GEMM_CACHE_MB` | prefill tok/s | decode tok/s | KV cache |
|---|---|---|---|
| 768 (default) | 352.7 | 42.0 | 0.66 GiB / ~39k tok |
| 1200 | 365.1 | 42.1 | 0.4 GiB / ~24k tok |
| 1350 | 366.6 | 42.0 | 0.26 GiB / ~16k tok |
| 1536+ | OOM at `gpu_memory_utilization=0.98` on this 8GB card | | |

Decode throughput is unaffected by this cache in every configuration
measured -- it's bounded by the split-attention and GEMV kernels above,
neither of which this cache touches. This card can safely run
`gpu_memory_utilization` up to ~0.98 (no unrelated processes hold VRAM on
a dedicated box).

**Routing was originally gated to `gate_up_proj`-shaped weights only**
(out_features >= 8192), on the belief that Tensile's kernel won at
`qkv_proj`/`o_proj`/`down_proj`'s smaller shapes. That claim was never
re-verified after this session's earlier measurement-methodology fixes
(see the split-KV attention section above for the same pattern) and
turned out false: `rocprofv3` traced the Tensile kernel actually
dispatched at those shapes to Tensile's **untuned fallback library**
(`TensileLibrary_..._fallback_gfx803.hsaco` -- this repo's rocBLAS patch
tunes a different kernel type, not this one), and head-to-head
benchmarking at real shapes (qkv_proj K=2048/N=2560, o_proj K=2048/N=2048,
down_proj K=6144/N=2048, M=128-2101) found this kernel wins every single
case, 20-95% faster (down_proj nearly 2x at M=2101). The gate is now just
`N % 64 == 0` (this kernel's tile width; not verified below that, every
real shape in this model satisfies it) -- no shape-name allowlist.

That fallback dispatch isn't a missed-tuning-opportunity gap this repo's
rocBLAS patch could still close, either: the shapes hit here use fp16
**HPA** (high-precision/fp32-accumulate) contraction, and
`patches/rocblas/TENSILE_GFX803_FP16_TODO.md` (from the investigation
behind `tensile-gfx803-fp16-nond16.patch`) confirms fp16-HPA Tensile
kernels are structurally impossible on gfx803 -- traced to AMD's own 2020
Tensile commit history: `AsmCaps.py`'s gfx803 entry hardcodes
`v_pk_fma_f16`/`v_mad_mix_f32`/`v_fma_mix_f32` all `False` (Polaris
predates the packed/mixed-precision fp16 hardware those need, which
shipped starting Vega/gfx900), and the commit that moved fp16 codegen
into components carries the maintainers' own comment: `# No HPA on 803,
every other combination should work though.` The patch that *is* shipped
only unlocks non-HPA fp16 -- a different `HighPrecisionAccumulate: False`
kernel variant Tensile can generate correctly on this hardware, but not
the one these prefill shapes need. This is the real reason the hand-written
HIP kernel above wins so decisively here: it isn't competing against a
merely-undertuned Tensile kernel, it's doing something Tensile cannot
express in its codegen on this silicon at all.

**The realized gain is much smaller than that 20-95% kernel-level number**
because the cache stores each weight **twice** -- the original layout
plus a transposed copy -- so a VRAM budget that respects the ~10% ceiling
only fits a fraction of the now-much-larger set of eligible weights (most
still decline-to-cache and fall back to Tensile, unchanged from before
this widening). Confirmed real but modest: 352.6 -> 352.7 tok/s at the
unchanged 768MB default (most newly-eligible weights don't fit),
352.6 -> 365-367 tok/s (+4%) at 1200-1350MB before hitting VRAM OOM on
this card. Two follow-up ideas were tried and did **not** pan out, kept
here so they aren't retried blind:

- **Replace the original weight storage with the transposed copy** (drop
  the double storage entirely) -- doesn't work. Decode's M=1 path
  (`gfx803_skinny_linear` -> `LLGemm1_kernel`, `csrc/rocm/skinny_gemms.cu`)
  reads every weight in its native `[N, K]` row-major layout for
  coalesced `float4` loads; every weight is used by decode, not just the
  ones prefill routes through this kernel, so both layouts are genuinely
  needed at once, not redundant duplication to eliminate. Doing this
  safely would mean rewriting `LLGemm1_kernel` to also work from the
  transposed layout -- real scope on an already-tuned, already-verified
  critical path, not attempted.
- **Populate the cache at model-load time** (`process_weights_after_loading`)
  instead of lazily on first forward call -- tried, reverted, real
  regression (352.6 -> ~342 tok/s, reproduced twice). Cause: Qwen3.5-2B is
  multimodal (`vision_config`), and this hook fires for every linear layer
  including the vision tower's, which text-only generation never
  exercises. Eager population filled the entire 768MB budget with
  never-used vision weights (confirmed directly: cache dump showed 100%
  vision-shaped tensors, zero text-decoder weights) before a single real
  layer got cached. The original lazy, execution-order-driven design was
  quietly doing something right by never caching weights that are never
  actually called -- fixing the load-time version would need real
  vision-vs-text layer detection, not attempted tonight.

Our own GEMM kernel's occupancy was re-checked too (same VGPR-extraction
technique that found attention's problem): 57 VGPR, 0 spills, unchanged
from earlier tuning -- already solid, no further easy win there.

**gfx803 split-KV decode attention** (`gfx803_split_attn.py`): decode's
Triton paged-attention fallback (no MFMA hardware means no ROCm custom
attention kernel) launches only `num_seqs * num_kv_heads` threadblocks --
2 on this model's 32-CU RX 470, each looping serially over the whole KV
sequence. This kernel splits that loop across a `NUM_KV_SPLITS` grid
dimension instead. That split count was hardcoded to 2, tuned only
against a ~192-token context; at this model's actual ~2229-token decode
context it left 42.6% of decode's GPU time in this one kernel. Sweeping
`{2,4,8,16,32}` on real hardware at the real context length found 16
fastest (31.1 tok/s vs 22.6 at the old fixed 2, +37%; confirmed via
`rocprofv3`: this kernel's own GPU-time share dropped from 42.6% to
17.6%). Now scales with `max_seq_len` at the call site in
`chunked_prefill_paged_decode.py` instead of a fixed constant -- see that
function's inline comment for the exact formula and its (thin, two-point)
justification.

**The split kernel itself was then replaced with a hand-written HIP
kernel** (`gfx803_attn_split.hip`/`libgfx803attn.so`, ctypes-bridged, same
pattern as `gfx803_prefill_gemm.py`; toggle back to Triton with
`VLLM_GFX803_ATTN_HIP_KERNEL=0`). Pulling the Triton kernel's compiled
ISA straight from `~/.triton/cache` found it pinned at 256/256 VGPR with
**637 spilled registers** at this model's HEAD_SIZE_PADDED=256 -- Triton
materializes a full-width K/V tile and accumulator per iteration with no
way to chunk the head dimension in its frontend (list comprehensions,
tensor slicing, and loop-carried tensor collections were all tried and
all rejected by Triton's compiler -- a real language limitation, not a
bug to work around). The HIP kernel processes one KV token at a time with
a wavefront-cooperative `__shfl_xor` dot-product reduction instead of a
tile load, so per-thread state never exceeds a handful of floats: 61
VGPR, **0 spills**. Verified against the Triton kernel bit-for-bit at
fp32-rounding level (<3e-5 max abs diff) across seq_len 1-4096 and
NUM_KV_SPLITS 1-32 via an isolated harness before ever touching the real
model. Real-hardware result: the kernel's own GPU time dropped from
2190ms to 242ms for the same 762 calls (9x, via `rocprofv3`), decode
throughput 33.5 -> 42.0 tok/s (+25%) -- **surpassing llama.cpp's 40.42
tok/s** on this exact comparison.

**Decode's GEMV kernel (`LLGemm1_kernel`, `csrc/rocm/skinny_gemms.cu`) is
already close to this card's memory-bandwidth ceiling** -- checked
directly, not assumed: isolated per-shape benchmarking of every decode-path
GEMV shape in this model measures ~200-213 GB/s against this card's
~211 GB/s theoretical peak (rows_per_block=2, the current hardcoded value,
already ties or beats 4/8/16 on every shape checked). `lm_head`'s call
alone (vocab_size=248320, tied embeddings, ~970MB weight matrix, once per
decode step) accounts for ~640ms of decode's ~2000ms GEMV total at ~92%
of peak bandwidth -- a real hardware ceiling, not a kernel gap. Enabling
CUDA-graph capture (see below) gave a real but modest +7% decode gain from
cutting CPU-side dispatch gaps between kernel launches; profiling
confirmed the GEMV kernels' own GPU execution time is unchanged before and
after (120.5us vs 121us average) -- the win was in the gaps between
launches, not the kernels themselves.

**For comparison, `llama.cpp`/Vulkan (RADV) on the same card, same model,
no quantization** (`Qwen3.5-2B-BF16.gguf`, `llama-bench -ngl 99`, 3 reps):

| test | vLLM (CUDA graphs, HIP attention kernel) | llama.cpp/Vulkan (BF16) |
|---|---|---|
| prefill, 2101 tok | 352.6 tok/s | 760.42 tok/s |
| decode (tg128) | **42.0 tok/s** | 40.42 tok/s |

Decode now beats llama.cpp/Vulkan on this card. Getting there confirmed
the gap was never a hardware wall -- RADV reports `matrix cores: none`
on this card too, same as ROCm; llama.cpp's edge was llama.cpp's Vulkan
flash-attention kernel (`ggml-vulkan/vulkan-shaders/flash_attn.comp`)
being a hand-tuned scalar (non-coopmat) shader with explicit `AMD_GCN`
occupancy control that vLLM's generic Triton kernel didn't have; writing
a real occupancy-controlled kernel of our own (the HIP split-attention
kernel above) closed it. Decode's remaining headroom is now mostly GEMV
memory bandwidth (measured near this card's ~211 GB/s ceiling already,
not much left there). Prefill's remaining ~2.2x gap hasn't been
investigated with the same rigor yet -- likely the next place to look if
pushing further.

## Building

Both lines build the same way -- multi-stage Dockerfile, patches applied
per-project before each component compiles from source:

```sh
# rocm7 (this repo's root)
docker build -t rocm-gfx803:rocm7 .

# rocm6.4.4
docker build -t rocm-gfx803:rocm6.4.4 -f rocm6.4.4/Dockerfile rocm6.4.4/
```

Every component (ROCR-Runtime/CLR, rocBLAS, MIOpen, MIGraphX, PyTorch,
torchvision, torchaudio, ONNX Runtime) is compiled from source, every time --
there is no prebuilt-wheel shortcut for gfx803 anywhere upstream, unlike the
mainline repo's newer/more-common architectures, which can sometimes import
an already-published wheel instead of recompiling. CI reflects that: no
`ARG *_IMAGE=...`-style "pull instead of build" branch exists for gfx803.

For on-hardware checks, `verify.py` (and `rocm6.4.4/verify.py`, identical in
spirit) asserts the MIGraphX EP is present and does real GPU work -- run it
inside a container started with `--device=/dev/kfd --device=/dev/dri
--group-add video`.

## Patches: philosophy and conventions

Every patch under `patches/`/`rocm6.4.4/patches/` documents, in its own
header, the WHY (what's broken, how it was found, hardware measurements
where applicable) and the WHAT (the actual fix), plus a re-diff note when a
patch was carried from one line to the other and something in the upstream
source shifted. Read the patch header before touching the code it targets --
the reasoning usually isn't obvious from the diff alone.

**Two apply dialects, on purpose, not by accident**: `.sh` drivers under
`patches/rocm-systems/` use `git apply` (their target, `rocm-systems`, is
cloned as a real git repo root); everything else (`rocblas/`, `miopen/`,
`migraphx/`, `pytorch/`) uses `patch -p1`, because those targets are
sparse-checked-out *subdirectories* of a monorepo, and this box's git
version silently no-ops `git apply --check` there ("Skipped patch", exit 0,
nothing modified) instead of failing loudly. Every driver script is
self-verifying: it greps for a marker string after applying and fails the
build if the marker isn't there, so a patch that silently stopped applying
can't ship unpatched code.

## When a patch needs updating

A gfx803 patch stops applying (or starts applying with fuzz) whenever the
pinned upstream commit moves and the target file changed shape around it --
that's expected, not a sign something is wrong with the patch itself. Before
re-diffing:

1. **Check whether the bug is even still there.** Re-pinning to a newer
   upstream commit sometimes fixes the underlying issue outright (it's
   happened before -- see `MIGRATION_NOTES.md`'s MIGraphX section, where two
   6.4.4-era ONNX-parser patches turned out to be fully obsolete against
   7.14, one because the fix already landed upstream, one because the whole
   code path it patched was replaced). Grep the new source for the
   patch's target function/struct before assuming a straight re-diff is
   needed.
2. **Check whether the fix is still gfx803-specific.** Some of these bugs
   are architecture-general defects that just happen to be *exposed* by
   gfx803's kernel/solver selection (the WGM Tensile swizzle bug, the
   small-GEMM assembly miscompute) rather than genuine hardware quirks. If
   re-investigating turns up a bug that would also misfire on other
   architectures using the same code path, that's a signal to report it
   upstream instead of (or in addition to) patching around it here -- see
   the `rewrite_convolution.cpp`/ConvTranspose writeup in
   `MIGRATION_NOTES.md` for a worked example of that judgment call.
3. **Re-verify on real hardware, not just "applies clean."** A patch that
   compiles is not a patch that's confirmed fixed -- several patches in this
   repo's history were re-diffed successfully but flagged "not yet
   re-verified on real hardware" until someone actually ran the repro
   against the new binaries. Don't assume a clean apply means the original
   bug is still handled correctly.

## Host BIOS setting: keep PCIe ASPM off

On at least one gfx803 host, PCIe ASPM (link power management) being enabled
in the BIOS caused rare, extremely hard-to-diagnose stalls/hangs under GPU
load -- the kind that look like a driver or kernel bug and burn hours of
debugging before the actual cause turns out to be a power-management setting
outside the software stack entirely. Keep ASPM disabled in BIOS (and via
`setpci`/kernel cmdline if the board won't hand OS-level control to Linux)
on any gfx803 host until proven otherwise on that specific board. See
`tools/host-setup/` for a working `setpci`-based systemd unit for boards
where BIOS/firmware won't actually hand ASPM control to Linux.

## Host VBIOS setting: mining-tuned VRAM clocks cause random GPU faults and hangs

At least one gfx803 card in use with this repo (Sapphire RX 470 8GB Mining
UEFI, Hynix `H5GQ8H24MJR` VRAM) shipped with a mining-tuned VBIOS running
VRAM (MCLK) at 2000-2100MHz -- above what a 7Gbps-rated Hynix chip is
actually spec'd for. Under a correctness-checked compute workload this
produced two symptoms that both took real investigation to correctly rule
out as driver/software bugs (ioctl tracing, PM4 dispatch tracing,
kernel-side TLB-flush review, GPU-side wave-state capture via debugfs --
see `RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md` for the full investigation on both):

- `tools/correctness-suite/pool_sweep`: an intermittent (~50% of runs),
  deterministic-address GPU VM fault.
- vLLM and other sustained-load workloads: an intermittent, unrecoverable
  hang -- a wave parked forever in `s_waitcnt vmcnt(0)`, waiting on a
  vector-memory op that never returns. Killing the stuck process fails a
  KFD queue eviction and needs a reboot.

Mining workloads tolerate occasional VRAM bit errors that a
correctness-checked or long-running one won't -- it'll eventually surface
as a fault, a hang, or (worse) silently wrong results.

**Fix: flash a VBIOS whose VRAM clock matches the installed memory's real
rated speed**, via `amdvbflash`'s force-flash mode (`amdvbflash -f -p 0
<rom>`; always dump/keep the existing ROM first). Confirmed on hardware two
independent ways:

- Keeping the card's own mining VBIOS but capping MCLK to 1750MHz via core
  overdrive (`amdgpu.ppfeaturemask=0xffffffff` + `pp_od_clk_voltage`):
  64/64 clean runs, vs. repeated same-boot hangs/crashes at 2000MHz with
  every other binary held identical.
- Flashing a real Sapphire RX570 Nitro VBIOS with correct-vendor Hynix
  straps (`113-2E366AU-X56`, downloaded from
  https://www.techpowerup.com/vgabios/212597/212597) whose stock MCLK table
  tops out at 1750MHz, no overdrive needed: 75/75 clean runs at stock
  settings. This is the recommended fix for anyone with the same
  Sapphire RX 470 8GB Mining UEFI card and Hynix memory -- no software
  workaround required at all.

Two other RX570 VBIOS files with **Samsung** straps (wrong vendor for this
card's Hynix chips) failed to probe entirely (`SMU load firmware failed`,
`probe with driver amdgpu failed with error -22`) rather than hang -- a
different, harder failure mode. Match the VBIOS's memory-vendor strap to
the physically installed chips, not just the card model/VRAM size.

Check VRAM clock (`cat /sys/class/drm/card*/device/pp_dpm_mclk`) against
the card's actual rated spec before assuming a gfx803 GPU fault or hang
report is a software bug.

## What needs real gfx803 hardware to validate, and what doesn't

- **Needs the real card**: anything that dispatches a GPU kernel --
  `verify.py`, `tools/correctness-suite/`, `tools/reduce-harness/`, any real
  transcription/inference run, MIOpen's own `MIOpenDriver -V 1` verification.
  Silent miscompute is the recurring bug class here (`rocblas_status_success`
  returned with wrong numbers) -- CPU/emulation cannot reproduce it, and a
  patch that only "applies clean" and "compiles" has verified nothing about
  correctness.
- **Doesn't need the card**: whether a Dockerfile builds at all, whether a
  patch applies against a given pin, source-level tracing of *where* a bug
  lives (MIOpen's own `MIOPEN_ENABLE_LOGGING_CMD`/`MIGRAPHX_TRACE_COMPILE`
  traces and upstream source diffs found several root causes in this repo's
  history without ever touching a GPU), and cross-arch differential testing
  against an image for a *different* card (used repeatedly in
  `MIGRATION_NOTES.md` to separate "this line broke it" from "upstream never
  worked here").

## Convergence

Once the rocm7 line is confirmed at least as solid as `rocm6.4.4/` --
correctness-suite clean (already true), ORT suite parity (one open item, see
Status above), and comparable real-model results (already true) -- the two
lines get diffed and deliberately merged: whatever's still 6.4.4-only that
should generalize moves up, and the two independent copies collapse back
into a shared structure. Not done yet; both lines are still receiving
patches independently.

## See also

- `MIGRATION_NOTES.md` -- the detailed, as-found rocm7 investigation log.
  Read this before assuming something is broken or fixed; it's the primary
  source of truth for what's been checked and how.
- `rocm6.4.4/KERNEL_BUGS.md` -- the original gfx803 bug-hunting methodology
  and bug record for the 6.4.4 line.
- [`rocm-migraphx-ort-builder`](../rocm-migraphx-ort-builder) -- the
  mainline (gfx900+) build this repo split off from.
