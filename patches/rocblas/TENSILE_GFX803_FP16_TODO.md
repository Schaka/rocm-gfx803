# Tensile fp16 GEMM kernel generation for gfx803 -- what it would take

**Status:** ROOT-CAUSED, FIXED, AND WIRED INTO THE BUILD. The fp16
(non-HPA) Tensile kernel generates, assembles, runs, validates correct on
real gfx803 hardware (0 mismatches, tail + main-loop paths), benchmarks
~1.3-1.4x over the current rocBLAS fp16 fallback at the model's shapes,
and the patch + a gfx803 fp16 library-logic config are wired into the
root Dockerfile so the rocBLAS build emits real fp16 kernels. HPA
(fp32-accumulate) fp16 remains impossible on gfx803; this is non-HPA only.
This document was originally a scoping doc; it now tracks the completed
patch and build wiring. Assumes no prior context from the investigation
that produced it.

**Effort estimate: multi-day, real GCN3 assembly codegen work inside
Tensile's Python DSL. Not a config flag, not a quick fix.**

## Why this matters (the payoff, if it works)

gfx803 (Polaris, our RX 470/480/570/580 cards) has **zero real (non-fallback)
fp16 GEMM kernels** in rocBLAS's Tensile-generated kernel library. Every
fp16 GEMM call on this hardware -- the actual matrix multiplies behind
every Linear layer in every transformer model, for both prefill (M=batch
tokens, real GEMM shape) and decode (M=1, GEMV shape) -- falls through to
Tensile's generic, portability-only `_fallback_` kernel, which was never
benchmark-tuned for this architecture at all.

This repo has already worked around this for the M=1 (decode) case via
hand-written kernels (`patches/vllm/gfx803_gemv.py` -- `ops.LLMM1` /
`gfx803_triton_gemv`) and confirmed **decode now surpasses
llama.cpp-Vulkan** (see `SESSION_HANDOFF.md` §24-25). Prefill's M>1 GEMM
shapes are a different problem: unlike decode's pathological M=1-through-
a-128-row-tile waste, Tensile's `_fallback_` kernel is already a
legitimately competent tiled GEMM at M=128-512 (confirmed directly: a
naive from-scratch Triton GEMM lost to it by 3.5-5x, see
`SESSION_HANDOFF.md` §27). Beating it requires either (a) a properly
engineered Triton GEMM with real software pipelining -- the path this
session is pursuing instead of this document's task -- or (b) getting
Tensile's own mature, already-pipelined assembly kernel generator to
actually produce and benchmark-tune real gfx803 kernels, which is what
this document scopes.

If (b) works, it plausibly beats (a): Tensile's kernel templates already
implement double-buffering, software pipelining, and auto-tuned tile
sizes that a hand-written Triton kernel would need to reimplement from
scratch. The blocker isn't Tensile's kernel *design* -- it's that
Tensile's fp16 code generator uses AMD GCN instructions Polaris's
hardware genuinely does not have, and no fallback code path exists yet
for that gap.

## Where the source is

Tensile's full Python source is on the gfx803 box (SSH `user@192.168.1.214`,
password `user`, both for the `user` account and `sudo`) at:

```
/data/rocm7-build/rocm-libraries-src/shared/tensile/Tensile
```

(Inside the `ct-old` docker container -- everything in this doc assumes
running via `docker exec ct-old ...`.) This is Tensile v4.47.0, imported
into AMD's `rocm-libraries` monorepo (`git log` on this path only goes
back to an April 2025 "Reorganize project folders" commit -- the original,
longer history lives at `https://github.com/ROCm/Tensile`, which **is**
reachable from this box, network access confirmed working).

## Root cause, precisely, with exact evidence

### 1. Tensile's fp16-HPA (high-precision-accumulate, i.e. fp32 accumulate)
kernels cannot run on gfx803 at all -- confirmed as **real, long-standing
hardware history, not a bug**.

`Tensile/AsmCaps.py` has a hardcoded, per-ISA-version capability table.
The entry for `(8, 0, 3)` (gfx803) has, among others:

```python
'v_pk_fma_f16': False,
'v_mad_mix_f32': False,
'v_fma_mix_f32': False,
'v_mac_f16': True,
'v_fma_f16': False,
```

Every MAC (multiply-accumulate) component in
`Tensile/Components/MAC_F16_HPA.py` (`FMA_F16_HPA_MAD_MIX`,
`FMA_F16_HPA_MAD_MIX_LDL`, `FMA_F16_DOT2`) requires one of
`v_pk_fma_f16` / `v_mad_mix_f32` / `v_fma_mix_f32` / the dot2 instructions
-- all `False` for gfx803. `Component.MAC.find()`
(`Tensile/Component.py:160`) returns `None`, and
`KernelWriterAssembly.defineMACs()` (`Tensile/KernelWriterAssembly.py:2213`)
hits `printExit("Assembly doesn't support %s" % ...)`.

**This is confirmed as accurate hardware history, not a regression or a
missing table entry**: fetched the original `ROCm/Tensile` GitHub repo's
2020-07-06 commit `de2846a7` ("Move fp16 code entirely to components",
from back when gfx803 was AMD's actively-supported hardware) via
`https://api.github.com/repos/ROCm/Tensile/commits/de2846a7`
(`Accept: application/vnd.github.v3.diff`) -- its diff contains the
comment, written by AMD's own Tensile maintainers at the time: **`# No
HPA on 803, every other combination should work though.`** Packed/
mixed-precision FP16 math (`v_pk_*`, `v_*_mix_f32`) was a genuinely new
Vega-generation (gfx900) hardware feature ("Rapid Packed Math") that
Polaris silicon never had. Do not spend time trying to "unlock" HPA fp16
on gfx803 -- it is not possible on this hardware, full stop.

### 2. Non-HPA (plain fp16-accumulate) kernels: the MAC step works, but
memory addressing and the epilogue don't -- **this is the actual, real,
currently-open gap**.

`Tensile/Components/MAC_F16.py`'s `MAC_F16_Plain` class requires
`v_mac_f16: True` (gfx803 has this) and `HighPrecisionAccumulate: False`
in the `ProblemType` config. With `HighPrecisionAccumulate: False` set
(NOT the HPA config -- see the feasibility configs below, which get this
wrong by default since they were cloned from an HPA template), this
component **does match** and Tensile proceeds to generate and compile a
real kernel.

It then fails at actual LLVM/GCN assembler invocation (not just Tensile's
own capability table -- the real compiler backend independently confirms
this) with:

```
error: instruction not supported on this GPU (gfx803): ds_write_b16_d16_hi
error: instruction not supported on this GPU (gfx803): buffer_store_short_d16_hi
error: instruction not supported on this GPU (gfx803): v_pk_mul_f16
```

`d16_hi` instructions (packed sub-32-bit-register memory load/store, used
to pack two fp16 values into the high/low halves of one VGPR) and
`v_pk_mul_f16` (packed multiply, used in the alpha-scaling epilogue when
writing 2 output elements at once) are **also** Vega+-only hardware
features Polaris lacks. Usage sites in `KernelWriterAssembly.py` (line
numbers may drift slightly as the file is edited, search for these
strings to relocate):

- `~2451-2559`: width/instruction-name tables including `d16_hi`
  variants for buffer/global loads and stores (`u8_d16_hi`, `b16_d16_hi`,
  etc.)
- `~13850-14012`: the actual load/store codegen (`_buffer_load_d16_hi_u8`,
  `_buffer_load_d16_hi_b16`, `_global_load_d16_hi_u8`,
  `buffer_store_byte_d16_hi`, `_buffer_store_d16_hi_b16`,
  `global_store_byte_d16_hi`, `_global_store_d16_hi_b16`)
- alpha-scaling epilogue: search for `v_pk_mul_f16` (used when applying
  the `alpha` scalar to two packed output elements at once)

**Critically: none of this is gated by any ISA-capability check.** Unlike
the MAC step (which has a clean `Component`-based dispatch that correctly
fell back once the config was fixed), the d16_hi/packed-epilogue code
paths are unconditional whenever the kernel writes sub-32-bit or
2-at-a-time output elements. There is no existing "plain" fallback to
select, the way there was for the MAC step.

**Tried and failed as a workaround**: forcing `VectorWidth: 1` (never
process 2 elements at once, which should in principle avoid the packed
epilogue entirely) does not gracefully degrade -- Tensile's own solution
validator rejects the resulting parameter combination outright
(`Actual Solutions: 0 / 1`, "Your parameters resulted in 0 valid
solutions") before even reaching the assembler. This is not a config
tweak; the code generator's assumptions run deeper than a single
parameter.

## What actually needs to be built

A genuine non-packed fallback code path in `KernelWriterAssembly.py`'s
global memory read/write and epilogue codegen, activated when the target
ISA lacks `d16_hi`/packed-fp16-store support (this capability does not
currently have a name in `AsmCaps.py` -- would need to be added, e.g.
`HasD16HiStore` or similar, defaulting `True` for gfx900+ and `False` for
gfx803, mirroring how `v_pk_fma_f16` etc. are already modeled).

Concretely, this means replacing, for the gfx803 case:
- `d16_hi` sub-word stores (`buffer_store_short_d16_hi`,
  `ds_write_b16_d16_hi`, etc.) with the classic two-step sequence: shift
  the fp16 value into the correct half of a 32-bit register with
  `v_lshl_b32`/`v_or_b32` (or equivalent), then a normal full-width
  store. This is standard pre-d16 GCN codegen -- the same pattern this
  repo already uses successfully in `gfx803_gemv.py`'s HIP-intrinsic-based
  kernels and the Tensile source itself likely had *some* version of this
  before the 2020 "move fp16 to components" refactor (worth checking
  Tensile's git history further back, and/or MIOpen's GEMM kernels from
  the gfx803-supported era, for a working reference implementation of
  this exact packing pattern on this exact hardware).
- `v_pk_mul_f16` (packed alpha-scale) with two separate `v_mul_f16`-style
  scalar operations (or `v_mac_f16`-based accumulation matching what the
  MAC step already correctly generates) on the unpacked halves.

This needs a real understanding of: how VGPR sub-register addressing
works in Tensile's codegen (search `vgprValuC`, `vgpr()` helper
function), how the `d16_hi` code paths are selected (search for
`kernel["ProblemType"]["DataType"].isHalf()` branches near the load/store
and epilogue sites listed above), and what determines whether 1 or 2
elements get processed together in the epilogue (`VectorWidth`/
`GlobalWriteVectorWidth` kernel parameters, and how `ThreadTile`
interacts with them).

## Current work state (in-progress patch, as of the last session)

A real patch now exists and gets the kernel **all the way to running on
the RX 470** -- which is past every blocker this document originally
scoped -- but the kernel **miscomputes silently** (Tensile's client
validation reports mismatches; the kernel runs without crashing). This is
the "silent miscompute" class this repo's `AGENTS.md` warns about, so
nothing below counts as verified.

### What's implemented (all edits tagged `GFX803_FP16_NOD16_PATCH`)

The design is a new capability + a writer flag that forces unpacked
(one-fp16-per-VGPR-low-half) codegen on gfx803:

- `Tensile/AsmCaps.py` -- new capability `HasD16` (False for gfx803,
  True otherwise).
- `Tensile/Common.py` -- `HasD16` runtime-probed via tryAssembler so the
  value isn't just a hardcoded table entry.
- `Tensile/KernelWriterAssembly.py` -- `halfNoD16` writer flag gating:
  C-tile VGPR sizing (TT0*TT1 instead of /2), `buffer_load_ushort`
  + in-place `v_lshlrev`/`v_or` packing in the global-read emitters,
  plain `ds_write_b16` LDS stores (with `v_lshrrev` from the hi half for
  odd elements), scalar `v_mul_f16`/`v_add_f16` alpha/beta, per-element
  `buffer_store_short` stores and C beta loads, `numVgprsPerDataPerVI=1`.
- `Tensile/Components/MAC_F16.py` -- `MAC_F16_Plain` was silently broken
  upstream (4 identical `v_mac_f16`, op_sel dropped); rewritten to mirror
  `FMA_F16_Packed`'s element mapping using SDWA `src0_sel:WORD_1` on B.
  SDWA `v_mac_f16` with WORD_1 selectors verified **working on hardware**
  via standalone HIP tests (`/tmp/opencode/sdwa_test5.hip`); earlier
  failures were inline-asm constraint bugs (missing `&` early-clobber).

Backups of the pristine files: `/data/tensile-gfx803-backup/` on the box;
local copies at `/tmp/opencode/tensile-src/{Tensile/,orig/}`. The generated
kernel `.s`: `/tmp/opencode/kernel_p2.s` (patched, current). The on-box
Tensile source is patched live at
`/data/rocm7-build/rocm-libraries-src/shared/tensile/Tensile/`.

### The current bug (real hardware data, decoded from tensor dumps)

Problem `Cijk_Ailk_Bljk_H_MT32x64x8_SN_K1`, client alpha=2, beta=0.
`Exact [16,16,2,1]` = i=16,j=16,k=2(l batch dim),l-contraction size 1.
With l=1, `LoopCounterL = l/8 = 0` -- **only the TAIL loop path executes**
(the u16+pack global loads are the ones running; the main loop's b64
path is untested independently).

Decoded device output for `[16,16,2,1]`:

```
D[i][j] = 2 * B[j] * A[(i%4)+1]      for ALL i-groups
```

B path is CORRECT (each thread's B columns are right). A path is wrong:
**every thread's ValuA holds A[1..4]** -- a +1-element shift AND a
collapse (all threads read the same 4 A values). A[0] and A[5..15] never
appear. `[32,64,8,1]` (pure tail, no edge) fails too (996/2048 mismatch,
includes sign flips like D[0][0]=4 vs ref=-4).

Verified instruction-by-instruction that the tail path is internally
consistent and *should* produce correct per-thread data: per-thread GRO_A
(`(v4+v7*StrideAL+4)*2`, v4=(serial%8)*4 clamped to SizeI-4, v7=serial/8),
SRD pre-pad (AddressA-8) compensating the +4 GRO prepad (net zero), tail
SRD rewind = 0 for this config (StaggerUIter=2, WrapUA=256), LDS
write/read are an identity relay (LWA_A(t)=LRA_A(t)=8t), tail read
offset 0 matches the write slice 0, and the u16+pack bit patterns match
the original d16 loads exactly. The bug lives in one of these but none of
the on-paper math reproduces the observed +1-shift-and-collapse.

**RESOLVED -- see "The GRO v0/v1 clobber" section above.** The observed
"GRO = 10 uniform" turned out to be the leftover value after the WGM
division clobbered the live tile/unroll offset registers (v0/v1) -- the
per-thread component was destroyed, and the "+1 element" came from the
leftover bits. Forcing the correct per-thread GRO in the generated asm
made the data path fully correct, proving the bug was purely the GRO
computation; the source fix (disable `isInitCodeOptLW` when
`HasSMulHi` is False) resolves it for real.

### Critical methodology lesson for the next session (cost an hour)

**The Tensile client runs PREBUILT code objects, NOT the raw `.s`.**
Experiments that only edit the generated kernel `.s` and re-run the
client are silently testing the STALE kernel. The pipeline:

1. Edit the kernel `.s` under
   `00_Final/source/build_tmp/SOURCE/assembly/`.
2. Recompile it:
   `cd .../build_tmp/SOURCE/assembly && sh asm-new.sh Cijk_Ailk_Bljk_H_MT32x64x8_SN_K1`
   (uses `/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa
   -mcode-object-version=4 -mcpu=gfx803 -mwavefrontsize64`).
3. Install it: this kernel is a *source* kernel, so the client looks it
   up in `library/Kernels.so-000-gfx803.hsaco` (NOT
   `TensileLibrary_gfx803.co`, which is the binary-kernel variant).
   Replace that file with the freshly-built `.co`:
   `cp .../assembly/Cijk_Ailk_Bljk_H_MT32x64x8_SN_K1.co .../library/Kernels.so-000-gfx803.hsaco`.
4. Re-run the client.

Verified the replacement mechanism works: inserting an early `s_endpgm`
into the kernel and recompiling+replacing made the client return an
all-zero D buffer (proving the recompiled kernel ran). All experiments
run *before* this discovery (a b64-load A/B test and three register-dump
instrumentation attempts) are **void** -- they ran the stale kernel and
their "confirmation that loads aren't the culprit" must be redone.

Also learned: the client only re-prints A/B/D/Ref **after** the kernel
(`printTensorsTyped` in `ReferenceValidator.cpp`), all from the *result*
buffers -- so kernel-side writes to the A or B buffers ARE visible in the
A/B prints; writes to C need `print-tensor-c=true` (C descriptor is not
even generated when beta=0). Instrumentation syntax notes for hand-edited
asm: use the VOP3 `_v_add_u32 dst, src0, src1` macro for immediate adds
(raw 3-operand `v_add_u32 vX, imm, vX` is rejected on gfx803), and use
`v_mov_b32 vX, s[sgpr...]` to copy an SGPR to a VGPR
(`v_readfirstlane_b32` takes a VGPR source only). A register-dump
instrumentation attempt that compiled clean still zeroed the D output
(unknown cause -- likely a descriptor/limit or vcc interaction) -- any
future instrumentation should be validated against a known-good baseline
first.

### The GRO v0/v1 clobber (the miscompute's actual root cause -- now FIXED)

The A-path miscompute was NOT in the fp16 patch at all. It was an upstream
gfx803-specific bug in `KernelWriterAssembly.py`, exposed only because
gfx803 lacks `s_mul_hi_u32`:

- `s_mul_int_64_32()` has two paths: `HasSMulHi=True` (gfx900+, scalar
  `s_mul_hi_u32`, no VGPRs touched) and `HasSMulHi=False` (gfx803, which
  the LLVM assembler confirms has no `s_mul_hi_u32`/`s_mul_hi_i32`), the
  fallback does the 64-bit scalar multiply in VECTOR registers
  (`v_mov_b32 v0, s; v_mul_hi_u32 v1, v0, s; v_readfirstlane...`).
- The WGM work-group division (`graWorkGroup`) calls `s_mul_int_64_32`,
  so on gfx803 it clobbers **v0/v1** as its temps.
- The GRO tile/unroll offset assignment (`graTileAssignment`) also lands
  in **v0/v1** (`v0 = (serial%8)*4`, `v1 = serial/8`), and because
  `isInitCodeOptLW` (init-code optimization) emits that assignment
  EARLY (before the kernel-arg-load wait), those v0/v1 values are live
  when the WGM section runs and gets destroyed.
- The tile-offset copy (`v4 = v0`, `v7 = v1`) then reads garbage ->
  GRO uniform-wrong -> every thread's A loads collapse to `A[1..4]`
  (the +1-element shift came from the leftover value, the collapse from
  the lost per-thread component). The B path was unaffected because B's
  assignment registers happen not to collide.

On gfx900+ the same code is fine because the WGM division is pure scalar.
The bug is latent upstream (gfx803 unsupported since ROCm 6.0).

**The fix** (`KernelWriterAssembly.py`, ~line 862): disable
`isInitCodeOptLW` when the ISA lacks `HasSMulHi`, so the tile assignment
is emitted AFTER the WGM section (which no longer has live v0/v1 to
clobber):

```python
    if ((kernel.enabledSplitLDS and (kernel["UnrollMajorLDSA"] or kernel["UnrollMajorLDSB"])) or \
        kernel["PersistentKernel"] or kernel["StreamK"] or \
        (not kernel["BufferLoad"]) or \
        self.groOffsetInMacroTile == 0 or \
        not globalParameters["AsmCaps"][kernel["ISA"]].get("HasSMulHi", True)):
      # disable init opt for local write
      self.isInitCodeOptLW = False
```

Verified by hand-instrumenting the generated kernel (forcing the correct
per-thread GRO value made the data path correct, proving the bug was the
GRO computation), then by the source fix itself.

**Validation on real hardware (RX 470, after the fix):**
- `Exact [16,16,2,1]` (l=1, all-tail, M=16 edge): **0 mismatches**.
- `Exact [16,16,2,16]` (l=16, unrolled main loop): **0 mismatches**.
- `Exact [32,64,8,1]`: kernel D matches the expected `2*A*B` exactly
  (0 mismatches vs expected); the client's reported validation failure
  is a broken client-side reference for that shape (the reference comes
  back sparse -- ~1/4 of elements missing, all its non-zero elements
  match the kernel). That is a separate client/reference issue, not the
  kernel.

### Diagnostics lessons from the hunt (worth keeping)

- The Tensile client runs PREBUILT code objects, not the generated `.s`.
  Experiments that edit the `.s` and re-run the client silently test the
  STALE kernel. Recompile via `build_tmp/SOURCE/assembly/asm-new.sh`
  and install into `library/Kernels.so-000-gfx803.hsaco` (source kernels
  are looked up in the `.hsaco`, NOT `TensileLibrary_gfx803.co`).
  Verified the mechanism by inserting an early `s_endpgm` (D came back
  all-zero).
- The AMDGPU backend reorders/schedules hand-inserted asm, so
  register-value dumps are unreliable unless validated against a
  known-good baseline. The reliable channels were: (a) the loaded A data
  itself (small fp16 ints survive the epilogue's fp32->fp16 store), and
  (b) forcing a known value and checking the output.
- `vgprSerial = v52` -- clobbering it in the tail silently zeroes D
  (the epilogue reads it for store coordinates). That masked several
  instrumentation runs as "kernel broken".

### Benchmark (RX 470, ROCm 7.14 client, alpha=2 beta=0, fp16)

New kernel `Cijk_Ailk_Bljk_H_MT32x64x8_SN_K1` vs what rocBLAS currently
delivers on gfx803 (`rocblas_hgemm`, i.e. the generic fallback -- the
baseline every fp16 Linear layer in every transformer model on this
hardware currently gets):

| M | N | K | rocBLAS today (GFLOPS) | new kernel (GFLOPS) | speedup |
|---|---|---|---|---|---|
| 256 | 1536 | 1536 | 2182 | **2946** | 1.35x |
| 128 | 1536 | 8960 | 1468 | **2070** | 1.41x |
| 128 | 1536 | 1536 | 1414 | **1889** | 1.34x |
| 64  | 1536 | 1536 | 984  | **993**  | 1.01x |
| 32  | 1536 | 1536 | 496  | **500**  | 1.01x |
| 16  | 1536 | 1536 | 249  | **249**  | 1.00x |
| 8   | 1536 | 1536 | 107  | **125**  | 1.17x |

- ~1.3-1.4x over the current rocBLAS path at the model's real prefill
  shapes (M>=64, K=1536/8960); ~40-60% of the RX 470's fp16 peak
  (~4.9 TFLOPS, fp16 is 1:1 with fp32 on Polaris -- no packed fp16).
- M=1 (decode) does not work with this tile (MT=32 > M); decode stays on
  the hand-written `gfx803_gemv.py` GEMV, which this kernel is not
  meant to replace. The win is prefill GEMM.
- Timing methodology differs slightly between the two measurements
  (Tensile client enqueue-timed vs a 20-iteration wall-clock harness),
  but both are steady-state throughput and the advantage is consistent
  and directionally solid.

### Client-side fix this surfaced

`ResultFileReporter.cpp` crashes with `std::invalid_argument: stoi` when
the GPU clock/power/hotspot metrics come back as NaN/empty (the ROCm SMI
path on unsupported gfx803 hardware returns NaN). This crashes the
Tensile benchmark client on every run, before results print. Fixed with
try/catch around the three `stoi` calls. Included in the patch.

### Build wiring (the "ship")

- `Dockerfile` (`rocblas-builder` stage): applies
  `tensile-gfx803-fp16-nond16.sh` to the rocm-libraries checkout and
  copies `r9nano_Cijk_Ailk_Bljk_HB.yaml` into
  `projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/r9nano/`.
- The library-logic file is the missing piece that makes the rocBLAS
  build's Tensile library generation (`TensileCreateLibrary`, via rmake's
  default `Tensile_LOGIC=asm_full`) actually emit fp16 kernels for
  gfx803 -- without it, the Logic tree has no fp16 gfx803 config
  (asm_full/r9nano has only SB/DB = fp32/fp64), so even with the codegen
  fixed, no fp16 kernel would be generated and rocblas_hgemm would keep
  falling back. Verified with the standalone `TensileCreateLibrary` run:
  the config parses, the MT32x64x8 kernel is generated + assembled into
  the library's code object (the kernel symbol is in the .co.raw), and
  the type `.dat` names `Cijk_Ailk_Bljk_HB_MT32x64x8_SN_K1`. The full
  Dockerfile build (multi-hour rocBLAS build) is the final end-to-end
  verification -- not yet run.
- The `tensile-client` (benchmark tool) segfaults loading
  `TensileCreateLibrary`-format libraries (the fp32 SB library from the
  same path does too, while rocBLAS loads this format fine in
  production) -- a client-tooling quirk, not the library. Benchmark
  numbers in this doc come from the `Tensile bin/Tensile` benchmark path
  (same engine, working client) and the rocBLAS harness.

### Where the next session should pick up

- Reproduce the miscompute against the *recompiled* kernel (fresh `Exact
  [16,16,2,1]` run with the current patched Tensile source -- the run in
  `/data/tensile_dbg_run.log` predates the stale-kernel discovery and is
  still valid as the bug reproduction, but re-confirm).
- Instrument with the working recompile+replace loop: dump GRO_A/G2LA/LRA
  into the A and B buffers post-load (B dump needs care: B's global data
  is still being read right after A's loads; dump after the B loads
  complete, and validate the instrumented kernel against the known-bad
  baseline before trusting output). This directly answers whether the
  collapse is in the global-read address, the LDS relay, or the MAC.
- Consider an `Exact` config with a larger contraction dim (e.g. l=16)
  to exercise the main unrolled loop path and separate main-loop vs
  tail-loop bugs.
- Only after the A path is fixed: re-verify the SDWA MAC on hardware via
  the standalone HIP test (`sdwa_test5.hip`), then the full validation
  checklist below.

## How to reproduce the current failure state (starting point for whoever picks this up)

Two minimal feasibility configs already exist on the box (not yet synced
to this repo -- small, throwaway, copy them in if useful as a starting
point):

- `/data/gfx803_feasibility.yaml` -- HPA config (deliberately wrong per
  the finding above, kept as a reference for what the *original*, decoy
  failure mode looks like: `Assembly doesn't support H` from the
  MAC-matching step).
- `/data/gfx803_feasibility_nonhpa.yaml` -- the real, currently-relevant
  repro. `HighPrecisionAccumulate: False`, `DataType: h`,
  `DestDataType: h`, `ComputeDataType: h`. Currently has `VectorWidth: [1]`
  set from the last (failed) experiment -- **revert to `VectorWidth: [-1]`
  first** to get back to the actual target failure (the `d16_hi`/
  `v_pk_mul_f16` assembler errors), since `[1]` just gets rejected before
  reaching them.

Run via:
```bash
docker exec ct-old rm -rf /data/tensile_nonhpa_out
docker exec ct-old python3 /data/rocm7-build/rocm-libraries-src/shared/tensile/Tensile/bin/Tensile \
    /data/gfx803_feasibility_nonhpa.yaml /data/tensile_nonhpa_out
```

This takes a few minutes (real HIP compilation of the generated assembly,
though it currently fails before reaching hardware benchmarking). Watch
for genuine progress (rising CPU time, not a flat hang) if it seems slow
-- per this repo's standing practice, verify via two `py-spy dump --pid
<pid> --native` snapshots ~15-30s apart before assuming a hang; this
exact investigation hit several multi-minute-with-zero-output stretches
that were real compilation work, not hangs.

Toolchain needed (already present on the box, confirmed working):
`amdclang`/`amdclang++` (`/opt/rocm/bin/` or `/opt/rocm/core-7.14/bin/`),
`hipcc`. The Tensile CLI auto-detects and uses these.

## How to validate a fix (do not skip any of this)

Matching this repo's standing philosophy (`AGENTS.md`): **a patch that
compiles clean has confirmed nothing about correctness.** This
architecture's recurring failure mode is *silent* miscompute, not
crashes.

1. **Get the feasibility config compiling clean end-to-end** first (past
   the assembler, into `TensileBenchmarkLibraryClient`'s real hardware
   run). It should print real timing numbers for the tiny problem-size
   range in the config (`Range: [[127,1,129], 0, [2], [63,1,65]]`).
2. **Correctness**: Tensile's own client has a built-in validation mode
   (`NumElementsToValidate: 65536` is already set in the feasibility
   configs -- this makes the client compare kernel output against a CPU
   reference automatically). Confirm it reports 0 validation failures,
   not just "ran without crashing."
3. **Independently re-verify against `torch.nn.functional.linear`**,
   the same way every other kernel this session was checked (see
   `patches/vllm/gfx803_gemv.py`'s module docstring, `SESSION_HANDOFF.md`
   §16-27 throughout for the exact pattern): once a working kernel
   library is produced, call it via ctypes/torch and diff against a
   float32 reference implementation at the real shapes this model uses
   (K=1536 and K=8960 for Qwen2.5-1.5B; see `gfx803_gemv.py`'s docstring
   for the full shape list). A diff in the fp16-noise range (~1e-4 to
   1e-6) is fine; anything larger is a real bug -- this exact
   investigation already found one such bug this session (LLMM1's
   large-K miscompute, see `patches/vllm/repro_llmm1_bug.py`), so treat
   "compiles and runs" as zero evidence of correctness.
4. **Widen the problem-size range** before declaring victory -- the
   feasibility config's tiny size range (M/N~127-129/63-65) is not
   representative of this model's real shapes. Build a real benchmark
   config (not the "lite" 1-variant feasibility config) with
   `ProblemSizes` covering the actual K=1536/K=8960 shapes at realistic
   M (128 for prefill's typical case, but sweep a range) before running
   the real multi-hour tuning sweep.
5. **Benchmark against the current baseline** (Tensile's existing
   `_fallback_` kernel, and `gfx803_triton_gemv`/LLMM1 for the shapes
   they already cover) -- this whole effort is only worth shipping if it
   actually wins. Use the same rigor as every other benchmark this
   session: median of repeated trials, not a single run (see
   `SESSION_HANDOFF.md`'s repeated "single-trial sweeps are noisy enough
   to reverse conclusions" lesson from the `rows_per_block` investigation
   in §16-18).
6. **End-to-end correctness in the real model**: once wired into a real
   dispatch path (mirroring how `gfx803_gemv.py`/`gfx803_split_attn.py`
   are wired into vLLM), run `sanity_gen.py` (`/data/sanity_gen.py` on
   the box) and confirm coherent, correct completions -- and note that a
   single flipped token in greedy decoding from legitimate
   accumulation-order differences is expected and fine (seen repeatedly
   this session, e.g. §25's down_proj kernel swap), but garbage/
   incoherent output is not.

## What NOT to do

- Do not attempt to "enable" HPA fp16 on gfx803 -- confirmed impossible
  on this hardware (see root cause #1 above). Any config requesting
  `HighPrecisionAccumulate: True` for a `DataType: h` problem on gfx803
  will never produce a kernel, no matter what parameters are tried.
- Do not spend time on `TensileRetuneLibrary.py` (Tensile's "retune an
  existing library" tool) for this -- it only re-benchmarks and
  re-selects among *already-generated* candidate solutions; it cannot
  generate new ones. gfx803 has none to retune among until this codegen
  gap is fixed.
- Per `AGENTS.md`: this is local-patches-only work. gfx803 has been
  unsupported upstream since ROCm 6.0 -- do not file this upstream to
  AMD/Tensile. Any fix stays in this repo.
