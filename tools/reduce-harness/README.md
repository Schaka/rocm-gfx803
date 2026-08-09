# gfx803 MIOpen dynamic-reduce kernel harness

Faithful, minimal repro for the MIOpen `miopenReduceTensor` intermittent
wrong-sum bug on gfx803 (see `gfx803/KERNEL_BUGS.md`, "The ReduceSum
kernel-cache mystery").

It drives **MIOpen's actual compiled CK reduction kernel code objects** through
`hipModule` — the same kernel binaries MIOpen JITs — but with **no MIOpen C++
dispatch in the loop** and with `hipMalloc`/`hipFree` data-buffer **address
reuse** between calls (the confirmed trigger). It reproduces wrong reductions
purely from the real kernel + plain HIP runtime + caller-side allocator reuse,
which pins the defect at the kernel/runtime level rather than MIOpen's host
logic.

This is a portable re-validation tool: rebuilt and rerun against any ROCm stack
(including the ROCm 7 port, where we must confirm the bug is fixed / unchanged).
There are no MIOpen dependencies beyond the prebuilt kernel code objects.

## What it proved (on ROCm 6.4.4 / gfx803)

Running the default 52-shape sweep (reduce axis 0 of `[m,n]` floats) produces
wrong reductions on a subset of shapes (observed 10/52), concentrated in the
same groups MIOpen's own test hits (n=128 mid-size `m`, and n=16 `m>=512`
blockwise). Because the harness uses MIOpen's real kernels with no MIOpen host
code and the same malloc/free reuse ORT performs, it demonstrates the defect is
in the kernel/runtime layer.

Note: exactly which shapes go wrong shifts run-to-run / layout-to-layout (that
is the bug's defining trait); a shape that is "ok" here may be "FAIL" in a
different process layout and vice-versa. Count/rate is what is stable.

## Layout

```
miopen_reduce_kernel_harness.cpp   the harness (configurator replica + hipModule dispatch)
harness_exp.cpp                    experiment variant: HAR_MODE=reuse|fresh|reuse-in|reuse-out
                                   (+ ring/copyin/bench knobs) used to pin the VA-recycling
                                   mechanism; builds the same way as the main harness
build_variants.sh                  compiles the CK wrappers -> .hsaco code objects (LLVM JIT)
codeobj/*.hsaco                    prebuilt gfx803 code objects (one per method/padding/call)
README.md
```

The bug this reproduces is **fixed at the runtime level** by
`gfx803/patches/rocr/va-reuse-defer.patch` (see `gfx803/KERNEL_BUGS.md`,
"FIXED (runtime) -- 2026-08-09"). With the patched libhsa the 52-shape sweep
is 0/52; against stock libhsa it is ~6-10/52. Keep using this harness to
re-validate whenever the ROCR/MIOpen layer is touched (including the ROCm 7
port).

## Pre-requisites

- A ROCm install with `hipcc` at least as old as the code objects' target arch
  (gfx803), and a gfx803 GPU.
- The MIOpen CK kernel sources (the wrappers live in MIOpen's
  `src/composable_kernel/composable_kernel/src/kernel_wrapper/`); `build_variants.sh`
  needs them on disk and the CK include tree.

## Build the code objects (one time per ROCm)

```
./build_variants.sh    # <- adjust the CK source/include paths at the top
```
This emits `codeobj/v_*.hsaco` and `codeobj/v2_*.hsaco` (first/second call
variants for threadwise/warpwise/blockwise/multiblock at the padding combos the
sweep needs). The committed `codeobj/` are for ROCm 6.4.4 / gfx803; **rebuild
them for a different ROCm**.

## Build + run the harness

```
hipcc -O2 -I<rocm>/include miopen_reduce_kernel_harness.cpp -o reduce_harness \
      -L<rocm>/lib -Wl,-rpath,<rocm>/lib
REDUCE_HSACO_DIR=<dir containing the .hsaco files> ./reduce_harness
```

Prints one line per shape:
`[k] m=.. n=.. ok|FAIL bad=..` then a `Total: N, failed: F` summary.
Exit code 0 iff zero failed.

`--shapes m1 n1 m2 n2 ...` runs an explicit shape list instead of the default
52-shape sweep.

## How to validate a fix / a new ROCm

1. Rebuild `codeobj/` for the target ROCm (same CK source).
2. Run the harness; record the baseline failure count.
3. Apply candidate fix (e.g. a HIP-runtime patch), rerun, confirm count -> 0.
4. Also run the real ORT `ReductionOpTest` suite — standalone harnesses have
   historically passed while ORT still failed, so both gates are required.
