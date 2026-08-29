# gfx803 wave-state debugging

Reads the GPU's own shader state during a live hang. This is what finally
localized the silent dispatch hang (RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md problem 1)
after months of experiments against the queue/doorbell/interrupt layers came
back negative -- none of those layers can park a wave in `s_waitcnt`, and
nothing had looked at the shader side.

The localization itself held up: the hang was root-caused to VRAM-clock
marginality (see RESOLVED_VRAM_MARGINALITY_INVESTIGATION.md's top-of-file summary), and a
wave stuck in `s_waitcnt vmcnt(0)` -- waiting on a vector-memory op that
never returns -- is exactly the shader-side signature that condition
predicts. These tools remain useful for any future gfx803 hang that isn't
this one.

Everything here is read-only against `amdgpu`'s debugfs. It needs root on the
host (not in a container), and the GPU must still be enumerated.

## Why it does not kill the hung process

A hung process left alone spins in userspace with **zero** kernel-side
consequence. It is specifically tearing it down that forces a KFD queue
eviction, which fails (`qcm fence wait loop timeout expired` -> "unsuccessful
queues preemption") and leaves the GPU needing a reboot -- sometimes a
physical power-cycle, since the OS shutdown then hangs on the same state. So
`repro-and-capture.sh` captures and stops, leaving the process alive for
further inspection. Reboot when you are done looking, and expect to need
physical access.

## Files

- `wavescan.py` -- scans every SE/SH/CU/SIMD/wave slot and dumps resident
  waves: PC, current instruction, `STATUS`, `TRAPSTS` (decoded -- `MEM_VIOL`
  is the one that matters), `IB_STS` (live `vm_cnt`/`lgkm_cnt`, i.e. how many
  memory ops are actually outstanding), and `M0`. Also prints distinct PCs, so
  "one stuck wave" vs "the whole dispatch stalled" is obvious at a glance.
  Writes the minority PC to `/tmp/stuck_pc.txt` for `codedump.py`.
- `codedump.py <pid>` -- reads the code object around the stuck PC out of
  `/proc/<pid>/mem`. Code objects are mapped in the process at the same VA the
  wave's PC reports, so this needs no GPU involvement at all.
- `gprdump.py <se> <cu> <simd> <wave>` -- VGPRs/SGPRs of one wave. For the
  blit copy kernel the flat load address is `v[2:3]` and the store address is
  `v[4:5]`; reading them tells you the exact address a stalled load is waiting
  on, which is the fork between "software computed a bad address" and
  "hardware dropped a valid request".
- `repro-and-capture.sh` -- runs the workload until one attempt hangs, then
  fires the above automatically. Runs the workload as the invoking user
  (rootless podman owns the image store) and uses sudo only for debugfs; do
  **not** run the whole script under sudo or podman will not find the image.

## Disassembling a capture

`llvm-objdump`'s raw-binary mode is not available in this ROCm build; use
`llvm-mc` instead:

```sh
python3 -c 'd=open("stuck_code.bin","rb").read(); print(" ".join("0x%02x"%b for b in d[384:640]))' > hex.txt
podman run --rm -v $PWD/hex.txt:/hex.txt:ro <rocm-image> \
    /opt/rocm/llvm/bin/llvm-mc -arch=amdgcn -mcpu=gfx803 -disassemble /hex.txt
```

`codedump.py` dumps 1280 bytes starting 512 before the PC, so the stuck
instruction sits at byte offset 512.

## Register decoding notes

Field order comes from `gfx_v8_0_read_wave_data()` in `gfx_v8_0.c` -- read it
from the kernel source you are actually running rather than trusting this
list, it is ASIC-specific:

    0 type, 1 STATUS, 2 PC_LO, 3 PC_HI, 4 EXEC_LO, 5 EXEC_HI, 6 HW_ID,
    7 INST_DW0, 8 INST_DW1, 9 GPR_ALLOC, 10 LDS_ALLOC, 11 TRAPSTS, 12 IB_STS,
    13 TBA_LO, 14 TBA_HI, 15 TMA_LO, 16 TMA_HI, 17 IB_DBG0, 18 M0, 19 MODE

`TRAPSTS` bits 31:29 are `DP_RATE`, not an error -- `0x20000000` alone is
clean. `M0` is only meaningful if the kernel actually initializes it; the blit
copy kernel does not, so garbage there is expected and is not a finding.
