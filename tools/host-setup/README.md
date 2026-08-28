# Host setup: PCIe ASPM disable, native ROCm+vLLM install

## PCIe ASPM disable

See the main `README.md`'s "Host BIOS setting: keep PCIe ASPM off" section
for why this matters. Short version: on at least one gfx803 host (a
Supermicro board whose ACPI FADT declares `the system doesn't support PCIe
ASPM` and never hands OS-level ASPM control to Linux), enabled ASPM caused
rare, extremely hard-to-diagnose multi-minute stalls under GPU load.
`pcie_aspm=off` on the kernel cmdline does **not** fix this on a board like
this -- it only disables Linux's own ASPM subsystem, it does not clear
already-firmware-programmed Link Control register bits on a device Linux
was never given control of. The only fix that actually worked was writing
the PCIe Link Control register directly with `setpci`, on **both** ends of
the link (the GPU endpoint and the CPU-side root port -- fixing only one
end still reproduced the stall).

`gfx803-aspm-disable.service` re-applies this on every boot (needed because
the firmware re-programs the same bits at each boot). Install it:

```sh
sudo cp gfx803-aspm-disable.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gfx803-aspm-disable.service
```

The two `setpci` targets (`00:01.1` and `02:00.0`) and the register values
(`0c40`, `0040`) are specific to the box this was found on -- find yours
before reusing this unit elsewhere:

```sh
# Identify your GPU's BDF and confirm it's actually the card you think it is
lspci -d 1002:67df          # 67df = Ellesmere/RX470; use your card's device ID

# Find the root port it's attached to (single level for most desktop/server boards)
lspci -tv

# Check current ASPM state on both ends
lspci -s <gpu-bdf> -vvv | grep -iE 'LnkCtl|LnkSta'
lspci -s <root-port-bdf> -vvv | grep -iE 'LnkCtl|LnkSta'

# Read the current Link Control register (offset 0x10 in the PCIe cap),
# clear bits 0-1 (the ASPM Control field), write it back
setpci -s <bdf> CAP_EXP+10.w
setpci -s <bdf> CAP_EXP+10.w=<value-with-bits-0-1-cleared>
```

Verify with `lspci -s <bdf> -vvv | grep LnkCtl` -- should read
`ASPM Disabled` on both ends after the service runs.

## Native ROCm+vLLM install (skip the container for real-hardware debugging)

`native-rocm-vllm-setup.sh` copies a working ROCm+PyTorch+vLLM stack from
an already-working container straight onto the box's own host filesystem
at `/opt/rocm`, `/opt/venv`, `/opt/uv-python`. Once installed, run
anything (gdb, ftrace/kprobes, py-spy, the actual benchmark) directly on
the host with no `docker exec` wrapper and no PID-namespace translation
between what a debugger sees and what `/sys/kernel/debug/...` reports --
this box is dedicated gfx803 test hardware, so there's no isolation
benefit to running the real workload in a container here, only cost.

```sh
sudo bash native-rocm-vllm-setup.sh          # copies from ct-old by default
sudo bash native-rocm-vllm-setup.sh <name>   # or another container name
```

Writes a `gfx803-env.sh` to the invoking user's home directory -- `source`
it before running anything from `/opt/venv`:

```sh
source ~/gfx803-env.sh
rocminfo | grep gfx803
python3 -m vllm.entrypoints.cli.main bench latency --model ... 
```

The script's own header documents the three container-relative-path traps
it works around (alternatives symlinks, a uv-managed Python interpreter
symlinked into `/root` which a non-root user can never reach, and a
system `libopenblas` dependency the host's package manager doesn't
provide under the same name by default) -- read it before reusing this on
a different box, since exact paths/package names may differ.
