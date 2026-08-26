# Host setup: PCIe ASPM disable

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
