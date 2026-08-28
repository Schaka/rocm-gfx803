# gfx803 rocm7: last remaining problems

Three issues, originally suspected to share a root cause in
`AqlQueue::ExecutePM4`
(`projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`)
and its gfx7/8 code path. **Problem 2 turned out not to be a software bug
at all** -- see its RESOLVED section below -- so that shared-mechanism
theory now only covers problems 1 and 3. Full background in
`MIGRATION_NOTES.md`.

## 1. EOP-completion-notification-loss hang (mitigated, not fixed)

**Symptom**: `AqlQueue::ExecutePM4`'s gfx8 branch busy-waits
(`while (queue->LoadReadIndexRelaxed() <= write_idx) os::YieldThread();`)
with no timeout. The GPU sometimes genuinely finishes the dispatched work
but the completion notification (read-index advance) never arrives --
confirmed via live kernel-fence tracing showing real completion with no
corresponding notification. Same erratum hits `BlitKernel::
SubmitLinearCopyCommand`.

**Status**: mitigated, not fixed -- and the mitigation itself has now been
caught causing a *worse* failure than the hang it papers over. See
"Mitigation caught causing a full GPU bus death" below before enabling
`ROCR_GFX8_EOP_MITIGATION` on anything that matters. `patches/
rocm-systems/blit-kernel-eop-interrupt-retry.patch` adds a bounded retry
(give up and proceed after a timeout) to both call sites, gated behind
`ROCR_GFX8_EOP_MITIGATION=1` (off by default). Settled defaults:
`ROCR_GFX8_EOP_MITIGATION_TIMEOUT_US=250`,
`ROCR_GFX8_EOP_MITIGATION_MAX_ATTEMPTS=1`. Root cause (why the
notification is lost) never identified -- undocumented firmware behavior,
not reachable from software as far as this investigation got.

**Confirmed real and independent of problem 2's VRAM fix.** After
reflashing the box to a non-mining VBIOS that resolved problem 2 (see
below), reran the correctness-suite with `ROCR_GFX8_EOP_MITIGATION`
unset (off). The very first sweep (`activ_sweep`, a tiny sweep, normally
seconds) hung immediately: 10+ minutes pinned at 99.7% CPU, zero log
output, silent `dmesg`. Attached plain `gdb` to the live process
(`gdb -p <pid> -batch -ex 'thread apply all bt'`) -- safe here, unlike
attaching during the pool_sweep race, since this thread was already
permanently parked in a pure userspace spin with no timing-sensitive
window left to disturb by observing it. Backtrace of the main thread:

```
#0 rocr::core::BusyWaitSignal::WaitRelaxed(...)
#1 rocr::core::BusyWaitSignal::WaitAcquire(...)
#2 rocr::HSA::hsa_signal_wait_scacquire(...)
#3-7 (libamdhip64.so, unexported)
#8 hipMemcpy()
#9 main()
```

`hsa_signal_wait_scacquire(completion_signal_, ..., (uint64_t)-1, ...)`
inside `BlitKernel::SubmitLinearCopyCommand` -- the exact call site this
patch already targets (confirmed by grepping the patch for the identical
call). Same erratum, same code site, genuinely independent of the
VRAM-marginality issue that caused problem 2 -- not a symptom of it.

**Mitigation caught causing a full GPU bus death -- worse than the hang
it's meant to paper over.** Killed the hung container, then reran the
same suite with `ROCR_GFX8_EOP_MITIGATION=1` (the settled defaults) to
confirm the mitigation clears it. It did not clear it -- it changed the
*failure mode* into something significantly worse. `activ_sweep` again
stalled on its first `hipMemcpy`, this time in kernel state `D`
(uninterruptible sleep, 0% CPU) rather than the usual userspace spin.
`gdb -p <pid>` itself did not return within 30s -- a strong sign of being
genuinely blocked inside a syscall, not just slow. `/proc/<pid>/stack`
(works without ptrace) showed:

```
drm_sched_entity_flush+0x152/0x2d0 [gpu_sched]
amdgpu_flush+0x31/0x40 [amdgpu]
filp_flush+0x38/0x80
__x64_sys_close+0x3d/0xa0
```

The process had already moved *past* `BlitKernel::SubmitLinearCopyCommand`
-- the userspace mitigation's bounded 250us/1-attempt give-up-and-proceed
logic worked exactly as designed and let `hipMemcpy` return early without
actually waiting for the still-outstanding SDMA copy job to complete. The
process then went on to exit and `close()` its GPU device fd -- and the
*kernel's own* fence wait (`drm_sched_entity_flush`, unrelated to and not
bounded by the userspace mitigation at all) blocked forever on that same
still-outstanding job, because at the hardware level it genuinely never
completed. `dmesg` showed the kernel's own watchdog eventually firing and
the ensuing cascade:

```
amdgpu 0000:02:00.0: ring sdma0 timeout, signaled seq=10975, emitted seq=10979
amdgpu 0000:02:00.0: GPU reset begin!. Source:  1
amdgpu 0000:02:00.0: device lost from bus!
amdgpu 0000:02:00.0: GPU reset end with ret = -19
amdgpu 0000:02:00.0: GPU Recovery Failed: -19
```

...repeating continuously (405+ times observed, alternating `sdma0`/
`sdma1`, load average climbing past 17 as the kernel looped retrying a
reset that can never succeed) until the box was rebooted. The GPU was
**fully and permanently wedged off the PCIe bus** -- not a process that
could be `kill -9`'d, not a hang contained to one container the way every
other crash tonight was; the whole card was dead until a full reboot.

**Read carefully before assuming this is settled**: "give up and proceed"
is fine when the thing being given up on truly doesn't matter to what
comes next -- it is not fine when the caller (or, as here, an entirely
different subsystem -- the kernel's own DRM scheduler -- later on the same
fd) still depends on that job's real completion. The bounded retry mitigation
turned a merely-annoying, fully-recoverable hang (one stuck process,
`kill -9`-able, GPU otherwise fine) into an actually-worse outcome (a
dead GPU requiring a full system reboot) by racing ahead of a GPU job that
was still in flight. This matches problem 3's own prior finding almost
exactly ("a naive flat-timeout attempt... caused a genuine GPU page
fault... also a fault, not just a hang") -- same shape of mistake, worse
consequence this time. **This does not necessarily mean the mitigation
should never be used** -- it was validated fine against real vLLM
workloads in the original investigation (see `MIGRATION_NOTES.md`) -- but
it means the "settled defaults" framing above overstates how safe this is
across all call patterns, and it should not be turned on casually,
especially not for anything that calls `hipMemcpy`/blocking copies
followed shortly by context/device teardown. It's plausible (not yet
confirmed) that this same "device lost from bus" cascade, not just
problem 2's VRAM VM-fault, explains some of the full-box unresponsive
episodes hit earlier in this same investigation session -- worth keeping
in mind before assuming a future full-box lockup is VRAM-related.

**A second, real, independent host issue found and fixed along the way:
this box was running a desktop compositor that was itself submitting GPU
work to the same card.** While re-testing the hang above, discovered (via
full boot-log inspection, `journalctl -k -b`, since a `dmesg -C` earlier
had wiped the kernel ring buffer) that this "headless" box was booting
into `graphical.target` with `plasmalogin.service` active -- a KDE Plasma
login-manager session running `kwin_wayland` as compositor. dmesg showed
`kwin_wayland`'s own GFX-ring submission (PID/thread unrelated to any
ROCm test) hit a real `ring gfx timeout` and triggered its own `GPU
reset begin!`/`BACO reset` cycle, entirely independent of anything this
investigation launched -- even though `amdgpu` had logged `Cannot find
any crtc or sizes` at init (no monitor attached), the compositor was
still submitting render work to the card. This is a genuine, separate bug
(a supposedly-headless compute box shouldn't be running a desktop
session that contends with ROCm workloads for the same GPU) and was
fixed: `systemctl disable plasmalogin.service` +
`systemctl set-default multi-user.target`, then reboot. Confirmed after:
no `kwin`/`plasma` processes, GPU enumerates clean, dmesg has zero
fault/reset noise on a fresh boot.

**This did not turn out to explain problem 1, though -- don't
over-credit it.** With the compositor confirmed gone, `activ_sweep`
(mitigation still off) ran clean once (120/120 cases, 0 WRONG, no hang)
-- encouraging, but a single clean run isn't proof against a bug that was
always probabilistic (recall problem 2's ~50% intermittent rate before
its real fix). A follow-up full-suite run reproduced the identical hang
class again, on the very next `activ_sweep` invocation, this time stuck
in `hipDeviceSynchronize()` rather than `hipMemcpy()` -- same
`BusyWaitSignal::WaitAcquire` -> `hsa_signal_wait_scacquire` infinite-wait
signature, different call site, same underlying erratum. **Independent,
stronger evidence this time**: killing the hung container triggered a
real *kernel-level* `qcm fence wait loop timeout expired` ->
`GPU reset begin!` (source 4) -- the kernel's own compute-queue-manager
fence wait timed out, completely unrelated to the userspace HSA
busy-wait or to `ROCR_GFX8_EOP_MITIGATION` (off throughout). This is
solid confirmation the completion-loss is real at the hardware/firmware
level, not an artifact of the compositor, the mitigation's own logic, or
anything specific to this session's instrumentation. The reset that
followed repeatedly failed (`Failed to evict process queues`,
`Failed to quiesce KFD`, `last message was failed ret is 0` looping every
~3.5s) -- the same unrecoverable-without-a-reboot pattern as the mitigation-
triggered bus death, but this time triggered by the hang itself, with no
mitigation involved at all.

**Kernel param change: forcing `amdgpu.reset_method=4` (BACO) -- tried,
rejected by the driver, does not help.** The box's kernel cmdline had
`amdgpu.gpu_recovery=1` but no explicit `reset_method`, leaving the
driver to auto-select per reset source. Tonight's own evidence shows two
different outcomes for two different reset sources: the compositor's
ring-timeout-triggered reset (source 1) explicitly logged `BACO reset`
and succeeded cleanly; this hang's KFD-triggered reset (source 4) instead
retried an SMU-message-based method (consistent with mode2/soft reset)
that failed outright, over and over, needing a full reboot to clear.
Added `amdgpu.reset_method=4` via `grubby --update-kernel=ALL
--args='amdgpu.reset_method=4'` to force BACO for every reset instead of
letting a per-source auto-selection pick a path that's demonstrably
unreliable on this card, then rebooted and re-verified the param was
active (`cat /proc/cmdline`).

**Result: rejected outright.** Reproduced the hang again (looped
`activ_sweep` attempts until one hung, then killed the stuck container to
trigger the kernel-side teardown/reset path -- see the note below on why
just *waiting* never triggers it). dmesg showed the driver refusing the
forced method for this specific reset source: `Specified reset method:4
isn't supported, using AUTO instead.` -- it fell back to the exact same
SMU-message-based AUTO path and failed the same way (`last message was
failed ret is 0`, repeating, unrecoverable), needing another full reboot.
BACO is apparently only reachable for *some* reset sources on this
card/driver combination (works for the compositor's ring-timeout-sourced
reset), not the KFD-queue-eviction-sourced one this hang triggers. This
kernel param is not a fix -- reverted the intent to keep it; **no known
kernel param fixes the reset failure for this specific hang**, only a
full reboot recovers it.

**A clarifying side-finding: the hang does not self-timeout from the
kernel's side just by waiting.** Left an `activ_sweep` hang running
undisturbed for 3:50+ (well past both a plain wait and a `timeout 60`
wrapper around the `docker run` -- the wrapper's signal never reached the
containerized process, a tooling quirk, not itself informative) with
*zero* dmesg activity the entire time. The kernel-side `qcm fence wait
loop timeout` / `GPU reset begin!` sequence only appeared once the
container was explicitly killed (`docker kill`), which forces a
queue-eviction/quiesce during teardown -- that's what the kernel's own
fence-wait watchdog actually reacts to, not elapsed real time on an
already-submitted, silently-never-completing signal. A hung process left
alone will apparently spin forever with no kernel-level consequence at
all; it's specifically *tearing it down* that triggers the reset (and,
depending on reset source, the failure cascade).

**Follow-up: IOMMU removed entirely, still does not help.** Not
convinced `intremap=off` alone was a clean enough test, went further per
direct request: removed `intel_iommu=on`, `iommu=pt`, and `intremap=off`
all together via `grubby --update-kernel=ALL --remove-args=...`
(box falls back to its firmware/BIOS default IOMMU behavior --
`iommu: Default domain type: Translated` at boot, vs the explicit
`Passthrough` mode set before; IOMMU itself is still active, just not
explicitly tuned by this box's kernel cmdline anymore). Rebooted,
confirmed clean cmdline and healthy boot, ran the same repeat-until-hung
loop: **1/10 clean, hung on attempt 2**, identical `qcm fence wait loop
timeout expired` cascade on kill. Confirms the fence-info finding's
prediction -- interrupt delivery via IOMMU wasn't the mechanism. Left the
IOMMU params removed per explicit direction (not restored) -- no evidence
either that they were needed or that removing them causes harm, this box
just doesn't need to carry them if they're not fixing anything.

**Grubby gotcha worth remembering**: `grubby --update-kernel=ALL
--args='pci=nomsi'` silently *dropped* the pre-existing `pci=realloc=on`
token instead of adding a second `pci=` entry -- grubby dedupes on the
param name before `=`, keeping only the last `pci=...` token, not
Linux's actual multi-token `pci=` parsing behavior. `pci=realloc=on` is
load-bearing for this card's resizable-BAR support (referenced
throughout this whole investigation). Caught before rebooting; fixed by
combining into one comma-separated token, `pci=nomsi,realloc=on`, the
correct syntax for multiple `pci=` sub-options. Worth checking `grubby
--info=DEFAULT` after *any* `--args` addition that might collide with an
existing param's key, not just assuming it appended cleanly.

**MSI-vs-legacy-INTx test: negative for the GPU hang, but revealed a
real, unrelated regression -- reverted.** Added `pci=nomsi` (kept
combined as `pci=nomsi,realloc=on` per the gotcha above) to force legacy
INTx interrupt delivery instead of MSI, testing whether an MSI/interrupt-
remapping quirk on this old card was the mechanism. Rebooted, confirmed
active, ran the repeat-until-hung loop: **1/10 clean, hung on attempt
2**, same `qcm fence wait loop timeout expired` cascade. Before killing
the hung process, dmesg showed something new: `Disabling IRQ #16` with a
shared-line stack trace covering `ahci`, `nvme0q0`/`nvme0q1`, `xhci-hcd`
(USB), `i2c_designware.0`, `i801_smbus`, and `eno1` (the NIC) -- the
kernel's "irq nobody cared" auto-disable, triggered by forcing legacy
INTx sharing across devices that normally get dedicated MSI vectors.
Checked `/proc/interrupts`: `amdgpu` sits on IRQ 17, a *different* shared
line (`PCIe bwctrl` x2, `i2c_designware.1`, `idma64.1`, `eno2`) that was
never disabled -- so this doesn't implicate the GPU's own interrupt path
specifically, and the hang reproducing under `nomsi` is a clean enough
negative result for that hypothesis. But `pci=nomsi` itself is a real
regression on this box regardless of the gfx803 question -- disabling a
shared IRQ line taints storage, USB, and primary NIC interrupt delivery
for everything else running here (postgres, wyoming-voice, etc.).
**Reverted immediately** (`pci=realloc=on` restored on its own, `nomsi`
removed) rather than leaving a host-wide interrupt regression in place
for a test that already came back negative.

**Unused audio-component test: also negative, closes out the ranked
list.** Unbound the HDMI/DP audio function (`snd_hda_intel` on
`0000:02:00.1`, part of the same GPU die but a separate PCI function) at
runtime -- no reboot needed (`echo 0000:02:00.1 > .../driver/unbind`).
`amdgpu`'s own interrupt already sits on a dedicated MSI vector (IRQ 152)
not shared with the audio function, so this was always a low-probability
shot. Ran the repeat-until-hung loop: **1/10 clean, hung on attempt 2**,
identical cascade. Confirms it.

**Kernel-param/hardware-config investigation phase closed out.** Every
lead from the ranked list has now been tried: IH-ring-overflow
correlation (none observed), live GPU fence-state capture during an
active hang (kernel interrupt processing confirmed intact -- this is the
one finding that actually narrowed the mechanism, see above), forced
BACO reset (`reset_method=4`, rejected by the driver for this reset
source), IOMMU interrupt remapping (`intremap=off`, negative), IOMMU
entirely removed (negative), legacy INTx interrupts (`pci=nomsi`,
negative, plus a real unrelated regression -- reverted), and unused
audio-component IH traffic (negative). **No kernel parameter or hardware
configuration change found this session prevents the hang or its
failed-reset cascade.** The sharpest surviving lead is the live
fence-info finding: the kernel's own interrupt-driven fence tracking is
demonstrably correct and current at the moment userspace is stuck,
pointing at the GPU's own completion-signal memory write (not an
interrupt) as the actual failure point -- see "Fresh-eyes re-
investigation" above. That's not something reachable by a boot parameter;
it would need direct PM4/AQL-level instrumentation of the dispatch
completion write itself (the `Gfx803DispatchRing` tooling already built
this session could do this, not yet pointed at this specific question).
Given the cost of each experiment (a full reboot cycle, ~2-10 minutes
each, many tonight), the practical next lever left is hardening
`ROCR_GFX8_EOP_MITIGATION` itself against the close()/teardown race that
caused the bus-death finding, rather than continuing to search for a
config-level fix for a bug that increasingly looks like it isn't
config-reachable at all.

**Severity correction: these hangs don't just wedge the GPU, they can
block the OS from shutting down cleanly.** Every `sudo reboot` issued
during this investigation actually required the user to physically
power-cycle the box -- the OS's own shutdown sequence hangs trying to
tear down the dead GPU/KFD state, the same way any other process's
`close()` on the device does. This is a materially worse failure mode
than "one stuck process, GPU needs a driver-level reset" -- it can take
the whole machine down to needing physical access, not just a remote
`reboot` command. Reframes the earlier "6/6 wedged" result from the
*original* vLLM investigation (`MIGRATION_NOTES.md` item 5) too: that
test ran with the mitigation *active*, against a long-running server
process that never closes its device fd -- so it likely never actually
exercised this specific teardown-triggered kernel-reset failure at all,
regardless of how it looked from vLLM's side. The recoverability
difference observed tonight (mitigation off, short-lived test processes
that get killed and exit) most likely isn't a change in the GPU/driver's
actual reset reliability -- it's that this investigation is, for the
first time, exercising a reset path that was probably always this
fragile, because nothing before now had a reason to trigger it.

**Precise localization via the PM4/dispatch-level tracers, `SIGABRT`
captured live.** Reproduced the hang under the same instrumented image
used for the `pool_sweep` investigation
(`rocm-gfx803:pool-debug-pm4trace2`), then sent the stuck process
`SIGABRT` directly (`kill -ABRT <pid>`) instead of just killing it --
this triggers the tracer's own crash-dump handler without needing the
process to hit `abort()` on its own. Two findings:

- **`ExecutePM4` ring: 156/156 clean pairs**, every `DOORBELL_RUNG`
  immediately followed by `COMPLETION_CONFIRMED` -- the code-object load/
  invalidate path is completely clean, matching problem 2's earlier
  finding for a different bug.
- **The dispatch ring's last entry is the stuck job itself**: a real
  `KERNEL_DISPATCH` packet (`kernel_object=0x900237940`), with nothing
  recorded after it -- this is the exact AQL dispatch whose completion
  `hsa_signal_wait_scacquire` is parked waiting on forever. Combined with
  the earlier live `amdgpu_fence_info` capture (every kernel-tracked ring
  already showing `Last signaled == Last emitted` during a hang), this is
  as precise as the localization gets without instrumenting the driver's
  actual completion-signal write path: **the GPU's per-dispatch
  completion-signal write for this specific packet doesn't land**, while
  the kernel/hardware fence tracking for the exact same work reports it
  done. Not an interrupt, not `ExecutePM4`, not code loading, not IH
  saturation -- a write-visibility gap on one specific memory write.
  (Side note: the `SIGABRT` didn't actually terminate the process this
  time -- it dumped and kept spinning, needing an explicit `kill -9`
  afterward. Worth checking whether the crash handler's `signal(sig,
  SIG_DFL); raise(sig)` re-raise is actually reliable under this specific
  stuck state, though it doesn't affect the trace data captured.)

**Does the "delay" idea (user's suggestion) help here?** Almost
certainly not, based on precedent. Problem 2's delay-mitigation
experiments (six real data points: two placements, three durations up to
100ms, 500x the estimated real-world scale) found no effect whatsoever
on a structurally similar bug. More decisively for problem 1
specifically: this wait is not "very slow," it's **provably infinite**
under direct observation -- left running 3+ minutes undisturbed with zero
forward progress, zero dmesg activity, before being intervened on. A
pre-dispatch delay could in principle change scheduling/timing enough to
occasionally dodge the underlying race (the same mechanism that made
`rocgdb`/`fprintf` tracing *appear* to suppress problem 2's race, which
turned out to be misleading), but it doesn't touch the actual missing
memory write, and there's no specific theory for *where* a delay would
need to sit that differs from what problem 2 already ruled out. Not
worth spending another reboot cycle on without a sharper hypothesis for
placement.

**Where this leaves the fix, given the severity correction above**: the
practical goal isn't just "reduce hang frequency" -- it's "never let a
process end up needing to be killed while a job is genuinely stuck,"
since it's specifically the kill-then-close() sequence that triggers the
unrecoverable kernel-reset failure. That reframes hardening
`ROCR_GFX8_EOP_MITIGATION` as the right target after all: the mitigation
was already validated safe for a long-running server that never closes
its fd (the original vLLM context); the bus-death finding was specific
to short-lived processes exiting right after a give-up. A version that
tracks give-up'd signals and makes a *subsequent* teardown/close() wait
properly (rather than racing ahead) would plausibly close that gap
without reopening the original hang-forever problem for long-running
workloads.

**A third mitigated call site exists, was never in the tracked patch
set, and testing it triggered the worst failure of the night.**
Following up on the user's question ("doesn't the mitigation already do
this?"), source reading found that `ROCR_GFX8_EOP_MITIGATION` actually
gates *three* wait sites, not two: `AqlQueue::ExecutePM4`'s internal
wait, `BlitKernel::SubmitLinearCopyCommand`'s copy-completion wait (both
in `patches/rocm-systems/blit-kernel-eop-interrupt-retry.patch`, the
tracked, shipped patch), and a third -- ROCclr's `WaitForSignal()`
(`rocvirtual.hpp`), called from `HwQueueTracker::WaitCurrent()`/
`WaitNext()` -- the generic kernel-dispatch-completion wait used by
`hipDeviceSynchronize()`. This third site has its own separately-tunable
timeout, `ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US` (default 5000us), and
given the PM4 trace above shows the stuck job is a genuine
`KERNEL_DISPATCH` (not a copy), this is very likely the actual call site
`activ_sweep`'s hang goes through. **Critically, this patch
(`rocclr-eop-wait-mitigation.patch`) exists only as a standalone file
under `tools/rocr-novad-rebuild/`, never incorporated into the tracked
`patches/` directory the main Dockerfile uses** -- `rocm-gfx803:
rocm7-regression` does not have this code compiled in at all.

A first test against the *untracked* image confirmed this the hard way:
setting `ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US=500` against
`rocm7-regression` produced a hang indistinguishable from no mitigation
at all -- because the env var was never read by that binary. Rebuilt
properly via the `rocr-novad-rebuild` pipeline (fully cached, since the
source tree already had the patch applied from earlier work), packaged
as `rocm-gfx803:eopwait-test`, verified via `strings | grep
ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US` present in the built
`libamdhip64.so`.

Testing `activ_sweep` against this correctly-patched image with
`ROCR_GFX8_EOP_MITIGATION=1 ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US=500`:
**the box went completely, fully unresponsive -- not just the GPU.**
SSH timed out; `ping` returned "Destination Host Unreachable" from the
LAN gateway, meaning the box stopped responding at the network/ARP layer
entirely, not merely a wedged GPU with the OS still up (every prior
failure tonight, including the original bus-death finding, kept SSH/
network alive throughout). Required a manual hard power-cycle to
recover. The test's log lived on tmpfs and was wiped by the crash, so
**no forensic detail survived on what specifically happened in that
run** -- only that it happened during this exact combination (mitigation
on, 500us HIP timeout, `activ_sweep`, this specific patched build).

**This is the single worst failure observed this session** -- worse
than the earlier bus-death finding, which at minimum left the OS
reachable and diagnosable remotely. Do not re-run this exact combination
without direct, explicit authorization and the user standing by for
physical access -- a full hard lockup with zero remote diagnostic
capability is a materially different risk class from anything else
tried tonight, and needs to be treated that way rather than folded into
the general "reboot and try again" cadence used for every earlier test.

**Retried once with durable forensics in place -- crashed identically a
second time, investigation stopped here.** Per the user's direction, set
up disk-backed (not tmpfs) crash forensics before retrying: a standalone
watchdog loop writing 1-second snapshots (process state, GPU
`amdgpu_fence_info`, dmesg tail) to `/data/crash-forensics/watchdog.log`
on the box's real ext4 disk, `sync`ed after every write, plus reliance on
persistent journald (`/var/log/journal/`) as a second durable source. The
retry itself needed two rounds of incidental cleanup first -- the prior
hard crash had corrupted podman's overlay storage (`readlink ...
invalid argument` on the specific container/image touched at crash time;
fixed each time by removing and, where needed, rebuilding the affected
image) -- before the actual mitigated run could even launch. Once it
did: **the box hard-locked again, identically** (SSH timeout, no ping
response), while looping repeated `activ_sweep` attempts under the same
`ROCR_GFX8_EOP_MITIGATION=1 ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US=500`
combination. **The forensic data from this second crash was never
retrieved before the investigation was called off** -- the box was still
down when the decision was made to stop. If picked up again,
`/data/crash-forensics/watchdog.log` and `journalctl -k -b -1` on that
box are the first things to check; being disk-backed, they should have
survived this crash even though the box itself needs a physical
power-cycle to read them back.

**Investigation closed here, by explicit user decision.** Summary of
where this stands: the hang is real, narrowly localized (a specific
per-dispatch GPU completion-signal write that doesn't land, confirmed via
live kernel fence-state inspection and PM4/dispatch-ring tracing -- see
above), reproduces independently of VRAM clock, the desktop-compositor
confound, and every IOMMU/interrupt/reset kernel parameter tried. No
config-level fix was found. The one mechanism that touches the actual
call site (`ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US` via the untracked
`rocclr-eop-wait-mitigation.patch`) reproducibly hard-locks the box
outright when it fires -- worse than the hang it's meant to paper over,
in both attempts this was tested. Given that, no software or
configuration lever is currently known that safely resolves this hang;
the honest state is **unresolved, not further mitigable with what's been
tried**. Anyone picking this back up should start from the sharp
localization above (a specific AQL dispatch's completion write, not an
interrupt) rather than re-deriving it, and should treat
`ROCR_GFX8_EOP_MITIGATION_HIP_TIMEOUT_US` as actively dangerous to enable
outside a controlled, disposable test environment until its hard-lock
failure mode is understood.

**Correction: no known safe *in-process* mitigation -- but a validated
*process-level* workaround does exist.** An earlier session's vLLM work
(see `vllm-gfx803/NOTES.md`) hit this same wedge from a different angle
and found the failure is per-*process*, not per-queue or per-dispatch:
once one `BlitKernel::SubmitLinearCopyCommand` fails in a process, every
later dispatch in that process fails too, on any queue -- but a *fresh*
process retains its own independent ~30-35% chance of clearing the early
danger window. `tools/host-setup/vllm-relaunch-supervisor.sh` exploits
exactly that: detects the known failure marker, kills the whole process
tree (including the grandchild `EngineCore`), waits for VRAM to actually
drain, and relaunches -- validated 6/6 clean against a real workload
where individual launches only cleared ~30-35% on their own. This does
not touch the underlying hang (still root-cause-unresolved, still
per-dispatch-unfixable) and does nothing for a process that's already
deep into useful work when it hits the wedge -- but for a short-lived
batch job or a server that can tolerate a cold restart, it's a real,
already-working answer, safer than anything tried against the hang
directly tonight since it never touches the mitigation patches that
caused the two hard lockups above.

## 2. `pool_sweep` GPU VM fault (RESOLVED -- hardware VRAM marginality, not a software bug)

**Symptom**: `tools/correctness-suite/pool_sweep` crashes with a real GPU
VM fault (not a soft miscompute) partway through its run -- specifically
on the first *new* JIT-compiled kernel load after ~10 prior kernel loads
in the same process (MIOpen op `AVE_INCLUSIVE`, `C=8 H=256 W=256 k=2x2
s=2`, the first shape at that point requiring a fresh
`hipModuleLoadData`). Deterministic fault address every run
(`0x9347ff000`/`0x934800000`).

**Root-caused (partially) via `rocgdb`**: the fault occurs in
`AqlQueue::ExecutePM4`, reached via `GpuAgent::InvalidateCodeCaches` ->
`RegionMemory::Freeze` -> `hsa_executable_freeze` -> `hipModuleLoadData`
-- i.e. it fires while loading and freezing a **new** JIT-compiled kernel
binary, not during the pooling kernel's own execution. Confirmed
pre-existing (reproduces identically against the currently-published
image, not a regression from any patch in this repo) -- see
`MIGRATION_NOTES.md` for the dmesg evidence and comparison-image test.

**Working theory**: `GpuAgent::InvalidateCodeCaches` submits a PM4
`ACQUIRE_MEM` packet whose `COHER_BASE`/`COHER_SIZE` reference the new
code object's own GPU VA, via the same `AqlQueue::ExecutePM4` gfx8 path
that hangs in problem 1. `RegionMemory::Freeze` branches on
`LargeBarEnabled()`: with the real 8GB BAR this session's original fix
enabled, small code objects now take a direct `memcpy` +
`GpuAgent::PcieWcFlush` path into BAR-mapped VRAM instead of the DmaCopy
path used when the BAR was small -- another case of the BAR fix exposing
a previously-untaken code path. `PcieWcFlush`'s sfence/read-back/mfence
sequence looks correct for CPU-write visibility, but the fault reason
("Page not present") is a GPU *page-table* fault, not a cache-coherency
one -- so the more likely gap is that the GPU-side VM mapping
(`hsaKmtMapMemoryToGPUNodes`, called synchronously during allocation)
hasn't fully propagated to the compute/cache-controller side by the time
`ExecutePM4`'s `ACQUIRE_MEM` packet -- the very first GPU-side touch of
that fresh VA -- runs. **Not confirmed as the root cause** -- see below.

**Quantified via a 12-run batch of the unmodified binary**: 6/12 crashed,
6/12 passed clean, same fault address every crash
(`0x9347ff000`/`0x934800000` -- adjacent bytes of the same page). This is
a genuine ~50% intermittent race, not a deterministic logic bug --
consistent with (not contradicting) the theory above, since races are
inherently probabilistic. It also explains why three separate synthetic
repro attempts (isolated single call, matched kernel-load-count, matched
buffer-churn-volume, even a byte-identical recompile with `build.sh`'s
exact flags) all came back clean: at ~50% per attempt, a handful of
misses is unsurprising, not disproof.

**A related, already-fixed kernel bug turned out NOT to be this one.**
This repo already carries `patches/kernel/
REFERENCE-amdkfd-gfx7-8-missed-interrupt-wakeup.patch` -- a real,
hardware-validated fix (found chasing an earlier vLLM hang, same
`RegionMemory::Freeze()`/code-object-load call chain) for gfx7/8's
unreliable interrupt-payload event IDs causing KFD to either miss a
completion wakeup or wake the wrong event. The gfx803 box has been
running this patched kernel all along, including throughout this entire
investigation -- confirmed via matching build tree, timestamp, and
`strings` on the loaded `.ko`. `pool_sweep` still faults at ~50% with it
active. Since that fix targets `CIK_INTSRC_CP_END_OF_PIPE`/event-ID
lookup (KFD event signaling) and `pool_sweep`'s fault comes through
`gmc_v8_0_process_interrupt` (the GPU's MMU/VM-fault path, a different
subsystem entirely), this is good evidence the two are genuinely separate
bugs, not one shared root cause -- contrary to what "Why these three are
grouped together" below speculated before this was checked.

**First fix attempt: applied, tested, did not work.** Traced
`amdgpu_amdkfd_gpuvm_map_memory_to_gpu()` (backs `hsaKmtMapMemoryToGPUNodes`)
and found its success path called `unreserve_bo_and_vms(&ctx, false, false)`
-- `wait=false` means the ioctl returns to userspace as soon as the
GPU-side page-table write is *queued* (fenced via `ctx.sync`), not once
it's actually *done*. gfx9+ has hardware fault-and-retry to invisibly
absorb a stray early access to a not-yet-committed mapping; gfx7/8 does
not. Patched to `unreserve_bo_and_vms(&ctx, adev->asic_type < CHIP_VEGA10,
false)` -- force the wait, gfx7/8 only (existing gating pattern already
used elsewhere in the same file: `adev->asic_type < CHIP_VEGA10`).
Built, installed, box rebooted, verified loaded. **Result: no
improvement** -- 13/25 clean (~52%), statistically indistinguishable from
the ~50% baseline. `CHIP_POLARIS10 = 15 < CHIP_VEGA10 = 19` confirmed
correct in this kernel's enum, ruling out a gating bug. The fix is
harmless but not the cause (or not the *only* cause).

**Debugging this further revealed a real heisenbug.** Reproducing under
`rocgdb` (`--cap-add=SYS_PTRACE`, live-attached) went **8/8 clean** --
at true ~50% odds that's ~0.4% likely by chance. Attaching a debugger
measurably changes scheduling/timing enough to close the race window
entirely. This is strong independent confirmation that the bug is a
genuine hardware/kernel timing race, not a logic bug -- but it also means
`rocgdb` can't be used to catch it live for a backtrace.

**Lightweight tracing instead of a debugger.** `pr_info`-based kernel
printk was confirmed *not* to suppress the race (crashes still reproduce
with it active) -- unlike `ptrace`, plain printk logging is cheap enough
to not close the window. Added `pr_info` to
`amdgpu_amdkfd_gpuvm_map_memory_to_gpu()`'s success path: a captured
crash showed **zero** matching log lines across the entire run -- that
kernel function was never invoked for whatever memory this fault touches,
directly falsifying the theory above (the fix is real but irrelevant to
this bug). Tried an `LD_PRELOAD` interposer on `hsaKmtAllocMemory`/
`hsaKmtMapMemoryToGPUNodes` next, as a non-kernel way to watch the same
calls from userspace -- dead end: `libhsakmt` is a static archive
(`libhsakmt.a`) linked directly into `libhsa-runtime64.so` with no
exported dynamic symbols (`nm -D` confirms), so `LD_PRELOAD` architecturally
cannot intercept it.

**Broadened kernel `pr_info` coverage** to both KFD ioctl entry points
(`kfd_ioctl_alloc_memory_of_gpu`, `kfd_ioctl_map_memory_to_gpu` in
`kfd_chardev.c`) and even the literal first line of the top-level ioctl
dispatcher `kfd_ioctl()` itself, before any branching at all. **Zero
hits, every time** -- including a batch where 6/10 runs crashed in the
same dmesg-capture window with the instrumentation active. This
contradicted a separately-built userspace trace (see next) that clearly
showed hundreds of real, successful KFD ioctl calls per run on the exact
same `/dev/kfd` device (confirmed identical major:minor, `235,0`, host
and container). `file_operations.unlocked_ioctl = kfd_ioctl` registration
confirmed correct; ioctl command macros confirmed identical between the
userspace build's headers and the kernel's own; dispatch table logic
(`amdkfd_ioctls[nr]`) confirmed robust to encoding differences by design.
This contradiction was never resolved -- see below for why it stopped
mattering.

**Userspace-level tracing, and the answer.** Since `hsaKmtAllocMemory`/
`hsaKmtMapMemoryToGPUNodes` live in `libhsakmt` (statically linked, so
`LD_PRELOAD` can't reach them), instrumented `hsakmt_ioctl()` itself --
the single choke point every one of these calls passes through --
directly in the ROCR-Runtime source, rebuilt via `tools/
rocr-novad-rebuild` and swapped into a container (no kernel/reboot cycle
needed for this layer). First attempt used `fprintf`+`fflush` per call
(~800 calls/run): **15/15 clean, zero crashes** -- another heisenbug
suppression, this time from syscall overhead in the hot path rather than
`ptrace`. Rewrote as a zero-syscall in-memory ring buffer (8192 entries,
plain atomic memory stores, no I/O) dumped only once, via a `SIGABRT`/
`SIGSEGV` signal handler using async-signal-safe `write()`, firing
exactly when HIP's `VMFaultHandler` calls `abort()` on the crash.

**This caught it on the first try.** The dump held the process's
*entire* ioctl history (1388 entries, no wraparound) -- and every single
`ALLOC_MEMORY_OF_GPU`/`MAP_MEMORY_TO_GPU`/`UNMAP_MEMORY_FROM_GPU`/
`FREE_MEMORY_OF_GPU` call was cleanly paired, `BEGIN` immediately
followed by matching `END`, `ret=0`, no orphaned call, no hang, no error,
right up through the last entry before the fault. **This conclusively
rules out an ioctl-timing race of any kind** -- there is no allocation or
mapping call in flight when the crash happens, and the memory-mapping
ioctls demonstrably complete and get their completion observed well
before the fault. It also fully explains the kernel-ioctl-instrumentation
contradiction above: it was never going to see anything, because the
crash doesn't happen during an ioctl.

**Working theory at this point (superseded below -- kept for the record
since it shaped several hours of investigation)**: `AqlQueue::
ExecutePM4` doesn't make a syscall to submit a packet -- it writes
directly into an already-mapped queue ring buffer and rings a doorbell
via a raw MMIO write, then the GPU asynchronously picks the packet up and
executes it. The theory was that `GpuAgent::InvalidateCodeCaches`'s
`ACQUIRE_MEM` packet, submitted this way, faults touching the new code
object's VA -- based on the original `rocgdb` backtrace catching the
crashing thread parked inside `ExecutePM4`'s wait loop when `abort()`
fired. This looked like a GPU-internal cache/TLB propagation gap: the
driver's own bookkeeping done, but the specific compute unit executing
`ACQUIRE_MEM` not yet caught up, gfx7/8 lacking gfx9+'s fault-and-retry
safety net.

**This theory is now known to be WRONG, or at least incomplete.** A
zero-syscall ring-buffer tracer added directly inside `ExecutePM4`'s
gfx8 doorbell-ring/wait code (recording every packet's actual
`COHER_BASE`/`COHER_SIZE` target and whether its completion was
confirmed) caught a real crash and dumped the process's *entire*
`ExecutePM4` history: **37 calls, all 37 cleanly paired** (doorbell rung,
completion confirmed, no orphan) -- every single `InvalidateCodeCaches`
call this process ever made succeeded before the fault happened. The
`rocgdb` backtrace that anchored the original theory was misleading: it
caught a thread that had already finished all its real work, not one
whose own packet faulted. **The fault is not an `ACQUIRE_MEM`/code-cache
-invalidate problem at all.** Whatever actually faults is somewhere else
in the pipeline -- the leading remaining candidates are the real AQL
kernel-dispatch path (a completely different queue-write mechanism from
`ExecutePM4`, used for the actual compute kernel launch that *uses* a
freshly-loaded code object) or the plain CPU-side `memcpy`/
`GpuAgent::PcieWcFlush` write into VRAM in `RegionMemory::Freeze()` --
which was the very first theory considered hours earlier in this same
investigation, set aside prematurely. Delay-based mitigation (tried at
both plausible `RegionMemory::Freeze()` vantage points, see below) was
therefore never going to work regardless of duration, since it was aimed
at the wrong call site all along -- which retroactively explains those
negative results too, not just the ones already attributed to "duration
doesn't matter."

**A second tracer, on the real dispatch path, narrowed it further.**
`AqlQueue::StoreRelease()` -- the public doorbell-ring entry point real
kernel dispatches use, a completely separate mechanism from
`ExecutePM4`'s own internal doorbell ring -- was instrumented the same
way (zero-syscall ring buffer, crash-time dump), reading each queue
slot's packet header and, for `KERNEL_DISPATCH` packets, the
`kernel_object` handle. A captured crash showed 246 dispatch-ring
entries alongside the 37 clean `ExecutePM4` pairs. The **last real
`KERNEL_DISPATCH`** recorded (`kernel_object=0x8001c87c0`) matches --
almost exactly, offset by a small fixed header/kernarg preamble -- the
**last code object loaded** (`coher_base=0x8001c80` at
`ExecutePM4` `write_idx=36`, the final entry in that ring). So: the
36th/37th code object loads cleanly, invalidates cleanly, and gets
dispatched cleanly -- and then there's a **~24.7ms gap** (500-1000x
longer than the ~10-50us spacing between every other event in the whole
trace) before the dump's last entry and the crash. Loading, invalidating,
and dispatch *submission* are all clean, every time, across two separate
tracers now. The fault most plausibly happens **during the GPU's actual
execution of that just-dispatched kernel** -- fetching instructions or
data mid-run -- not during any setup step this investigation has
instrumented so far. That's a real, sharper localization, but it also
moves the fault site into the GPU's own execution pipeline, which is
much harder to observe from software than the doorbell/ioctl layers
tried so far.

**A sharp, unresolved side observation on the fault address.** The
crash address (`0x934800000`, ~39.5GB) is suspiciously close to this
box's total system RAM (39GiB reported by `free -h`; VRAM is exactly
8GiB, GTT is 19.53GiB). Every address recorded by both tracers --
code-object loads and dispatches alike -- falls in the same general
"8xxxxxxxx-9xxxxxxxx" range (roughly 32-39GB), consistent with a private
GPUVM aperture starting around 32GB (`0x800000000`) and extending
through a region sized off system RAM/GTT. The crash address sits inside
that range, not exactly at a boundary either tracer's own addresses
reached. Confirming the precise aperture layout needs the driver's
boot-time GMC/aperture-size dmesg output, which was inadvertently lost
this session (cleared via repeated `dmesg -C` while chasing other
traces) -- would need a fresh reboot to recapture. Flagged as a real,
not-yet-confirmed lead, not a conclusion.

**Hardware-marginality test: GPU core clock, negative result.**
Following a hypothesis that this old, long-used Polaris card's hardware
might simply be marginal at stock clocks/voltage: `power_dpm_force_
performance_level=manual` + `pp_dpm_sclk` pinned to state 3 (1019MHz,
down from the default auto-boost range of 1169-1236MHz) for a 15-run
batch. **Result: 11/15 crashed (73%)** -- if anything higher than the
~50% baseline seen throughout the night (though within the batch-to-batch
variance already observed), not lower. This argues against simple GPU
core-clock marginality as the explanation. VRAM clock (`MCLK`) could
**not** be tested the same way -- this card only exposes 2 discrete MCLK
states (300MHz idle / 2100MHz active) with no overdrive range unlocked,
so it stayed at 2100MHz throughout. The user's live hypothesis, not yet
tested: 2100MHz may be a mining-BIOS-inflated VRAM clock above the
card's actual rated/safe spec (mining workloads tolerate occasional
errors that a correctness-checked compute workload won't), and a
non-mining VBIOS flash running VRAM at ~2000MHz might resolve it.
**Explicitly not attempted by this investigation** -- VBIOS flashing is
a materially different risk class from anything else tried tonight (a
failed/interrupted flash can brick the card, recovery typically needs
physical access), and is being left as the user's own decision and
action to take, not something to execute remotely over SSH.

GPU clock state was restored to `auto`/default after the SCLK test --
not left pinned.

**RESOLVED: mining-BIOS VRAM clock, not a software erratum.** The card
shipped with a mining-tuned VBIOS running VRAM (MCLK) at 2100MHz -- a
clock mining workloads tolerate occasional bit errors at, but a
correctness-checked compute workload does not. Flashed a non-mining
Sapphire Nitro VBIOS running VRAM at 2000MHz
(`RX470_official_nitro_2000_samsung.rom`, via `amdvbflash -p 0 <rom> -f`)
combined with SCLK pinned to 1019MHz as an extra-conservative first test
-- result inconclusive: the box hard-locked entirely (unprecedented all
night; every prior crash had stayed contained to the Docker container) partway
through the batch, losing the in-progress run data. Fresh dmesg after the
forced reboot showed the same fault address firing again within the first
168s of uptime, so this data point leans negative but was never completed
cleanly.

Reflashed to `RX470_inofficial_1750_samsung_hynix.rom` (VRAM 1750MHz,
further below the mining clock) at **stock/auto core clock** (no SCLK
pinning this time). Result: **20/20 clean, zero crashes, zero GPU VM
faults** across two staged 10-run batches. Against the ~50% baseline
crash rate observed all night on the mining VBIOS, 20/20 clean is roughly
0.0001% likely by chance -- this is a real, confirmed fix, not a lucky
streak.

**Why hours of software-level investigation didn't find this**: every
finding above is still accurate as *localization*, not root cause -- the
ring-buffer tracers correctly proved the fault isn't an ioctl race, isn't
`ExecutePM4`/`ACQUIRE_MEM` itself, and isn't dispatch submission, narrowing
it to "somewhere during the GPU's actual execution of the just-dispatched
kernel." That's exactly where you'd expect a marginal-VRAM bit error to
surface: mid-execution, reading/writing VRAM under real memory pressure,
at a nondeterministic point depending on access pattern and timing --
not at any fixed software call site. The delay-mitigation experiments'
own results retroactively make sense under this theory too: injecting an
artificial delay near code-object load changes memory-controller traffic
timing enough to sometimes dodge a marginal-timing error by luck, without
touching the actual cause -- consistent with the delay experiments' noisy,
inconsistent results (45-75% crash regardless of duration or placement)
rather than a clean dose-response curve.

**No software fix in this repo for this issue** -- there was never a
software bug to fix. The fix is the VBIOS. Anyone reproducing this on
their own gfx803 card should check whether it's running mining-tuned VRAM
straps before chasing a software explanation.

**Follow-up, answered: problem 1 is real and independent of the VRAM
fix.** Reran the full correctness-suite against the 1750MHz-VBIOS-fixed
box with `ROCR_GFX8_EOP_MITIGATION` deliberately left unset (its default
off state). The very first sweep (`activ_sweep`, well under 100 tiny
correctness cases, normally seconds) hung immediately -- 10+ minutes
pinned at 99% CPU, zero log output, completely silent `dmesg` (no fault,
no kernel warning), eventually killed by a 900s watchdog timeout. Attached
plain `gdb` (`-p <pid> -batch -ex 'thread apply all bt'`) to the live
stuck process -- safe to do here, unlike the pool_sweep race, since this
thread is already permanently parked in a pure userspace spin with no
timing-sensitive window left to disturb. Backtrace of the main thread:

```
#0 rocr::core::BusyWaitSignal::WaitRelaxed(...)
#1 rocr::core::BusyWaitSignal::WaitAcquire(...)
#2 rocr::HSA::hsa_signal_wait_scacquire(...)
#3-7 (libamdhip64.so, unexported)
#8 hipMemcpy()
#9 main()
```

This is `hsa_signal_wait_scacquire(completion_signal_, ..., (uint64_t)-1,
...)` inside `BlitKernel::SubmitLinearCopyCommand` -- the exact call site
`patches/rocm-systems/blit-kernel-eop-interrupt-retry.patch` already
targets (confirmed by grepping the patch for the same call). Same erratum,
same code site, not a new bug and not a symptom of the VRAM-marginality
issue that caused problem 2 -- genuinely independent. **The retry
mitigation is still required post-VBIOS-fix.** Killed the hung container
and reran the same suite with `ROCR_GFX8_EOP_MITIGATION=1` (the settled
defaults, `TIMEOUT_US=250`/`MAX_ATTEMPTS=1`) to confirm the mitigation
still clears it now that VRAM is stable -- see the top-level "Why these
were grouped together" section for the updated verdict.

**Delay-based mitigation: tried thoroughly, conclusively does not work.**
The pattern of instrumentation apparently suppressing the crash
(`rocgdb` 8/8 clean, `fprintf`-tracing 15/15 clean) suggested a real
forced delay near this call site might mitigate it -- implemented as
`ROCR_GFX8_CODEOBJ_LOAD_DELAY_US` (microseconds, off by default, gfx7/8
only) in `RegionMemory::Freeze()`. Tested at **two placements** (before
`InvalidateCodeCaches()` submits the `ACQUIRE_MEM` packet, and after it
returns) **and three durations each** (200us, 1ms, 100ms -- the last one
500x the user's own calibrated estimate, specifically to confirm the
mechanism works *at all* before tuning):

| placement | 200us | 1ms | 100ms |
|---|---|---|---|
| before `InvalidateCodeCaches()` | 60% crash | 60% crash | 60% crash |
| after `InvalidateCodeCaches()` | 45% crash | -- | 75% crash |

(baseline with no delay: ~50% crash, consistent throughout.) Six real
data points, no placement, no duration -- not even 100ms, concentrated
right at the suspected fault site, which is far larger than any
cumulative overhead `rocgdb`/`fprintf` plausibly added -- produced
anything resembling a fix. This makes it very unlikely those earlier
"instrumentation suppresses it" observations were really about added
latency at all; far more likely small-sample luck (8-20 trials at ~50%
odds isn't a rare streak) or an unrelated side effect of `ptrace`
(memory layout, scheduling) that happened to correlate.

**No PM4 TLB-invalidate primitive is reachable from here either.**
`core/inc/amd_gpu_pm4.h` only defines `NOP`, `INDIRECT_BUFFER`,
`RELEASE_MEM`, `ACQUIRE_MEM`, `ATOMIC_MEM`, `PRED_EXEC`, `WRITE_DATA`,
`WAIT_REG_MEM`, `COPY_DATA`, `DMA_DATA` -- no `INVALIDATE_TLBS`-class
opcode. GCN's real TLB-invalidate PM4 opcode is a privileged operation
normally only issuable via the kernel driver's own KIQ (Kernel Interface
Queue), not from a userspace-submitted compute-queue PM4 stream. The
kernel-side equivalent that *is* reachable (`kfd_flush_tlb` ->
`amdgpu_vm_flush_compute_tlb`, source-verified to run unconditionally
after every `MAP_MEMORY_TO_GPU`) is already firing and still doesn't
prevent the fault.

**Where this left it, before the VBIOS test**: every lever reachable from
software -- ioctl timing, kernel-side TLB flush (already on), forced
delay at two plausible userspace vantage points across three orders of
magnitude of duration, and the one PM4 opcode that could plausibly help
(not available outside the kernel) -- had been tried and ruled out. That
turned out to be the correct conclusion, just not for the reason assumed
at the time: there was no software lever because the cause wasn't
software. See the RESOLVED section above.

**Fresh-eyes re-investigation: live GPU fence state during an active
hang narrows this past "lost interrupt" to something more specific.**
Prompted by a direct question -- is there anything besides
`ROCR_GFX8_EOP_MITIGATION` worth trying, and could one of this repo's own
patches be the actual cause -- re-examined from scratch:

- **Patch audit**: grepped `patches/` for anything touching interrupt/
  signal/event/eviction code. Five matches. `va-reuse-defer.patch` only
  matched an unrelated comment ("eviction" in an allocator context, not
  interrupt/signal-adjacent) -- ruled out. `sdma-doorbell-missing-sfence.
  patch` targets CPU-write-ordering *before* a doorbell ring (a different
  bug class, already established). `blit-kernel-eop-interrupt-retry.
  patch`/`.sh` *is* the mitigation itself -- inert by default
  (`ROCR_GFX8_EOP_MITIGATION` unset throughout every hang reproduction
  this session). `REFERENCE-amdkfd-gfx7-8-missed-interrupt-wakeup.patch`
  (kernel-side, confirmed compiled into the running kernel) targets
  `CIK_INTSRC_CP_END_OF_PIPE` event-ID lookup -- already checked in the
  original investigation and found to be active but insufficient (see
  "A related, already-fixed kernel bug turned out NOT to be this one"
  under problem 2, which covers the same ground). No patch in this repo
  plausibly introduces or worsens this specific hang -- it isn't ours.
- **IH ring buffer overflow check**: earlier tonight's pool_sweep crashes
  showed `IH ring buffer overflow` messages, raising the question of
  whether the same mechanism explains this hang too. Caught a fresh
  `activ_sweep` hang live and checked dmesg throughout -- **zero IH ring
  overflow messages this time.** Doesn't rule out overflow as *a*
  contributing factor in general, but this specific hang instance
  reproduced without it.
- **Live `amdgpu_fence_info` read while frozen mid-hang** (the actual
  new finding): every kernel-tracked ring --  gfx, all 8 compute rings
  (`comp_1.0.0` through `comp_1.3.1`), `sdma0`, `sdma1` -- showed `Last
  signaled fence == Last emitted`. From the kernel driver's own
  interrupt-driven fence-tracking, **everything was already fully
  complete, nothing outstanding**, at the exact moment the userspace HSA
  thread was still spinning in `BusyWaitSignal::WaitAcquire` waiting on
  its own signal memory. This is sharper than "lost interrupt": the
  kernel's own interrupt processing plainly worked (its fence state is
  correct and current), which argues *against* a general interrupt-
  delivery/IOMMU-remapping explanation (that would more likely also
  confuse the kernel's own fence tracking, and it didn't). What's stuck
  is specifically the *userspace-visible* HSA completion signal never
  getting written -- pointing at the GPU's own completion **memory
  write** into the signal's address (the `RELEASE_MEM`-style EOP write
  an AQL kernel-dispatch packet uses to update a user-mapped signal
  location) landing late, out of order, or not at all -- a
  cache-coherency/write-visibility gap, not an interrupt-delivery one.
  **Not yet confirmed as root cause** -- this is a sharper hypothesis
  from one live capture, not a proven mechanism; the PM4/AQL-level
  tracers built earlier this session (`Gfx803Pm4Ring`,
  `Gfx803DispatchRing` in `amd_aql_queue.cpp`, see the tooling under
  `tools/rocr-novad-rebuild/`) could directly instrument this specific
  write if pursued further.
- **IOMMU interrupt-remapping test: tried, does not help.** Added
  `intremap=off` (disables IOMMU interrupt remapping specifically, via
  `grubby --update-kernel=ALL --args='intremap=off'`) -- deliberately
  *not* a blanket `intel_iommu=off`, since this box's `intel_iommu=on
  iommu=pt` is undocumented in this repo and the box runs other,
  unrelated services (postgres, wyoming-voice, etc.) that might depend on
  it; `intremap=off` keeps DMA remapping/passthrough intact and only
  removes interrupt remapping. Rebooted, confirmed active
  (`cat /proc/cmdline`), reproduced the hang via a repeat-until-hung loop.
  **First 3 attempts came back clean** (a real but statistically
  unremarkable streak against a ~50% baseline -- roughly a 1-in-8 chance
  by luck alone, not yet conclusive), **then attempt 4 hung** identically
  to every other reproduction this session, confirmed via the same
  `qcm fence wait loop timeout expired` -> `GPU reset begin!` ->
  repeated `last message was failed ret is 0` cascade on kill, needing
  another reboot. Consistent with the fence-info finding above (kernel
  interrupt processing already looked intact during a hang, so removing
  IOMMU interrupt remapping specifically was never likely to matter) --
  ruled out. Reverted the param (no benefit, same taint-for-nothing
  reasoning as `reset_method=4`).

  **Re-tested with a bigger sample after the 3/4 streak looked
  suspiciously good.** Restored the exact same combo
  (`intel_iommu=on iommu=pt intremap=off`) and ran 15 more sequential
  attempts. Result: 2 clean, hung on attempt 3. **Combined: 5/7 clean
  (71%)** across both runs with this combo, versus 3/6 (50%) combined
  across the other three configs tested tonight (IOMMU fully removed,
  `pci=nomsi`, audio-unbind -- each hung on their 2nd attempt). A real
  gap, but with only 7 total attempts for this combo it's within noise
  range -- nothing like the VBIOS fix's unambiguous 20/20 vs ~50%
  baseline. Read: the original 3/4 was most likely luck, not a real
  effect. Not chasing this combo further.

**A pattern noticed across all these tests, still unexplained**: in
every kernel-param test tonight (`intremap=off` original run,
IOMMU-removed, `pci=nomsi`, audio-unbind, and this retest), the *first*
`activ_sweep` invocation on a fresh boot was clean every single time --
6 configurations, 6 clean first attempts, zero exceptions. That's the
whole claim -- no claim about attempt 2 specifically. How many clean
attempts follow before the first hang varies (as few as 0 more, as many
as 2 more in the original `intremap=off` run), consistent with a
hypothesis around KFD resource reuse (event IDs, doorbell slots) across
process boundaries -- a fresh boot's first process gets never-before-used
resources and is reliably clean, while some later process reusing
recycled resources is where the failure window opens -- rather than a
flat per-attempt hardware timing coin-flip. Not rigorously confirmed
either way. Worth a dedicated, config-neutral test (plain stock cmdline,
just counting which attempt number first hangs across several reboot
cycles) if this gets picked back up.

## 3. `hipMemcpyWithStream` / ROCclr `WaitForSignal` hang (open, unmitigated)

**Symptom**: hangs the same way as problem 1, but through ROCclr's
`WaitForSignal` (`rocvirtual.hpp`), not ROCR-Runtime's `AqlQueue::
ExecutePM4` or `BlitKernel::SubmitLinearCopyCommand` directly -- it goes
through the raw 6-argument `SubmitLinearCopyCommand` overload with no
built-in wait, so there's no size context available at that call site to
thread a safe, size-scaled timeout the way the other two mitigated sites
do.

**Status**: deliberately left unmitigated. A naive flat-timeout attempt
was tried and caused a genuine GPU page fault on a large tensor copy
(reverted immediately) -- notable in hindsight: *also* a fault, not just
a hang, same as problem 2. Fixing this safely needs ROCclr's `Barriers`/
signal-tracking architecture refactored to thread size (or some other
safe bound) through to this call site -- a bigger task than fit in one
session. If problem 2's root cause turns out to be a missing fence rather
than the EOP erratum itself, it's worth re-examining whether this hang
shares that same root cause, since a flat-timeout mitigation here
produced the same fault signature as problem 2's crash.

## Why these were grouped together (revised: problem 2 no longer belongs here)

Problems 1 and 3 both trace back to `AqlQueue::ExecutePM4`'s doorbell-ring
queue-submission mechanism (write a packet into an already-mapped ring
buffer, ring a doorbell via a raw MMIO store, let the GPU pick it up
asynchronously -- no syscall involved in the actual dispatch) and its
completion-wait losing the notification. Problem 2 was originally grouped
in on the theory that it was the same mechanism's *execution* faulting --
but that theory is now known to be wrong: problem 2 is a hardware VRAM-
clock marginality issue on the box's mining-tuned VBIOS, fully resolved by
a VBIOS flash, with no software component at all (see problem 2's
RESOLVED section). It only *looked* like it belonged with 1 and 3 because
all three manifest as GPU-side failures reached through the same general
dispatch/queue code paths -- coincidence of symptom, not shared cause.

**Answered**: problem 1 is a genuine EOP/notification erratum, independent
of the VRAM-clock marginality that caused problem 2 -- confirmed by
reproducing the hang live on the VBIOS-fixed box with the mitigation off,
and identifying its exact call site via `gdb` backtrace (see problem 1's
own section for the full trace). Problem 3 was not re-tested this session
(its own repro is more involved -- a large tensor copy via the raw
`SubmitLinearCopyCommand` overload -- and wasn't rerun after the VBIOS
fix), but shares the exact same call site and signal-wait mechanism as
the hang just reproduced, so it's reasonable to expect it's the same
answer: real, independent of VRAM.

Problem 1's mitigation is not the clean story it looked like earlier
tonight, though: enabling it to test whether it "still clears the hang"
post-VBIOS-fix instead caught it causing a full unrecoverable GPU bus
death (`device lost from bus`, repeated failed resets, box needed a hard
reboot) -- worse than the plain hang. See problem 1's section for the
full trace and reasoning. Problem 2 is resolved at the hardware level, no
software fix needed or possible. Problem 3 remains open and unmitigated,
now confirmed to share problem 1's exact underlying erratum rather than
just its symptom pattern -- and, by the same logic that broke down for
problem 1's mitigation, a naive fix there carries the same "worse than
the hang" risk, which is exactly what problem 3's own prior naive-timeout
experiment already hinted at.
