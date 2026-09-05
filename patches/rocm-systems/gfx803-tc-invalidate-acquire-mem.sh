#!/bin/sh
# Apply gfx803-tc-invalidate-acquire-mem.patch. That file gives the reason and the
# change, and this driver makes sure that the apply worked.
#
# On gfx7 and gfx8 the CP ignores the AQL acquire and release scope bits, so ROCclr
# must publish one PM4 ACQUIRE_MEM ring slot (TC invalidate plus writeback) in three
# places: ahead of every kernel dispatch, so a kernel is not served stale shader-TC
# lines; inside releaseGpuMemoryFence(), so a copy engine cannot read GPU memory whose
# dirty lines are still in the caches; and ahead of an event-record barrier, so a copy
# engine waiting on another queue's event cannot read that queue's unwritten-back
# lines. The slot publishes no completion signal, which is what keeps it out of the
# marker path's signal bookkeeping.
#
# Run this after d2h-staged-copy, d2h-null-dsthost, and
# pinned-release-system-scope. They edit the same file in other functions.
set -eu

SRC="${1:-/rocm-systems-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-tc-invalidate-acquire-mem.patch"
CPP="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.cpp"
HPP="$SRC/projects/clr/rocclr/device/rocm/rocvirtual.hpp"
FLAGS="$SRC/projects/clr/rocclr/utils/flags.hpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$CPP" ] && [ -f "$HPP" ] && [ -f "$FLAGS" ] || {
    echo "FATAL: rocclr sources missing under $SRC" >&2; exit 1; }

if grep -q 'insertTcFence' "$CPP"; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

# The two returning-false call sites must be present: the dispatch-side one and the
# pre-copy-fence one.
[ "$(grep -c 'if (!insertTcFence()) return false;' "$CPP")" -eq 2 ] || {
    echo "FATAL: expected 2 insertTcFence() call sites in $CPP, found $(
        grep -c 'if (!insertTcFence()) return false;' "$CPP")" >&2; exit 1; }
# The third site is the event-record one. It is written differently on purpose, so a
# marker that fails to publish is reported instead of silently continuing.
grep -q 'CLR_GFX8_TC_RECORD_FENCE && vcmd.coalesceEvent() != 0' "$CPP" || {
    echo "FATAL: event-record TC writeback site missing from $CPP" >&2; exit 1; }
grep -q 'CLR_GFX8_TC_RECORD_FENCE' "$FLAGS" || {
    echo "FATAL: CLR_GFX8_TC_RECORD_FENCE flag missing from $FLAGS" >&2; exit 1; }
grep -q 'bool VirtualGPU::insertTcFence()' "$CPP" || {
    echo "FATAL: insertTcFence() definition missing from $CPP after git apply" >&2; exit 1; }
grep -q 'bool insertTcFence();' "$HPP" || {
    echo "FATAL: insertTcFence() declaration missing from $HPP" >&2; exit 1; }
grep -q 'CLR_GFX8_TC_INVALIDATE' "$FLAGS" || {
    echo "FATAL: CLR_GFX8_TC_INVALIDATE flag missing from $FLAGS" >&2; exit 1; }
echo "gfx803 TC-fence patch applied and verified (dispatch + pre-copy fence + event record)"
