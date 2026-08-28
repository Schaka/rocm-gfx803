#!/usr/bin/env bash
# Workaround, not a fix: gfx7/8 has a per-process GPU/KFD wedge (see
# LAST_REMAINING_PROBLEMS.md problem 1) that a small fraction of vLLM launches hit early in
# startup, during the first device-to-device blit-kernel copy (observed:
# HIP loading a compiled kernel module onto the GPU). Once a process hits
# it, every later blit-kernel dispatch in that SAME process fails too --
# confirmed this doesn't just mean the current dispatch, but the whole
# process: signal retry, queue recreation, and queue warm-up (all tried
# and measured against real hardware) recover 0% of the time once a
# process has hit it once. There is nothing left to retry *inside* the
# process. A fresh process, though, has its own independent ~30-35%
# chance of hitting it -- so relaunching the whole process until one
# clears the early danger window reaches the user's target reliability
# (0.65-0.70 pass rate compounds to >99% within a handful of attempts).
#
# Usage: vllm-relaunch-supervisor.sh <full vllm command...>
# Exit code and stdout/stderr behavior match running the wrapped command
# directly, once it's past the danger window -- this script gets out of
# the way at that point and just forwards output and the exit code.
set -uo pipefail

MAX_ATTEMPTS="${SUPERVISOR_MAX_ATTEMPTS:-8}"
DETECT_WINDOW_SECONDS="${SUPERVISOR_DETECT_WINDOW_SECONDS:-180}"
VRAM_DRAIN_TIMEOUT_SECONDS="${SUPERVISOR_VRAM_DRAIN_TIMEOUT_SECONDS:-20}"
VRAM_IDLE_THRESHOLD_BYTES="${SUPERVISOR_VRAM_IDLE_THRESHOLD_BYTES:-524288000}"
FAILURE_MARKER='giving up'

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <vllm command...>" >&2
  exit 2
fi

# kill -9'ing a wedged process doesn't make the driver reclaim its VRAM
# instantly -- relaunching too fast into still-held memory produces an
# unrelated "insufficient free GPU memory" failure that looks like (but
# isn't) another wedge, and isn't retried since it doesn't match
# FAILURE_MARKER. Wait for VRAM to actually drain before trying again.
wait_for_vram_drain() {
  command -v rocm-smi >/dev/null 2>&1 || return 0
  local waited=0
  while [ "$waited" -lt "$VRAM_DRAIN_TIMEOUT_SECONDS" ]; do
    local used
    used=$(rocm-smi --showmeminfo vram 2>/dev/null \
      | grep -oE 'VRAM Total Used Memory \(B\): [0-9]+' \
      | grep -oE '[0-9]+$')
    if [ -n "$used" ] && [ "$used" -lt "$VRAM_IDLE_THRESHOLD_BYTES" ]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  log="$(mktemp /tmp/vllm-relaunch-supervisor.XXXXXX.log)"
  "$@" > "$log" 2>&1 < /dev/null &
  pid=$!

  outcome=""
  elapsed=0
  while [ "$elapsed" -lt "$DETECT_WINDOW_SECONDS" ]; do
    if grep -qF "$FAILURE_MARKER" "$log" 2>/dev/null; then
      outcome="wedged"
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      outcome="exited"
      break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  if [ "$outcome" = "wedged" ]; then
    echo "[supervisor] attempt $attempt/$MAX_ATTEMPTS hit the known GPU wedge -- relaunching." >&2
    cat "$log" >&2
    kill -9 "$pid" 2>/dev/null
    pkill -9 -P "$pid" 2>/dev/null
    # The actual GPU-using process is usually a grandchild (vLLM's
    # multiprocess EngineCore), not reachable via -P on $pid alone.
    pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
    rm -f "$log"
    wait_for_vram_drain
    continue
  fi

  # Past the danger window (or the process already exited on its own for
  # an unrelated reason) -- forward whatever's accumulated so far.
  cat "$log"

  if [ "$outcome" = "exited" ]; then
    rm -f "$log"
    wait "$pid"
    exit $?
  fi

  # Still running past the window: keep streaming live output (tail
  # --pid stops on its own once $pid exits) and forward the final exit
  # code, exactly like running the command directly.
  tail -n 0 -f --pid="$pid" "$log"
  rm -f "$log"
  wait "$pid"
  exit $?
done

echo "[supervisor] gave up after $MAX_ATTEMPTS attempts, every one hit the GPU wedge." >&2
exit 1
