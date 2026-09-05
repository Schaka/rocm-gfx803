#!/bin/bash
# Validate a freshly built rocm-gfx803 image on the card.
#
#   tools/imgvalidate.sh <image-tag>
#
# It runs the on-card gate (verify.py), the cross-dispatch coherence probes with a
# control arm that must reproduce the corruption, and the fp16 GEMM/convolution sweep
# for the SGEMM shim's takeover routes. Every arm reaps leftover in-container processes
# first: `timeout` on `podman exec` kills the client and leaves the process running
# inside the container holding its GPU context, and a second GPU tenant invalidates
# everything the probes measure.
#
# The probes are copied out of this repo into /data each run, so the run tests this
# tree's harnesses and not whatever was on the box last week.
set -u
IMG="${1:?image tag required}"
NAME=imgval
D=/data/s6/imgval
PROBES=/data/s6/imgval-probes
REPO="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p "$D"
rm -rf "$PROBES"
mkdir -p "$PROBES"
cp "$REPO/verify.py" "$PROBES/"
cp "$REPO"/tools/tc-staleness/{ms3,ms4,multistream2,conbrace3,xprobe2,pinnedhost,graphprobe}.py "$PROBES/"
cp "$REPO"/tools/correctness-suite/{torch_op_suite,fp16_gemm_sweep}.py "$PROBES/"

cleanup() {
  podman exec "$NAME" bash -lc "pkill -9 -f '[p]ython3' >/dev/null 2>&1; true" 2>/dev/null
  sleep 2
}

# One probe, in the image container, summarized on one line.
arm() { # name, seconds, script, extra env...
  cleanup
  local name=$1 secs=$2 script=$3; shift 3
  timeout "$secs" podman exec "$@" -e TRIALS="${TRIALS_OVERRIDE:-60}" "$NAME" \
      python3 "/data/s6/imgval-probes/$script" > "$D/$name.log" 2>&1
  local rc=$?
  echo "--- $name rc=$rc :: $(grep -ahoE 'MS3.*|MS4.*|MS2.*|STATS.*|PINNEDHOST.*|XREADER.*|GRAPHPROBE.*|CONVPROBE.*|F16SWEEP.*|All checks passed.*|^BAD *:.*|^PASS *:.*|NONFINITE.*' "$D/$name.log" | tail -2 | tr '\n' ' ')"
}

echo "=== image: $IMG"
podman rm -f "$NAME" >/dev/null 2>&1
podman run -d --name "$NAME" --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined -v /data:/data "$IMG" sleep infinity >/dev/null || {
    echo "FATAL: container start failed"; exit 1; }
sleep 10

K=$(podman exec "$NAME" bash -lc "rocminfo 2>/dev/null | grep -c KERNEL_DISPATCH")
echo "KERNEL_DISPATCH agents: $K (want 2)"
[ "$K" = "2" ] || { echo "FATAL: card is not a dispatch agent; reload amdgpu (README box section)."; exit 1; }

echo "--- provenance of the shipped libraries"
podman exec "$NAME" bash -lc '
  L=$(readlink -f /opt/rocm/lib/libamdhip64.so)
  echo "  libamdhip64: $(basename "$L") $(md5sum "$L" | cut -c1-12)"
  for f in CLR_GFX8_TC_INVALIDATE CLR_GFX8_TC_RECORD_FENCE; do
    printf "  %-26s %s\n" "$f" "$(strings "$L" | grep -c "$f")"
  done
  S=/opt/rocm/lib/libgfx803_sgemm_shim.so
  echo "  sgemm shim: $(md5sum "$S" | cut -c1-12)"
  for m in f16-takeover f16-map-nm sb-takeover-no-algo-gate; do
    printf "  %-26s %s\n" "shim marker $m" "$(strings "$S" | grep -c "$m")"
  done
  objcopy -O binary --only-section=.hip_fatbin /opt/rocm/lib/librocsolver.so.0 /tmp/fb 2>/dev/null
  echo "  rocsolver .hip_fatbin: $(stat -c%s /tmp/fb 2>/dev/null) bytes"'

echo "--- coherence and numerics"
arm verify   1800 verify.py
arm ms3      2400 ms3.py
arm ms4      2400 ms4.py
arm ms2      2400 multistream2.py
arm conbrace 1800 conbrace3.py
arm xprobe   1800 xprobe2.py
arm pinned   1800 pinnedhost.py
arm graph    1800 graphprobe.py

echo "--- control: coherence site off (must reproduce)"
cleanup
timeout 1800 podman exec -e TRIALS=60 -e CLR_GFX8_TC_RECORD_FENCE=0 "$NAME" \
    python3 /data/s6/imgval-probes/ms3.py > "$D/ms3off.log" 2>&1
echo "--- ms3off rc=$? :: $(grep -ahoE 'MS3.*' "$D/ms3off.log")"

echo "--- fp16 GEMM and conv sweep, one case per process (a fault kills the context)"
cleanup
pass=0; fail=0
for i in $(seq 0 26); do
  r=$(timeout 200 podman exec "$NAME" python3 /data/s6/imgval-probes/fp16_gemm_sweep.py --case "$i" 2>/dev/null \
      | grep -aoE '\{"i".*' | tail -1)
  case "$r" in
    *'"ok": true'*)  pass=$((pass+1));;
    *)               fail=$((fail+1)); echo "   case $i FAILED: ${r:-<no output>}";;
  esac
done
echo "F16SWEEP pass=$pass fail=$fail (want 27/0)"

echo "--- control: same sweep with the shim's takeover disabled (real rocBLAS agreement)"
cleanup
pass2=0; fail2=0
for i in $(seq 0 26); do
  r=$(timeout 200 podman exec -e GFX803_SGEMM_SHIM_DISABLE=1 "$NAME" \
      python3 /data/s6/imgval-probes/fp16_gemm_sweep.py --case "$i" 2>/dev/null \
      | grep -aoE '\{"i".*' | tail -1)
  case "$r" in
    *'"ok": true'*)  pass2=$((pass2+1));;
    *)               fail2=$((fail2+1));;
  esac
done
echo "F16SWEEP-NOSHIM pass=$pass2 fail=$fail2 (want 27/0)"

echo "--- op suite (long)"
cleanup
timeout 4200 podman exec "$NAME" python3 /data/s6/imgval-probes/torch_op_suite.py > "$D/opsuite.log" 2>&1
echo "opsuite rc=$? :: $(grep -aE '^(CHECKS|BAD|NONFINITE|ERROR|PASS)' "$D/opsuite.log" | tr '\n' ' ')"

echo "--- dmesg: GPU resets or ring timeouts during this run"
dmesg | grep -aicE "resetting wave|ring .* timeout|GPU recovery disabled"
echo IMGVAL_DONE
