#!/bin/sh
LOG=/data/hangcap4.log
: > $LOG
for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "=== attempt $i $(date +%T) ===" >> $LOG
    timeout --signal=KILL 150 podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
      -v /data/pool-suite-bin3:/suite:ro \
      -v /data/ringfix-libs2/libhsa-runtime64.so.1.21.0:/opt/rocm/lib/libhsa-runtime64.so.1.21.0:ro \
      -v /data/ringfix-libs2/libamdhip64.so.7.14.60850-0000000:/opt/rocm/lib/libamdhip64.so.7.14.60850-0000000:ro \
      docker.io/library/rocm-gfx803:rocm7-regression /suite/activ_sweep > /tmp/k$i.out 2>&1
    rc=$?
    cases=$(grep -c "cos=\|maxabsdiff=" /tmp/k$i.out)
    echo "attempt $i rc=$rc cases=$cases" >> $LOG
    sync
    [ $rc -eq 0 ] && continue
    [ $rc -ne 137 ] && { echo "NOT A HANG rc=$rc" >> $LOG; break; }
    P=$(pgrep -x activ_sweep | head -1)
    echo "### HUNG attempt $i after $cases cases pid=$P ###" >> $LOG
    echo "-- waves (first 4) --" >> $LOG
    echo user | sudo -S python3 /tmp/wavescan2.py 2>/dev/null | head -6 >> $LOG
    W=$(echo user | sudo -S python3 /tmp/wavescan2.py 2>/dev/null | grep -m1 "^SE" | sed "s/SE\([0-9]*\) CU\([0-9]*\) SIMD\([0-9]*\) W\([0-9]*\).*/\1 \2 \3 \4/")
    echo "-- gprs of wave [$W] --" >> $LOG
    echo user | sudo -S python3 /tmp/gprdump.py $W >> $LOG 2>&1
    echo "-- process VA maps (kfd/gpu ranges) --" >> $LOG
    echo user | sudo -S cat /proc/$P/maps 2>/dev/null | head -40 >> $LOG
    sync
    break
done
echo "CAPTURE DONE" >> $LOG
