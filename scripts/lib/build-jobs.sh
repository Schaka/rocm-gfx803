# shellcheck shell=sh
# Sourced, never executed.
#
# Every compile stage here caps its own job count instead of letting ninja or
# make default to nproc. An -O3 clang job on this stack needs several GB of RSS,
# so the default parallelism can exceed the build host's RAM. That does not fail
# cleanly: clang segfaults on a different random file each run, or the OOM killer
# takes out something unrelated.
#
# BUILD_PARALLEL_LEVEL=auto sizes the count from MemAvailable at about 4GB per
# job, capped at nproc. Pass an explicit number when the host RAM is known.

resolve_build_jobs() {
    _jobs="${BUILD_PARALLEL_LEVEL:-auto}"
    if [ "$_jobs" = "auto" ]; then
        _jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo)
        _cpu=$(nproc)
        [ "$_jobs" -gt "$_cpu" ] && _jobs=$_cpu
        [ "$_jobs" -lt 1 ] && _jobs=1
    fi
    echo "$_jobs"
}
