# shellcheck shell=bash
# Sourced, never executed.
#
# `docker buildx bake --push` dies outright on a 429 from ghcr.io. BuildKit's
# lazy blob pull, for the base image and for the registry cache, does not retry
# on its own, and pulling from a shared repository credential under load is a
# short burst against ghcr.io's rate limiter rather than a quota. The 429 body's
# retry-after is under a second, so a few backoff retries clear it.
#
# A runner that is out of disk space will still be out of disk space five
# retries and forty minutes later -- nothing here frees any, and BuildKit was
# seen re-running (not serving from cache) the same multi-minute COPY on every
# attempt, so a retry loop on this failure only burns CI time to fail the same
# way each time. Recognized once and failed fast instead. If the runner is
# genuinely this tight on space, the fix belongs in docker-bake.hcl -- give the
# component a trimmed wheels-only image, so final copies less than it pulls --
# not in a longer retry loop here.
bake_push_with_retry() {
    _attempt=1
    _max_attempts=5
    _delay=5
    _log="$(mktemp)"
    while true; do
        docker buildx bake --push "$@" 2>&1 | tee "$_log"
        _status="${PIPESTATUS[0]}"
        if [ "$_status" -eq 0 ]; then
            rm -f "$_log"
            return 0
        fi
        if grep -qi "no space left on device" "$_log"; then
            echo "::error::bake --push $* failed on disk space, not retrying -- the final stage pulls more image than it copies; give the component a wheels-only image or free space on the runner" >&2
            rm -f "$_log"
            return 1
        fi
        if [ "$_attempt" -ge "$_max_attempts" ]; then
            rm -f "$_log"
            return 1
        fi
        echo "::warning::bake --push $* failed (attempt $_attempt/$_max_attempts), retrying in ${_delay}s" >&2
        sleep "$_delay"
        _attempt=$((_attempt + 1))
        _delay=$((_delay * 2))
    done
}
