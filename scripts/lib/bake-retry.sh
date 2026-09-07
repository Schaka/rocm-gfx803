# shellcheck shell=bash
# Sourced, never executed.
#
# `docker buildx bake --push` dies outright on a 429 from ghcr.io. BuildKit's
# lazy blob pull, for the base image and for the registry cache, does not retry
# on its own, and pulling from a shared repository credential under load is a
# short burst against ghcr.io's rate limiter rather than a quota. The 429 body's
# retry-after is under a second, so a few backoff retries clear it.
bake_push_with_retry() {
    _attempt=1
    _max_attempts=5
    _delay=5
    while true; do
        if docker buildx bake --push "$@"; then
            return 0
        fi
        if [ "$_attempt" -ge "$_max_attempts" ]; then
            return 1
        fi
        echo "::warning::bake --push $* failed (attempt $_attempt/$_max_attempts), retrying in ${_delay}s" >&2
        sleep "$_delay"
        _attempt=$((_attempt + 1))
        _delay=$((_delay * 2))
    done
}
