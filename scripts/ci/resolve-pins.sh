# shellcheck shell=bash
# Sourced, never executed. Needs bash for the here-document helper.
#
# Resolve every branch pin to the commit it points at right now, export each one
# as <NAME>_SHA, and export GFX803_PINS, the summary that gets stamped into the
# image.
#
# A branch cloned inside a RUN is invisible to the layer cache, because the cache
# key is the command text and that does not change when AMD pushes to the branch.
# Without a resolved commit, a build can silently reuse a layer from an older tip.
# Failing to resolve is fatal on purpose: an image that cannot say which commit it
# holds is what this guards against.
#
# The ref values come from docker-bake.hcl, so that file stays the only place the
# pins are written down. Every component resolves all of them, so that each image
# records the same complete provenance.
#
# usage: source scripts/ci/resolve-pins.sh

_pins_graph="$(mktemp)"
docker buildx bake --print pins 2>/dev/null | sed -n '/^{/,$p' > "$_pins_graph"

_bake_arg() { # <arg-name>
    python3 - "$_pins_graph" "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    graph = json.load(handle)
for target in graph.get("target", {}).values():
    value = target.get("args", {}).get(sys.argv[2])
    if value:
        print(value)
        break
PY
}

_resolve_commit() { # <url> <ref>
    local url="$1" ref="$2" listing="" sha attempt
    if printf '%s' "$ref" | grep -qE '^[0-9a-f]{40}$'; then
        printf '%s' "$ref"
        return 0
    fi
    # Retry, because a transient github failure must not fail a multi-hour build.
    for attempt in 1 2 3; do
        listing="$(git ls-remote "$url" "refs/tags/$ref^{}" "refs/heads/$ref" "refs/tags/$ref" 2>/dev/null || true)"
        [ -n "$listing" ] && break
        sleep $((attempt * 5))
    done
    # A tag can name a tag object, so prefer the peeled commit when the server
    # offers one.
    sha="$(printf '%s\n' "$listing" | awk -v w="refs/tags/$ref^{}" '$2==w {print $1; exit}')"
    if [ -z "$sha" ]; then
        sha="$(printf '%s\n' "$listing" | awk -v a="refs/heads/$ref" -v b="refs/tags/$ref" '$2==a || $2==b {print $1; exit}')"
    fi
    printf '%s' "$sha"
}

# Fixed order, because GFX803_PINS becomes a build-arg and a shuffled string
# would change the cache key on every run.
GFX803_PINS=""
while read -r _ref_name _sha_name _url; do
    [ -n "$_ref_name" ] || continue
    _ref="$(_bake_arg "$_ref_name")"
    if [ -z "$_ref" ]; then
        echo "FATAL: docker-bake.hcl declares no value for $_ref_name" >&2
        exit 1
    fi
    _sha="$(_resolve_commit "$_url" "$_ref")"
    if [ -z "$_sha" ]; then
        echo "FATAL: could not resolve $_ref_name=\"$_ref\" against $_url" >&2
        exit 1
    fi
    export "$_sha_name=$_sha"
    GFX803_PINS="$GFX803_PINS $_ref_name=$_sha"
    echo "pin: $_ref_name=$_ref -> $_sha"
done <<'PINS'
ROCM_SYSTEMS_REF   ROCM_SYSTEMS_SHA   https://github.com/ROCm/rocm-systems.git
ROCM_LIBRARIES_REF ROCM_LIBRARIES_SHA https://github.com/ROCm/rocm-libraries.git
MIGRAPHX_REF       MIGRAPHX_SHA       https://github.com/ROCm/AMDMIGraphX.git
PYTORCH_REF        PYTORCH_SHA        https://github.com/ROCm/pytorch.git
TORCHVISION_REF    TORCHVISION_SHA    https://github.com/pytorch/vision.git
TORCHAUDIO_REF     TORCHAUDIO_SHA     https://github.com/ROCm/audio.git
ORT_VERSION        ORT_SHA            https://github.com/microsoft/onnxruntime.git
PINS

export GFX803_PINS="${GFX803_PINS# }"
rm -f "$_pins_graph"
