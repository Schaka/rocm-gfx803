#!/bin/sh
# Line provenance for the gfx803 component images.
#
# A stage takes another stage's output either through `FROM ${X_IMAGE}`, where a
# tag name picks a published image, or through `COPY --from=x-export /opt/rocm`.
# Neither one can say which ROCm line an artifact belongs to. The intermediate tags
# now name the line (`:gfx803-rocm10`), but images published before that carried the
# unsuffixed `:gfx803` that an older line of this repo also used, and such an image
# was consumed silently: it assembles, it imports, and it misbehaves only on real
# hardware. This script puts an assertion on that boundary.
#
# stamp  <rocm-dir> <line>   record which line this stage's /opt/rocm belongs to
# verify <rocm-dir> <line>   fail if what we inherited says otherwise
#
# An inherited tree with no marker predates this scheme. That case is reported and
# accepted, because refusing it would break every build that reuses an already
# published component. Set GFX803_LINE_STRICT=1 to make it fatal too. A marker that
# names a different line is always fatal, and that is the case this exists for.
#
# usage: gfx803-line.sh {stamp|verify} <rocm-dir> <line>
set -eu

mode="$1"
dir="$2"
want="$3"
marker="$dir/.gfx803-line"

case "$mode" in
stamp)
    mkdir -p "$dir"
    {
        echo "line=$want"
        echo "rev=${GFX803_SOURCE_REV:-unrecorded}"
        echo "pins=${GFX803_PINS:-unrecorded}"
        echo "stage=${GFX803_STAGE:-unrecorded}"
    } > "$marker"
    echo "gfx803-line: stamped $marker (line=$want rev=${GFX803_SOURCE_REV:-unrecorded})"
    ;;
verify)
    if [ ! -f "$marker" ]; then
        if [ "${GFX803_LINE_STRICT:-0}" = "1" ]; then
            echo "FATAL: inherited $dir carries no .gfx803-line marker (strict mode)." >&2
            exit 1
        fi
        echo "gfx803-line: WARNING: inherited $dir carries no .gfx803-line marker." \
             "it predates line provenance, so its ROCm line is being assumed, not checked."
        exit 0
    fi
    have="$(sed -n 's/^line=//p' "$marker" | head -1)"
    if [ "$have" != "$want" ]; then
        echo "FATAL: inherited $dir belongs to ROCm line '$have', this Dockerfile builds '$want'." >&2
        sed 's/^/         /' "$marker" >&2
        echo "       Rebuild the component from this Dockerfile instead of reusing its published" >&2
        echo "       image (the *_IMAGE build-args select which image a stage inherits)." >&2
        exit 1
    fi
    echo "gfx803-line: verified line=$want ($(sed -n 's/^rev=//p' "$marker" | head -1))"
    ;;
*)
    echo "usage: $0 {stamp|verify} <rocm-dir> <line>" >&2
    exit 2
    ;;
esac
