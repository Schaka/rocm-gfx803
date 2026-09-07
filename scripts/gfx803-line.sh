#!/bin/sh
# Line provenance for the gfx803 component images.
#
# A stage inherits another stage's /opt/rocm either through a named build
# context or through COPY --from. Neither one can say which ROCm line the
# artifact belongs to. Intermediate tags name the line today (:gfx803-rocm10),
# but images published before that carried a bare :gfx803 that an older line of
# this repo used as well. Such an image is consumed silently: it assembles, it
# imports, and it misbehaves only on real hardware. This script puts an
# assertion on that boundary.
#
# An inherited tree with no marker predates this scheme. That case is reported
# and accepted, because a refusal would break every build that reuses an already
# published component. Set GFX803_LINE_STRICT=1 to make it fatal too. A marker
# that names a different line is always fatal, and that is the case this exists
# for.
#
# usage: gfx803-line.sh stamp  <rocm-dir> <line> <stage> [rev] [pins]
#        gfx803-line.sh verify <rocm-dir> <line>
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
        echo "stage=${4:-unrecorded}"
        echo "rev=${5:-unrecorded}"
        echo "pins=${6:-unrecorded}"
    } > "$marker"
    echo "gfx803-line: stamped $marker (line=$want stage=${4:-unrecorded} rev=${5:-unrecorded})"
    ;;
verify)
    if [ ! -f "$marker" ]; then
        if [ "${GFX803_LINE_STRICT:-0}" = "1" ]; then
            echo "FATAL: inherited $dir carries no .gfx803-line marker (strict mode)." >&2
            exit 1
        fi
        echo "gfx803-line: WARNING: inherited $dir carries no .gfx803-line marker," \
             "so its ROCm line is assumed, not checked."
        exit 0
    fi
    have="$(sed -n 's/^line=//p' "$marker" | head -1)"
    if [ "$have" != "$want" ]; then
        echo "FATAL: inherited $dir belongs to ROCm line '$have', this build is '$want'." >&2
        sed 's/^/         /' "$marker" >&2
        echo "       Rebuild that component instead of reusing its published image." >&2
        exit 1
    fi
    echo "gfx803-line: verified line=$want ($(sed -n 's/^rev=//p' "$marker" | head -1))"
    ;;
*)
    echo "usage: $0 {stamp|verify} <rocm-dir> <line> [stage] [rev] [pins]" >&2
    exit 2
    ;;
esac
