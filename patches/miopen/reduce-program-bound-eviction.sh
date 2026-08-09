#!/bin/sh
# EXPERIMENTAL -- bounds the number of resident dynamic-reduction programs as a
# gfx803 workaround for miopenReduceTensor's intermittent wrong sums. Wired into
# the Dockerfile only while under validation; see reduce-program-bound-eviction.patch
# for the full rationale and the validation gates it must pass before shipping.
#
# Apply reduce-program-bound-eviction.patch via `git apply`, then verify the
# markers actually landed.
set -eu

SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/reduce-program-bound-eviction.patch"
FILE="$SRC/src/reducetensor.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply "$PATCH"

count=$(grep -c 'REDUCE_PROGRAM_BOUND' "$FILE" || true)
# expect the marker in: the env-var decl comment, the NoteReduceProgram calls,
# the EvictReduceProgramsBeyondBound call, and the end-of-dispatch block.
if [ "$count" -lt 4 ]; then
    echo "FATAL: marker not found after git apply reported success (got $count)" >&2
    exit 1
fi
echo "Reduce program-bound eviction workaround applied and verified ($count markers) in $FILE"
