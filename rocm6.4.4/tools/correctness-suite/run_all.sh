#!/bin/sh
# Run the correctness suite (built via build.sh), saving full output and a
# machine-readable summary for later analysis. Safe to run detached
# overnight (nohup/background) -- everything lands in RESULTS_DIR, nothing
# depends on an attached terminal.
#
# Usage:
#   sh run_all.sh [bin_dir] [results_dir]
#   ONLY="reduce_sweep conv:ConvBinWinogradRxS" sh run_all.sh [bin_dir] [results_dir]
#
# ONLY (space-separated, optional): run just these instead of the whole
# suite -- e.g. while iterating on a fix for one specific bug, you don't
# want to pay for a 20-minute full run on every retry. Op-level sweeps are
# named after their binary (e.g. "reduce_sweep"); conv solver sweeps are
# named "conv:<SolverName>" matching a file in shapes/.
#
# Must run where the GPU is actually visible (i.e. with
# --device=/dev/kfd --device=/dev/dri --group-add video if in a container).
#
# Output layout in results_dir (default: <suite>/results/<timestamp>/):
#   summary.csv       -- sweep_name,status,cases_tested,wrong_count (one row per sweep)
#   summary.txt        -- human-readable pass/fail table + totals
#   <sweep_name>.log    -- full stdout+stderr of that sweep, one file per sweep
set -eu

SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BIN="${1:-$SELF_DIR/bin}"
RESULTS_DIR="${2:-$SELF_DIR/results/$(date +%Y%m%d-%H%M%S)}"
SHAPES="$SELF_DIR/shapes"
ONLY="${ONLY:-}"

[ -d "$BIN" ] || { echo "FATAL: $BIN doesn't exist -- run build.sh first" >&2; exit 1; }

# Returns 0 (true) if $1 should run: ONLY unset/empty means "run everything";
# otherwise $1 must appear as a whole word in ONLY.
should_run() {
    [ -z "$ONLY" ] && return 0
    case " $ONLY " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/summary.csv"
TXT="$RESULTS_DIR/summary.txt"
echo "sweep_name,status,cases_tested,wrong_count" > "$CSV"
: > "$TXT"

fail_count=0
run_count=0

# Pulls "SUMMARY: N cases/shapes tested, M WRONG" out of a sweep's own
# stderr line -- best-effort; falls back to blank fields if a sweep's
# output doesn't match (e.g. it crashed before printing one).
parse_summary_line() {
    grep -oE '[0-9]+ (cases|shapes) tested' "$1" | tail -1 | grep -oE '^[0-9]+' || true
}
parse_wrong_count() {
    grep -oE '[0-9]+ WRONG' "$1" | tail -1 | grep -oE '^[0-9]+' || true
}

report() {
    name="$1"; shift
    run_count=$((run_count + 1))
    log="$RESULTS_DIR/$name.log"
    status=0
    "$@" > "$log" 2>&1 || status=$?
    tested="$(parse_summary_line "$log")"
    wrong="$(parse_wrong_count "$log")"
    if [ "$status" -eq 0 ]; then
        result="PASS"
    else
        result="FAIL"
        fail_count=$((fail_count + 1))
    fi
    echo "$name,$result,${tested:-},${wrong:-}" >> "$CSV"
    line="[$result] $name (tested=${tested:-?}, wrong=${wrong:-?})"
    echo "$line" | tee -a "$TXT"
}

[ -n "$ONLY" ] && echo "ONLY set -- running just: $ONLY"
echo "Results going to: $RESULTS_DIR"
echo "=== op-level sweeps ===" | tee -a "$TXT"
for op in activ_sweep pool_sweep bn_sweep softmax_sweep layernorm_sweep groupnorm_sweep tensorop_sweep reduce_sweep reduce_extreme_sweep glu_sweep cat_sweep rope_sweep kthvalue_sweep; do
    should_run "$op" || continue
    if [ -x "$BIN/$op" ]; then
        report "$op" "$BIN/$op"
    else
        echo "[SKIP] $op (not built)" | tee -a "$TXT"
        echo "$op,SKIP,," >> "$CSV"
    fi
done

echo | tee -a "$TXT"
echo "=== conv solver sweeps (forced via MIOPEN_DEBUG_FIND_ONLY_SOLVER) ===" | tee -a "$TXT"
if [ -x "$BIN/conv_solver_sweep" ]; then
    for shape_file in "$SHAPES"/*.txt; do
        solver="$(basename "$shape_file" .txt)"
        should_run "conv:$solver" || continue
        run_count=$((run_count + 1))
        log="$RESULTS_DIR/conv_$solver.log"
        status=0
        MIOPEN_DEBUG_FIND_ONLY_SOLVER="$solver" MIOPEN_FIND_MODE=normal \
            "$BIN/conv_solver_sweep" < "$shape_file" > "$log" 2>&1 || status=$?
        tested="$(parse_summary_line "$log")"
        wrong="$(parse_wrong_count "$log")"
        if [ "$status" -eq 0 ]; then
            result="PASS"
        else
            result="FAIL"
            fail_count=$((fail_count + 1))
        fi
        echo "conv:$solver,$result,${tested:-},${wrong:-}" >> "$CSV"
        echo "[$result] conv:$solver (tested=${tested:-?}, wrong=${wrong:-?})" | tee -a "$TXT"
    done
else
    echo "FATAL: conv_solver_sweep not built" >&2
    exit 1
fi

summary_line="SUMMARY: $((run_count - fail_count))/$run_count passed"
echo | tee -a "$TXT"
echo "=== $summary_line ===" | tee -a "$TXT"
echo "Full results: $RESULTS_DIR (summary.csv, summary.txt, per-sweep .log files)"

[ "$fail_count" -eq 0 ]
