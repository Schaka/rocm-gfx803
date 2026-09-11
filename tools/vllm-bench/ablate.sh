#!/usr/bin/env bash
# Ablate the gfx803 fast paths to attribute batched-decode step time.
#
# Every row runs the same workload with exactly one thing changed, so the
# difference between rows is that one thing's cost. Small-concurrency decode
# is the workload, because that is where the batched path first engages and
# where step time jumps ~5x over the single-request path.
#
# Source env.sh first: it carries the shim and cache settings every row needs.
#
# Usage: ./ablate.sh [concurrency] [decode] [label]
set -u

CONC=${1:-8}
DECODE=${2:-32}
LABEL=${3:-ablate}
OUT=/data/bench/${LABEL}.log
PY=/opt/venv/bin/python
BENCH=/data/bench/bench_queries.py
COMMON="--concurrency ${CONC} --prompt-len 128 --decode ${DECODE} --repeats 2 --warmup 1 --max-num-seqs 32"

run() {
    local name=$1 extra_env=$2 extra_args=$3
    echo "=== ${name} ===" | tee -a "$OUT"
    env ${extra_env} ${PY} ${BENCH} ${COMMON} ${extra_args} --label "${name}" \
        2>&1 | grep -E '^\{' | tee -a "$OUT"
}

: > "$OUT"
run default "" ""
run no_cudagraph "" "--enforce-eager"
run tensors_only VLLM_GFX803_GEMM_CACHE_MB=0 ""
run triton_attn VLLM_GFX803_ATTN_HIP_KERNEL=0 ""
