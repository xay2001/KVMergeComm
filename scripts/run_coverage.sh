#!/bin/bash
# B-ReKV (Stage 1): training-free receiver-attention coverage budget.
#
# It keeps the minimum number of receiver-aware top tokens whose cumulative
# attention mass reaches coverage_tau, then scales/clamps the per-layer ratio.
#
# Usage:
#   TASK=musique GPU=2 bash scripts/run_coverage.sh
#   TASK=hotpotqa GPU=2 SCALES="0.7 0.9" TAUS="0.90 0.95" bash scripts/run_coverage.sh
#
# Recommended first pass from the offline simulation:
#   musique:  tau=0.90 scale=0.7/0.75/0.8/0.9, tau=0.95 scale=0.8
#   hotpotqa: tau=0.90 scale=0.7/0.9, tau=0.95 scale=0.9
set -e

TASK=${TASK:-musique}
GPU=${GPU:-0}
WIN=${WIN:-16}
MODEL=${MODEL:-/sharedspace/models/Llama-3.1-8B-Instruct}
OUT=${OUT:-snapshots/${TASK}/coverage}

# Default: run the most promising Stage-1 points. Override from shell if needed.
TAUS=${TAUS:-"0.90"}
SCALES=${SCALES:-"0.7 0.75 0.8 0.9"}
MIN_BUDGET=${MIN_BUDGET:-0.05}
MAX_BUDGET=${MAX_BUDGET:-0.7}

common() {
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task ${TASK} --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --merge --merge_mode evict --score_mode receiver --recv_window ${WIN} \
        --merge_sink 4 --merge_recent 8 \
        --budget_mode coverage \
        --budget_min ${MIN_BUDGET} --budget_max ${MAX_BUDGET} \
        --snapshot_path ${OUT} "$@"
}

for T in ${TAUS}; do
    for S in ${SCALES}; do
        echo "==== [${TASK} GPU${GPU}] coverage tau=${T} scale=${S} win=${WIN} ===="
        common --coverage_tau ${T} --coverage_scale ${S} --run_name cov_t${T}_s${S}_w${WIN}
    done
done

echo "==== coverage sweep done -> ${OUT} ===="
