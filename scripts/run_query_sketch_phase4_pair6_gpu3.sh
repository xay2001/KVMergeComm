#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# Pair #6 is used here because its Table-1 ReKV matrix is already complete on
# GPU3. These mechanism ablations do not depend on the frozen B-ReKV config.

GPU=${GPU:-3}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}

FOREGROUND=1 \
GPU="${GPU}" \
PAIRS=6 \
TASKS="${TASKS}" \
RATIOS=0.3 \
WINDOWS=8 \
ROOT=snapshots/query_sketch_score_function_ablation \
bash scripts/run_score_function_ablation_gpu6.sh

FOREGROUND=1 \
GPU="${GPU}" \
PAIRS=6 \
TASKS="${TASKS}" \
RATIOS=0.3 \
WINDOWS=8 \
AGGS="identity last mean top4 last4" \
ROOT=snapshots/query_sketch_layer_aggregation_ablation \
bash scripts/run_layer_aggregation_ablation_gpu5.sh
