#!/usr/bin/env bash
set -euo pipefail

# Dump top-token interpretability examples on GPU 7.
#
# Outputs:
#   snapshots/interpretability/pair1_llama31_same/<task>/top_tokens_*.jsonl
#   snapshots/interpretability/pair1_llama31_same/<task>/answer_overlap_*.csv
#
# Run:
#   bash scripts/run_interpretability_dump_gpu2_7.sh
#
# Optional:
#   LIMIT=20 TOP_K=30 RECV_WINDOW=16 bash scripts/run_interpretability_dump_gpu2_7.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
MODEL=${MODEL:-/sharedspace/models/Llama-3.1-8B-Instruct}
LIMIT=${LIMIT:-50}
TOP_K=${TOP_K:-20}
RATIO=${RATIO:-0.3}
RECV_WINDOW=${RECV_WINDOW:-8}
HIDDEN_DIM=${HIDDEN_DIM:-4096}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
OUT_DIR=${OUT_DIR:-snapshots/interpretability/pair1_llama31_same}

mkdir -p "${OUT_DIR}/logs"
TAG=$(date '+%m%d_%H%M')

run_dump() {
  local gpu=$1
  shift
  local tasks=("$@")
  CUDA_VISIBLE_DEVICES=${gpu} python scripts/dump_interpretability_examples.py \
    --tasks "${tasks[@]}" \
    --model "${MODEL}" \
    --device cuda:0 \
    --limit "${LIMIT}" \
    --top_k "${TOP_K}" \
    --ratio "${RATIO}" \
    --recv_window "${RECV_WINDOW}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --out_dir "${OUT_DIR}"
}

read -r -a TASK_ARRAY <<< "${TASKS}"

(
  echo "######## interpretability GPU${GPU} START $(date '+%F %T') ########"
  run_dump "${GPU}" "${TASK_ARRAY[@]}"
  echo "######## interpretability GPU${GPU} DONE $(date '+%F %T') ########"
) > "${OUT_DIR}/logs/gpu${GPU}_interpretability_${TAG}.log" 2>&1 &
P=$!

echo "GPU${GPU} interpretability pid=${P} -> ${OUT_DIR}/logs/gpu${GPU}_interpretability_${TAG}.log"
echo "Output root -> ${OUT_DIR}"
