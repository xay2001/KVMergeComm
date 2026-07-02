#!/usr/bin/env bash
set -euo pipefail

# Causal-ish evidence test for ReKV interpretability.
# It masks the top content tokens selected by ReKV / Evict / Random and measures
# answer-score degradation under the same ReKV communication pipeline.

cd /home/xay/KVComm || exit 1

GPU=${GPU:-2}
MODEL=${MODEL:-/sharedspace/models/Llama-3.1-8B-Instruct}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
LIMIT=${LIMIT:-50}
TOP_K=${TOP_K:-20}
RATIO=${RATIO:-0.3}
RECV_WINDOW=${RECV_WINDOW:-8}
OUT_DIR=${OUT_DIR:-snapshots/deletion_ablation/pair1_llama31_same}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${OUT_DIR}/logs"
TAG=$(date '+%m%d_%H%M')
LOG_PATH="${OUT_DIR}/logs/gpu${GPU}_deletion_ablation_${TAG}.log"

if [[ ! -d "${MODEL}" ]]; then
  echo "Model path does not exist: ${MODEL}" >&2
  exit 1
fi

(
  echo "######## deletion ablation GPU${GPU} START $(date '+%F %T') ########"
  echo "MODEL=${MODEL}"
  echo "TASKS=${TASKS}"
  echo "LIMIT=${LIMIT}"
  echo "TOP_K=${TOP_K}"
  echo "RATIO=${RATIO}"
  echo "RECV_WINDOW=${RECV_WINDOW}"
  CUDA_VISIBLE_DEVICES=${GPU} python scripts/run_deletion_ablation.py \
    --model "${MODEL}" \
    --tasks ${TASKS} \
    --limit "${LIMIT}" \
    --top_k "${TOP_K}" \
    --ratio "${RATIO}" \
    --recv_window "${RECV_WINDOW}" \
    --device cuda:0 \
    --out_dir "${OUT_DIR}"
  echo "######## deletion ablation GPU${GPU} DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "deletion ablation GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results root -> ${OUT_DIR}"
