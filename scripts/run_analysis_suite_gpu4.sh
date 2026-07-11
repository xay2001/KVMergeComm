#!/usr/bin/env bash
set -euo pipefail

# Serial analysis suite:
#   1) HotpotQA supporting-facts overlap (uses GPU for model forward)
#   2) Failure case analysis (CPU, existing per_sample files)
#   3) Task-type sensitivity (CPU, existing per_sample files)
#
# Defaults are chosen for a useful but not huge run. Override env vars if needed:
#   GPU=4
#   SUPPORT_LIMIT=200
#   SUPPORT_TOP_K=20
#   SUPPORT_MODEL=/sharedspace/models/Llama-3.1-8B-Instruct

cd /home/xay/KVComm || exit 1

GPU=${GPU:-4}
SUPPORT_LIMIT=${SUPPORT_LIMIT:-200}
SUPPORT_TOP_K=${SUPPORT_TOP_K:-20}
SUPPORT_WIN=${SUPPORT_WIN:-8}
SUPPORT_RATIO=${SUPPORT_RATIO:-0.3}
SUPPORT_MODEL=${SUPPORT_MODEL:-/sharedspace/models/Llama-3.1-8B-Instruct}

ROOT=${ROOT:-snapshots/analysis}
LOG_ROOT="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
LOG_PATH="${LOG_ROOT}/gpu${GPU}_analysis_suite_${TAG}.log"

mkdir -p "${LOG_ROOT}"

(
  echo "######## Analysis suite START $(date '+%F %T') ########"
  echo "GPU=${GPU}"
  echo "SUPPORT_LIMIT=${SUPPORT_LIMIT}"
  echo "SUPPORT_TOP_K=${SUPPORT_TOP_K}"
  echo "SUPPORT_WIN=${SUPPORT_WIN}"
  echo "SUPPORT_RATIO=${SUPPORT_RATIO}"
  echo "SUPPORT_MODEL=${SUPPORT_MODEL}"

  echo "######## 1/3 HotpotQA supporting-facts overlap ########"
  CUDA_VISIBLE_DEVICES=${GPU} python scripts/hotpot_supporting_overlap.py \
    --device cuda:0 \
    --model "${SUPPORT_MODEL}" \
    --limit "${SUPPORT_LIMIT}" \
    --top_k "${SUPPORT_TOP_K}" \
    --recv_window "${SUPPORT_WIN}" \
    --ratio "${SUPPORT_RATIO}" \
    --out_dir "snapshots/supporting_overlap/hotpotqa_pair1_full_context"

  echo "######## 2/3 Failure case analysis ########"
  python scripts/analyze_failure_cases.py \
    --out_dir "snapshots/analysis/failure_cases"

  echo "######## 3/3 Task-type sensitivity ########"
  python scripts/analyze_task_type_sensitivity.py \
    --out_dir "snapshots/analysis/task_type_sensitivity"

  echo "######## Analysis suite DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "analysis suite GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "supporting overlap -> snapshots/supporting_overlap/hotpotqa_pair1_full_context"
echo "failure cases -> snapshots/analysis/failure_cases"
echo "task sensitivity -> snapshots/analysis/task_type_sensitivity"
