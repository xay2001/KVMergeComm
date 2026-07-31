#!/usr/bin/env bash
set -euo pipefail

# First-layer retention ablation for fixed ReKV and coverage-based B-ReKV.
# Usage: GPU=5 PAIR=1 bash scripts/run_first_layer_ablation_gpu.sh

cd "$(dirname "$0")/.." || exit 1

GPU=${GPU:?set GPU}
PAIR=${PAIR:-1}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
LIMIT=${LIMIT:-200}
RATIO=${RATIO:-0.3}
WINDOW=${WINDOW:-8}
FIRST_LAYER_MODES=${FIRST_LAYER_MODES:-"full uniform half query"}
METHODS=${METHODS:-"rekv brekv"}
ROOT=${ROOT:-snapshots/first_layer_ablation_v1}

pair_paths() {
  case "$1" in
    1) echo "pair1_llama31_same /sharedspace/models/Llama-3.1-8B-Instruct /sharedspace/models/Llama-3.1-8B-Instruct" ;;
    7) echo "pair7_qwen25_uncensored_bespoke /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored /sharedspace/models/Bespoke-Stratos-7B" ;;
    *) echo "unsupported PAIR=$1" >&2; return 2 ;;
  esac
}

read -r PAIR_SLUG MODEL_A MODEL_B <<< "$(pair_paths "${PAIR}")"
mkdir -p "${ROOT}/logs"
LOG_FILE="${ROOT}/logs/gpu${GPU}_${PAIR_SLUG}_$(date +%m%d_%H%M).log"

run_one() {
  local task=$1 method=$2 mode=$3
  local parent="${ROOT}/${PAIR_SLUG}/${task}/${method}"
  local run_name="${method}_l1-${mode}"
  if compgen -G "${parent}/${run_name}_*/per_sample.jsonl" >/dev/null; then
    echo "[skip] ${PAIR_SLUG} ${task} ${run_name}"
    return
  fi
  local args=(
    --test_task "${task}" --do_test --limit "${LIMIT}"
    --model_A "${MODEL_A}" --model_B "${MODEL_B}"
    --merge --merge_mode evict --score_mode receiver
    --recv_window "${WINDOW}" --query_sketch_mode bf16
    --merge_sink 4 --merge_recent 8
    --first_layer_mode "${mode}"
    --snapshot_path "${parent}" --run_name "${run_name}"
  )
  if [[ "${method}" == "rekv" ]]; then
    args+=(--merge_ratio "${RATIO}")
  else
    args+=(
      --budget_mode coverage --budget_min 0.05 --budget_max 0.70
      --coverage_tau 0.95 --coverage_scale 0.75
    )
  fi
  echo "[GPU${GPU}] ${PAIR_SLUG} ${task} ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py "${args[@]}"
}

{
  echo "First-layer ablation: GPU=${GPU} PAIR=${PAIR_SLUG} LIMIT=${LIMIT}"
  for task in ${TASKS}; do
    for mode in ${FIRST_LAYER_MODES}; do
      for method in ${METHODS}; do
        run_one "${task}" "${method}" "${mode}"
      done
    done
  done
} 2>&1 | tee "${LOG_FILE}"
