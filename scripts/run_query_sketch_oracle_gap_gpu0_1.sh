#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
ROOT=${ROOT:-snapshots/query_sketch_oracle_gap}
SELECTION=${SELECTION:-snapshots/query_sketch_config_freeze/analysis/selection.json}
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
REKV_RATIO=${REKV_RATIO:-0.30}
SKIP_EXISTING=${SKIP_EXISTING:-1}

if [[ -z "${TAU:-}" || -z "${SCALE:-}" || -z "${BREKV_WINDOW:-}" ]]; then
  read -r ACCEPTED TAU SCALE BREKV_WINDOW < <(
    "${PYTHON}" - "${SELECTION}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
row = data.get("selection") or {}
print(int(bool(data.get("accepted"))), row.get("tau", ""), row.get("scale", ""), row.get("window", ""))
PY
  )
  if [[ "${ACCEPTED}" != "1" ]]; then
    echo "No B-ReKV candidate passed matched-budget validation: ${SELECTION}" >&2
    exit 2
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/cost_summary.json" > /dev/null
}

run_method() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4 task=$5 method=$6 score_mode=$7
  local out="${ROOT}/${pair}/${task}/${method}"
  local run_name="${method}_${score_mode}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${pair} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${pair} ${task} ${run_name} $(date '+%F %T') ===="
  local args=(
    --test_task "${task}" --do_test
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}"
    --model_A "${model_a}" --model_B "${model_b}"
    --merge --merge_mode evict --score_mode "${score_mode}"
    --query_sketch_mode bf16
    --merge_sink 4 --merge_recent 8
    --snapshot_path "${out}" --run_name "${run_name}"
  )
  if [[ "${method}" == "rekv" ]]; then
    args+=(--merge_ratio "${REKV_RATIO}" --recv_window 8)
  else
    args+=(
      --budget_mode coverage --budget_min 0.05 --budget_max 0.7
      --coverage_tau "${TAU}" --coverage_scale "${SCALE}"
      --recv_window "${BREKV_WINDOW}"
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py "${args[@]}"
}

run_cell() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4 task=$5
  run_method "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" rekv receiver
  run_method "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" rekv receiver_oracle
  run_method "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" brekv receiver
  run_method "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" brekv receiver_oracle
}

run_queue() {
  local gpu=$1
  shift
  local item pair model_a model_b task
  for item in "$@"; do
    IFS='|' read -r pair model_a model_b task <<< "${item}"
    run_cell "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}"
  done
}

P1_A=/sharedspace/models/Llama-3.1-8B-Instruct
P1_B=/sharedspace/models/Llama-3.1-8B-Instruct
P6_A=/sharedspace/models/Llama-3.2-3B-Instruct-abliterated
P6_B=/sharedspace/models/DeepSeek-R1-Distill-Llama-3B
P7_A=/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored
P7_B=/sharedspace/models/Bespoke-Stratos-7B

QUEUE0=(
  "pair1_llama31_same|${P1_A}|${P1_B}|hotpotqa"
  "pair1_llama31_same|${P1_A}|${P1_B}|musique"
  "pair1_llama31_same|${P1_A}|${P1_B}|multifieldqa_en"
  "pair7_qwen25_uncensored_bespoke|${P7_A}|${P7_B}|hotpotqa"
  "pair7_qwen25_uncensored_bespoke|${P7_A}|${P7_B}|musique"
)
QUEUE1=(
  "pair6_llama32_abliterated_deepseek3b|${P6_A}|${P6_B}|hotpotqa"
  "pair6_llama32_abliterated_deepseek3b|${P6_A}|${P6_B}|musique"
  "pair6_llama32_abliterated_deepseek3b|${P6_A}|${P6_B}|multifieldqa_en"
  "pair7_qwen25_uncensored_bespoke|${P7_A}|${P7_B}|multifieldqa_en"
)

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
LOG0="${ROOT}/logs/gpu${GPU0}_oracle_gap_${TAG}.log"
LOG1="${ROOT}/logs/gpu${GPU1}_oracle_gap_${TAG}.log"
(run_queue "${GPU0}" "${QUEUE0[@]}") > "${LOG0}" 2>&1 &
PID0=$!
(run_queue "${GPU1}" "${QUEUE1[@]}") > "${LOG1}" 2>&1 &
PID1=$!
echo "GPU${GPU0} pid=${PID0}: ${LOG0}"
echo "GPU${GPU1} pid=${PID1}: ${LOG1}"
wait "${PID0}" "${PID1}"
