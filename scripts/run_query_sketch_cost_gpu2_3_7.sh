#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

GPU_PAIR1=${GPU_PAIR1:-2}
GPU_PAIR6=${GPU_PAIR6:-3}
GPU_PAIR7=${GPU_PAIR7:-7}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
SELECTION=${SELECTION:-snapshots/query_sketch_config_freeze/analysis/selection.json}
ROOT=${ROOT:-snapshots/query_sketch_cost_v1}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
REKV_RATIO=${REKV_RATIO:-0.30}
REKV_WINDOW=${REKV_WINDOW:-8}
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
    echo "No frozen B-ReKV configuration is available: ${SELECTION}" >&2
    exit 2
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/cost_summary.json" > /dev/null
}

run_one() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4 task=$5 method=$6
  local out="${ROOT}/${pair}/${task}/${method}"
  local run_name
  if [[ "${method}" == "rekv" ]]; then
    run_name="rekv_bf16_w${REKV_WINDOW}_r${REKV_RATIO}"
  else
    run_name="brekv_bf16_t${TAU}_s${SCALE}_w${BREKV_WINDOW}"
  fi
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${pair} ${task} ${run_name} ===="
    return
  fi

  local args=(
    --test_task "${task}" --do_test
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}"
    --model_A "${model_a}" --model_B "${model_b}"
    --merge --merge_mode evict --score_mode receiver
    --query_sketch_mode bf16 --merge_sink 4 --merge_recent 8
    --snapshot_path "${out}" --run_name "${run_name}"
  )
  if [[ "${method}" == "rekv" ]]; then
    args+=(--recv_window "${REKV_WINDOW}" --merge_ratio "${REKV_RATIO}")
  else
    args+=(
      --recv_window "${BREKV_WINDOW}"
      --budget_mode coverage --budget_min 0.05 --budget_max 0.7
      --coverage_tau "${TAU}" --coverage_scale "${SCALE}"
    )
  fi
  echo "==== [GPU${gpu}] ${pair} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py "${args[@]}"
}

run_pair() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4
  for task in ${TASKS}; do
    run_one "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" rekv
    run_one "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" brekv
  done
}

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')

(
  run_pair "${GPU_PAIR1}" pair1_llama31_same \
    /sharedspace/models/Llama-3.1-8B-Instruct \
    /sharedspace/models/Llama-3.1-8B-Instruct
) > "${ROOT}/logs/gpu${GPU_PAIR1}_pair1_cost_${TAG}.log" 2>&1 &
PID1=$!

(
  run_pair "${GPU_PAIR6}" pair6_llama32_abliterated_deepseek3b \
    /sharedspace/models/Llama-3.2-3B-Instruct-abliterated \
    /sharedspace/models/DeepSeek-R1-Distill-Llama-3B
) > "${ROOT}/logs/gpu${GPU_PAIR6}_pair6_cost_${TAG}.log" 2>&1 &
PID6=$!

(
  run_pair "${GPU_PAIR7}" pair7_qwen25_uncensored_bespoke \
    /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored \
    /sharedspace/models/Bespoke-Stratos-7B
) > "${ROOT}/logs/gpu${GPU_PAIR7}_pair7_cost_${TAG}.log" 2>&1 &
PID7=$!

echo "pair1 GPU${GPU_PAIR1} pid=${PID1}"
echo "pair6 GPU${GPU_PAIR6} pid=${PID6}"
echo "pair7 GPU${GPU_PAIR7} pid=${PID7}"
wait "${PID1}" "${PID6}" "${PID7}"

"${PYTHON}" scripts/analyze_cost_profile.py \
  --root "${ROOT}" --csv "${ROOT}/cost_table.csv" > "${ROOT}/cost_table.md"
echo "Cost summary: ${ROOT}/cost_table.csv"
