#!/usr/bin/env bash
set -euo pipefail

# Controlled Query-Sketch validation used to freeze one global calibrated
# B-ReKV (tau, scale, window) configuration. Pair #1 and pair #6 run in
# parallel; every cell uses the same first LIMIT examples.

cd /home/xay/KVComm || exit 1

GPU_PAIR1=${GPU_PAIR1:-0}
GPU_PAIR6=${GPU_PAIR6:-1}
LIMIT=${LIMIT:-100}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
FIXED_RATIOS=${FIXED_RATIOS:-"0.20 0.25 0.30 0.35 0.40"}
CANDIDATES=${CANDIDATES:-"0.90:0.95 0.95:0.85 0.95:0.95 0.95:1.00 0.98:0.95 0.98:1.00"}
WINDOW=${WINDOW:-8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
ROOT=${ROOT:-snapshots/query_sketch_config_freeze}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_one() {
  local gpu=$1 model_a=$2 model_b=$3 pair=$4 task=$5 kind=$6 value1=$7 value2=${8:-}
  local out run_name
  if [[ "${kind}" == "fixed" ]]; then
    out="${ROOT}/${pair}/${task}/fixed"
    run_name="rekv_w${WINDOW}_r${value1}"
  else
    out="${ROOT}/${pair}/${task}/coverage"
    run_name="brekv_t${value1}_s${value2}_w${WINDOW}"
  fi
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${pair} ${task} ${run_name} ===="
    return
  fi

  echo "==== [GPU${gpu}] ${pair} ${task} ${run_name} $(date '+%F %T') ===="
  if [[ "${kind}" == "fixed" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
      --test_task "${task}" --do_test --limit "${LIMIT}" \
      --model_A "${model_a}" --model_B "${model_b}" \
      --merge --merge_mode evict --score_mode receiver \
      --recv_window "${WINDOW}" --merge_ratio "${value1}" \
      --merge_sink 4 --merge_recent 8 \
      --snapshot_path "${out}" --run_name "${run_name}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
      --test_task "${task}" --do_test --limit "${LIMIT}" \
      --model_A "${model_a}" --model_B "${model_b}" \
      --merge --merge_mode evict --score_mode receiver \
      --recv_window "${WINDOW}" --merge_sink 4 --merge_recent 8 \
      --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
      --coverage_tau "${value1}" --coverage_scale "${value2}" \
      --snapshot_path "${out}" --run_name "${run_name}"
  fi
}

run_pair() {
  local gpu=$1 model_a=$2 model_b=$3 pair=$4
  for task in ${TASKS}; do
    for ratio in ${FIXED_RATIOS}; do
      run_one "${gpu}" "${model_a}" "${model_b}" "${pair}" "${task}" fixed "${ratio}"
    done
    for candidate in ${CANDIDATES}; do
      IFS=: read -r tau scale <<< "${candidate}"
      run_one "${gpu}" "${model_a}" "${model_b}" "${pair}" "${task}" coverage "${tau}" "${scale}"
    done
  done
}

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
LOG1="${ROOT}/logs/gpu${GPU_PAIR1}_pair1_${TAG}.log"
LOG6="${ROOT}/logs/gpu${GPU_PAIR6}_pair6_${TAG}.log"

(
  run_pair "${GPU_PAIR1}" \
    /sharedspace/models/Llama-3.1-8B-Instruct \
    /sharedspace/models/Llama-3.1-8B-Instruct \
    pair1_llama31_same
) > "${LOG1}" 2>&1 &
PID1=$!

(
  run_pair "${GPU_PAIR6}" \
    /sharedspace/models/Llama-3.2-3B-Instruct-abliterated \
    /sharedspace/models/DeepSeek-R1-Distill-Llama-3B \
    pair6_llama32_abliterated_deepseek3b
) > "${LOG6}" 2>&1 &
PID6=$!

echo "pair1 GPU${GPU_PAIR1} pid=${PID1}: ${LOG1}"
echo "pair6 GPU${GPU_PAIR6} pid=${PID6}: ${LOG6}"
wait "${PID1}" "${PID6}"

"${PYTHON}" scripts/analyze_query_sketch_config_freeze.py --root "${ROOT}"
