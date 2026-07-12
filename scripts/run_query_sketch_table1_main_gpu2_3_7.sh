#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

GPU_PAIR1=${GPU_PAIR1:-2}
GPU_PAIR6=${GPU_PAIR6:-3}
GPU_PAIR7=${GPU_PAIR7:-7}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
SELECTION=${SELECTION:-snapshots/query_sketch_config_freeze/analysis/selection.json}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SELECTION_WAIT_SECONDS=${SELECTION_WAIT_SECONDS:-86400}

load_frozen_config() {
  if [[ ! -f "${SELECTION}" ]]; then
    echo "ReKV matrix finished; waiting for phase-1 configuration freeze: ${SELECTION}"
    local waited=0
    while [[ ! -f "${SELECTION}" && "${waited}" -lt "${SELECTION_WAIT_SECONDS}" ]]; do
      sleep 60
      waited=$((waited + 60))
    done
    if [[ ! -f "${SELECTION}" ]]; then
      echo "Timed out waiting for frozen B-ReKV configuration." >&2
      return 2
    fi
  fi
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
      return 2
    fi
  fi
}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_rekv() {
  local gpu=$1 root=$2 model_a=$3 model_b=$4 task=$5 window=$6 ratio=$7
  local out="${root}/${task}/mtc_receiver"
  local run_name="recv_w${window}_r${ratio}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${task} Query-Sketch ReKV w${window} r=${ratio} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver \
    --query_sketch_mode bf16 --recv_window "${window}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_brekv() {
  local gpu=$1 root=$2 model_a=$3 model_b=$4 task=$5
  local out="${root}/${task}/coverage_frozen"
  local run_name="frozen_t${TAU}_s${SCALE}_w${BREKV_WINDOW}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${task} Query-Sketch B-ReKV ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver \
    --query_sketch_mode bf16 --recv_window "${BREKV_WINDOW}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${TAU}" --coverage_scale "${SCALE}" \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_pair_rekv() {
  local gpu=$1 root=$2 model_a=$3 model_b=$4
  for task in ${TASKS}; do
    for window in 8 16; do
      for ratio in ${RATIOS}; do
        run_rekv "${gpu}" "${root}" "${model_a}" "${model_b}" "${task}" "${window}" "${ratio}"
      done
    done
  done
}

run_pair_brekv() {
  local gpu=$1 root=$2 model_a=$3 model_b=$4
  for task in ${TASKS}; do
    run_brekv "${gpu}" "${root}" "${model_a}" "${model_b}" "${task}"
  done
}

ROOT1=${ROOT1:-snapshots/table1_pair1_query_sketch_llama31_same}
ROOT6=${ROOT6:-snapshots/table1_pair6_query_sketch_llama32_abliterated_deepseek3b}
ROOT7=${ROOT7:-snapshots/table1_pair7_query_sketch_qwen25_uncensored_bespoke}
mkdir -p "${ROOT1}/logs" "${ROOT6}/logs" "${ROOT7}/logs"
TAG=$(date '+%m%d_%H%M')

(
  run_pair_rekv "${GPU_PAIR1}" "${ROOT1}" \
    /sharedspace/models/Llama-3.1-8B-Instruct \
    /sharedspace/models/Llama-3.1-8B-Instruct
) > "${ROOT1}/logs/gpu${GPU_PAIR1}_table1_rekv_${TAG}.log" 2>&1 &
PID1=$!

(
  run_pair_rekv "${GPU_PAIR6}" "${ROOT6}" \
    /sharedspace/models/Llama-3.2-3B-Instruct-abliterated \
    /sharedspace/models/DeepSeek-R1-Distill-Llama-3B
) > "${ROOT6}/logs/gpu${GPU_PAIR6}_table1_rekv_${TAG}.log" 2>&1 &
PID6=$!

(
  run_pair_rekv "${GPU_PAIR7}" "${ROOT7}" \
    /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored \
    /sharedspace/models/Bespoke-Stratos-7B
) > "${ROOT7}/logs/gpu${GPU_PAIR7}_table1_rekv_${TAG}.log" 2>&1 &
PID7=$!

echo "ReKV pair1 GPU${GPU_PAIR1} pid=${PID1}"
echo "ReKV pair6 GPU${GPU_PAIR6} pid=${PID6}"
echo "ReKV pair7 GPU${GPU_PAIR7} pid=${PID7}"
wait "${PID1}" "${PID6}" "${PID7}"

load_frozen_config

(
  run_pair_brekv "${GPU_PAIR1}" "${ROOT1}" \
    /sharedspace/models/Llama-3.1-8B-Instruct \
    /sharedspace/models/Llama-3.1-8B-Instruct
) > "${ROOT1}/logs/gpu${GPU_PAIR1}_table1_brekv_${TAG}.log" 2>&1 &
PID1=$!

(
  run_pair_brekv "${GPU_PAIR6}" "${ROOT6}" \
    /sharedspace/models/Llama-3.2-3B-Instruct-abliterated \
    /sharedspace/models/DeepSeek-R1-Distill-Llama-3B
) > "${ROOT6}/logs/gpu${GPU_PAIR6}_table1_brekv_${TAG}.log" 2>&1 &
PID6=$!

(
  run_pair_brekv "${GPU_PAIR7}" "${ROOT7}" \
    /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored \
    /sharedspace/models/Bespoke-Stratos-7B
) > "${ROOT7}/logs/gpu${GPU_PAIR7}_table1_brekv_${TAG}.log" 2>&1 &
PID7=$!

echo "B-ReKV pair1 GPU${GPU_PAIR1} pid=${PID1}"
echo "B-ReKV pair6 GPU${GPU_PAIR6} pid=${PID6}"
echo "B-ReKV pair7 GPU${GPU_PAIR7} pid=${PID7}"
wait "${PID1}" "${PID6}" "${PID7}"
