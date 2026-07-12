#!/usr/bin/env bash
set -euo pipefail

# B-ReKV with one per-query adaptive parameter: tau(Q).
# There is no coverage scaling factor or budget clamp.

cd /home/xay/KVComm || exit 1

PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B=${MODEL_B:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT=${ROOT:-snapshots/table1_pair6_adaptive_tau_only_brekv_query_sketch}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_one() {
  local gpu=$1 task=$2 tau_min=$3 tau_max=$4
  local out="${ROOT}/${task}/strict_coverage"
  local run_name="adapt_t${tau_min}-${tau_max}_w8"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi

  echo "==== [GPU${gpu}] ${task} tau-only adaptive tau=[${tau_min},${tau_max}] $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver \
    --recv_window 8 --merge_sink 4 --merge_recent 8 \
    --budget_mode strict_coverage \
    --coverage_tau_mode adaptive \
    --coverage_tau_min "${tau_min}" \
    --coverage_tau_max "${tau_max}" \
    --budget_min 0.0 --budget_max 1.0 --budget_floor 0.0 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_task() {
  local gpu=$1 task=$2
  run_one "${gpu}" "${task}" 0.90 0.99
}

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')

run_gpu() {
  local gpu=$1 task=$2
  local log="${ROOT}/logs/gpu${gpu}_${task}_adaptive_tau_${TAG}.log"
  (
    echo "######## GPU${gpu} ${task} adaptive-tau B-ReKV START $(date '+%F %T') ########"
    run_task "${gpu}" "${task}"
    echo "######## GPU${gpu} ${task} adaptive-tau B-ReKV DONE $(date '+%F %T') ########"
  ) > "${log}" 2>&1 &
  echo "GPU${gpu} pid=$!: ${task}; log=${log}"
}

run_gpu 0 hotpotqa
run_gpu 1 musique
run_gpu 2 multifieldqa_en

wait
echo "All adaptive-tau B-ReKV runs finished: ${ROOT}"
