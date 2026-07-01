#!/usr/bin/env bash
set -euo pipefail

# Unified queue for paper-aligned Table 1 / Table 8 ReKV and B-ReKV runs.
#
# Required:
#   TABLE_ID=1|8
#   PAIR_ID=<KVComm pair id>
#   MODEL_A=<sender model path>
#   MODEL_B=<receiver model path>
#   ROOT=snapshots/table{TABLE_ID}_pair{PAIR_ID}_{slug}
#
# Optional:
#   GPU_LIST="0 1"
#   TASKS="countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"

cd /home/xay/KVComm || exit 1

TABLE_ID=${TABLE_ID:?set TABLE_ID, e.g. 1 or 8}
PAIR_ID=${PAIR_ID:?set PAIR_ID, e.g. 6}
MODEL_A=${MODEL_A:?set MODEL_A}
MODEL_B=${MODEL_B:?set MODEL_B}
ROOT=${ROOT:?set ROOT, e.g. snapshots/table1_pair6_llama32_abliterated_deepseek3b}
GPU_LIST=${GPU_LIST:-0}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}

if [[ ! -d "${MODEL_A}" ]]; then
  echo "Sender model path does not exist: ${MODEL_A}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_B}" ]]; then
  echo "Receiver model path does not exist: ${MODEL_B}" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOGDIR="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
mkdir -p "${LOGDIR}"

read -r -a GPUS <<< "${GPU_LIST}"
read -r -a ALL_TASKS <<< "${TASKS}"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_LIST resolved to zero GPUs" >&2
  exit 1
fi

run_rekv() {
  local task=$1
  local gpu=$2
  local win=$3
  local ratio=$4

  CUDA_VISIBLE_DEVICES=${gpu} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${ROOT}/${task}/mtc_receiver" \
    --run_name "recv_w${win}_r${ratio}"
}

run_cov() {
  local task=$1
  local gpu=$2
  local win=$3
  local tau=$4
  local scale=$5

  CUDA_VISIBLE_DEVICES=${gpu} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${ROOT}/${task}/coverage" \
    --run_name "cov_t${tau}_s${scale}_w${win}"
}

run_task() {
  local task=$1
  local gpu=$2

  for win in 8 16; do
    for ratio in 0.3 0.5 0.7; do
      echo "==== [table${TABLE_ID} pair${PAIR_ID} GPU${gpu}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${task}" "${gpu}" "${win}" "${ratio}"
    done
  done

  echo "==== [table${TABLE_ID} pair${PAIR_ID} GPU${gpu}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 8 0.95 0.75

  echo "==== [table${TABLE_ID} pair${PAIR_ID} GPU${gpu}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 8 0.95 0.85

  echo "==== [table${TABLE_ID} pair${PAIR_ID} GPU${gpu}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 16 0.95 0.90
}

pids=()
for gpu_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$gpu_index]}
  gpu_tasks=()
  for task_index in "${!ALL_TASKS[@]}"; do
    if (( task_index % ${#GPUS[@]} == gpu_index )); then
      gpu_tasks+=("${ALL_TASKS[$task_index]}")
    fi
  done

  log_path="${LOGDIR}/gpu${gpu}_table${TABLE_ID}_pair${PAIR_ID}_${TAG}.log"
  (
    echo "######## [GPU${gpu} table${TABLE_ID} pair${PAIR_ID}] START $(date '+%F %T') ########"
    echo "MODEL_A=${MODEL_A}"
    echo "MODEL_B=${MODEL_B}"
    echo "ROOT=${ROOT}"
    echo "TASKS=${gpu_tasks[*]}"
    for task in "${gpu_tasks[@]}"; do
      run_task "${task}" "${gpu}"
    done
    echo "######## [GPU${gpu} table${TABLE_ID} pair${PAIR_ID}] DONE $(date '+%F %T') ########"
  ) > "${log_path}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  echo "GPU${gpu} table${TABLE_ID} pair${PAIR_ID} queue pid=${pid} -> ${log_path}"
done

echo "Results root -> ${ROOT}"
echo "Queue PIDs -> ${pids[*]}"
