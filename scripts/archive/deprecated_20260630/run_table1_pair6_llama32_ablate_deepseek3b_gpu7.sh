#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# KVComm main Table 1, model pair #6:
#   M_s = huihui-ai/Llama-3.2-3B-Instruct-abliterated
#   M_r = suayptalha/DeepSeek-R1-Distill-Llama-3B
#
# This script runs our methods only:
#   - ReKV: w8/w16 x r in {0.3, 0.5, 0.7}
#   - B-ReKV: three representative points

MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B=${MODEL_B:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT=${ROOT:-snapshots/table1_pair6_llama32_abliterated_deepseek3b}
GPU=${GPU:-7}
TASKS=${TASKS:-"countries tipsheets hotpotqa musique multifieldqa_en twowikimqa qasper tmath"}

LOGDIR="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')

mkdir -p "${LOGDIR}"

if [[ ! -d "${MODEL_A}" ]]; then
  echo "Sender model path does not exist: ${MODEL_A}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_B}" ]]; then
  echo "Receiver model path does not exist: ${MODEL_B}" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
import importlib.metadata as m

print("transformers", m.version("transformers"))
print("huggingface-hub", m.version("huggingface-hub"))
PY

run_rekv() {
  local task=$1
  local win=$2
  local ratio=$3

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${ROOT}/${task}/mtc_receiver" \
    --run_name "recv_w${win}_r${ratio}"
}

run_cov() {
  local task=$1
  local win=$2
  local tau=$3
  local scale=$4

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
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

  for win in 8 16; do
    for ratio in 0.3 0.5 0.7; do
      echo "==== [table1 pair6 GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [table1 pair6 GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.75

  echo "==== [table1 pair6 GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.85

  echo "==== [table1 pair6 GPU${GPU}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" 16 0.95 0.90
}

GPU_LOG="${LOGDIR}/gpu${GPU}_pair6_${TAG}.log"

(
  echo "######## [GPU${GPU} table1 pair6 llama32_abliterated -> deepseek3b] START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "TASKS=${TASKS}"

  for task in ${TASKS}; do
    run_task "${task}"
  done

  echo "######## [GPU${GPU} table1 pair6 llama32_abliterated -> deepseek3b] DONE $(date '+%F %T') ########"
) > "${GPU_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} table1 pair6 queue pid=${P} -> ${GPU_LOG}"
echo "Results root -> ${ROOT}"
