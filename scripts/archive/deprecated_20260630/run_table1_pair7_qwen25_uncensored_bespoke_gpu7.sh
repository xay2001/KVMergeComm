#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# KVComm main Table 1, model pair #7:
#   M_s = Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
#   M_r = bespokelabs/Bespoke-Stratos-7B
#
# This script runs our methods only:
#   - RASC: w8/w16 x r in {0.3, 0.5, 0.7}
#   - Coverage-BRASC: three representative points

MODEL_A=${MODEL_A:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B=${MODEL_B:-/sharedspace/models/Bespoke-Stratos-7B}
ROOT=${ROOT:-snapshots/table1_pair7_qwen25_uncensored_bespoke}
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

run_rasc() {
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
      echo "==== [table1 pair7 GPU${GPU}] ${task} RASC w${win} r=${ratio} $(date '+%F %T') ===="
      run_rasc "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [table1 pair7 GPU${GPU}] ${task} Coverage w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.75

  echo "==== [table1 pair7 GPU${GPU}] ${task} Coverage w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.85

  echo "==== [table1 pair7 GPU${GPU}] ${task} Coverage w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" 16 0.95 0.90
}

GPU_LOG="${LOGDIR}/gpu${GPU}_pair7_${TAG}.log"

(
  echo "######## [GPU${GPU} table1 pair7 qwen25_uncensored -> bespoke_stratos] START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "TASKS=${TASKS}"

  for task in ${TASKS}; do
    run_task "${task}"
  done

  echo "######## [GPU${GPU} table1 pair7 qwen25_uncensored -> bespoke_stratos] DONE $(date '+%F %T') ########"
) > "${GPU_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} table1 pair7 queue pid=${P} -> ${GPU_LOG}"
echo "Results root -> ${ROOT}"
