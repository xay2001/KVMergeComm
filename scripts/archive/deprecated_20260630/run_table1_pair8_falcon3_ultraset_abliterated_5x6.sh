#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# KVComm main Table 1, model pair #8:
#   M_s = ehristoforu/falcon3-ultraset
#   M_r = huihui-ai/Falcon3-7B-Instruct-abliterated
#
# This script runs our methods only:
#   - RASC: w8/w16 x r in {0.3, 0.5, 0.7}
#   - Coverage-BRASC: three representative points
#
# Results are separated from other model pairs under:
#   snapshots/table1_pair8_falcon3_ultraset_abliterated/

MODEL_A=${MODEL_A:-/sharedspace/models/falcon3-ultraset}
MODEL_B=${MODEL_B:-/sharedspace/models/Falcon3-7B-Instruct-abliterated}
ROOT=${ROOT:-snapshots/table1_pair8_falcon3_ultraset_abliterated}
GPU5=${GPU5:-5}
GPU6=${GPU6:-6}

# Split the 8 Table-1 datasets across two queues.
TASKS_GPU5=${TASKS_GPU5:-"countries tipsheets hotpotqa musique"}
TASKS_GPU6=${TASKS_GPU6:-"multifieldqa_en twowikimqa qasper tmath"}

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
      echo "==== [table1 pair8 GPU${gpu}] ${task} RASC w${win} r=${ratio} $(date '+%F %T') ===="
      run_rasc "${task}" "${gpu}" "${win}" "${ratio}"
    done
  done

  echo "==== [table1 pair8 GPU${gpu}] ${task} Coverage w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 8 0.95 0.75

  echo "==== [table1 pair8 GPU${gpu}] ${task} Coverage w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 8 0.95 0.85

  echo "==== [table1 pair8 GPU${gpu}] ${task} Coverage w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 16 0.95 0.90
}

GPU5_LOG="${LOGDIR}/gpu${GPU5}_pair8_${TAG}.log"
GPU6_LOG="${LOGDIR}/gpu${GPU6}_pair8_${TAG}.log"

(
  echo "######## [GPU${GPU5} table1 pair8 falcon3_ultraset -> falcon3_abliterated] START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "TASKS=${TASKS_GPU5}"

  for task in ${TASKS_GPU5}; do
    run_task "${task}" "${GPU5}"
  done

  echo "######## [GPU${GPU5} table1 pair8 falcon3_ultraset -> falcon3_abliterated] DONE $(date '+%F %T') ########"
) > "${GPU5_LOG}" 2>&1 &

P5=$!

(
  echo "######## [GPU${GPU6} table1 pair8 falcon3_ultraset -> falcon3_abliterated] START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "TASKS=${TASKS_GPU6}"

  for task in ${TASKS_GPU6}; do
    run_task "${task}" "${GPU6}"
  done

  echo "######## [GPU${GPU6} table1 pair8 falcon3_ultraset -> falcon3_abliterated] DONE $(date '+%F %T') ########"
) > "${GPU6_LOG}" 2>&1 &

P6=$!

echo "GPU${GPU5} table1 pair8 queue pid=${P5} -> ${GPU5_LOG}"
echo "GPU${GPU6} table1 pair8 queue pid=${P6} -> ${GPU6_LOG}"
echo "Results root -> ${ROOT}"
