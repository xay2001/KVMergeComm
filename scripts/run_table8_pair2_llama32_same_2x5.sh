#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# KVComm Appendix Table 8, model pair #2:
#   M_s = meta-llama/Llama-3.2-3B-Instruct
#   M_r = meta-llama/Llama-3.2-3B-Instruct
#
# This script only runs our methods (RASC / Coverage-BRASC). KVComm paper
# baselines are taken from the paper table.

MODEL=${MODEL:-/sharedspace/models/Llama-3.2-3B-Instruct}
ROOT=${ROOT:-snapshots/table8_pair2_llama32_same}
LOGDIR="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')

mkdir -p "${LOGDIR}"

if [[ ! -d "${MODEL}" ]]; then
  echo "Model path does not exist: ${MODEL}" >&2
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
    --model_A "${MODEL}" --model_B "${MODEL}" \
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
    --model_A "${MODEL}" --model_B "${MODEL}" \
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
      echo "==== [pair2 llama32_same GPU${gpu}] ${task} RASC w${win} r=${ratio} $(date '+%F %T') ===="
      run_rasc "${task}" "${gpu}" "${win}" "${ratio}"
    done
  done

  echo "==== [pair2 llama32_same GPU${gpu}] ${task} Coverage w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 8 0.95 0.75

  echo "==== [pair2 llama32_same GPU${gpu}] ${task} Coverage w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 8 0.95 0.85

  echo "==== [pair2 llama32_same GPU${gpu}] ${task} Coverage w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" "${gpu}" 16 0.95 0.90
}

GPU2_LOG="${LOGDIR}/gpu2_pair2_${TAG}.log"

# GPU2-only queue: run all eight datasets sequentially.
(
  echo "######## [GPU2 pair2 llama32_same] START $(date '+%F %T') ########"
  echo "MODEL=${MODEL}"
  echo "ROOT=${ROOT}"
  run_task countries 2
  run_task tipsheets 2
  run_task hotpotqa 2
  run_task musique 2
  run_task multifieldqa_en 2
  run_task twowikimqa 2
  run_task qasper 2
  run_task tmath 2
  echo "######## [GPU2 pair2 llama32_same] DONE $(date '+%F %T') ########"
) > "${GPU2_LOG}" 2>&1 &

P2=$!

echo "GPU2 pair2 queue pid=${P2} -> ${GPU2_LOG}"
echo "Results root -> ${ROOT}"
