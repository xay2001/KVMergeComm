#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# KVComm Appendix Table 8, model pair #3:
#   M_s = Qwen/Qwen2.5-7B-Instruct
#   M_r = Qwen/Qwen2.5-7B-Instruct
#
# This script only runs our methods (ReKV / B-ReKV). KVComm paper
# baselines are taken from the paper table.

MODEL=${MODEL:-/sharedspace/models/Qwen2.5-7B-Instruct}
ROOT=${ROOT:-snapshots/table8_pair3_qwen25_7b_same}
LOGDIR="${ROOT}/logs"
GPU=${GPU:-6}
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

run_rekv() {
  local task=$1
  local win=$2
  local ratio=$3

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL}" --model_B "${MODEL}" \
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

  for win in 8 16; do
    for ratio in 0.3 0.5 0.7; do
      echo "==== [pair3 qwen25_7b_same GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [pair3 qwen25_7b_same GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.75

  echo "==== [pair3 qwen25_7b_same GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.85

  echo "==== [pair3 qwen25_7b_same GPU${GPU}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" 16 0.95 0.90
}

GPU_LOG="${LOGDIR}/gpu${GPU}_pair3_${TAG}.log"

(
  echo "######## [GPU${GPU} pair3 qwen25_7b_same] START $(date '+%F %T') ########"
  echo "MODEL=${MODEL}"
  echo "ROOT=${ROOT}"
  run_task countries
  run_task tipsheets
  run_task hotpotqa
  run_task musique
  run_task multifieldqa_en
  run_task twowikimqa
  run_task qasper
  run_task tmath
  echo "######## [GPU${GPU} pair3 qwen25_7b_same] DONE $(date '+%F %T') ########"
) > "${GPU_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} pair3 queue pid=${P} -> ${GPU_LOG}"
echo "Results root -> ${ROOT}"
