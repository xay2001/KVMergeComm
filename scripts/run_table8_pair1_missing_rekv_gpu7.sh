#!/usr/bin/env bash
set -euo pipefail

# Fill the missing Table 8 pair #1 (Llama-3.1 same-model) ReKV paper-table runs.
# Missing runs:
#   hotpotqa: ReKV-w8 r=0.7, ReKV-w16 r=0.7
#   qasper:   ReKV-w8 r=0.3, ReKV-w8 r=0.5, ReKV-w8 r=0.7
#
# Existing probe runs are not treated as paper-table runs. This script skips only
# exact `recv_w{win}_r{ratio}_*/per_sample.jsonl` outputs.

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
MODEL=${MODEL:-/sharedspace/models/Llama-3.1-8B-Instruct}
ROOT=${ROOT:-snapshots}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOGDIR="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
mkdir -p "${LOGDIR}"
LOG_PATH="${LOGDIR}/gpu${GPU}_table8_pair1_missing_rekv_${TAG}.log"

if [[ ! -d "${MODEL}" ]]; then
  echo "Model path does not exist: ${MODEL}" >&2
  exit 1
fi

has_done_run() {
  local dir=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${dir}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_rekv() {
  local task=$1
  local win=$2
  local ratio=$3
  local out="${ROOT}/${task}/mtc_receiver"
  local run_name="recv_w${win}_r${ratio}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ReKV w${win} r=${ratio} already has per_sample.jsonl ===="
    return
  fi

  echo "==== [table8 pair1 GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL}" --model_B "${MODEL}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

(
  echo "######## table8 pair1 missing ReKV GPU${GPU} START $(date '+%F %T') ########"
  echo "MODEL=${MODEL}"
  echo "ROOT=${ROOT}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"

  run_rekv hotpotqa 8 0.7
  run_rekv hotpotqa 16 0.7
  run_rekv qasper 8 0.3
  run_rekv qasper 8 0.5
  run_rekv qasper 8 0.7

  echo "######## table8 pair1 missing ReKV GPU${GPU} DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "table8 pair1 missing ReKV GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results root -> ${ROOT}/{hotpotqa,qasper}/mtc_receiver"
