#!/usr/bin/env bash
set -euo pipefail

# Fill the Table 1 pair #7 TMATH paper-table block on GPU 7.
# It skips any run that already has a per_sample.jsonl file.

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
MODEL_A=${MODEL_A:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B=${MODEL_B:-/sharedspace/models/Bespoke-Stratos-7B}
ROOT=${ROOT:-snapshots/table1_pair7_qwen25_uncensored_bespoke}
TASK=${TASK:-tmath}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOGDIR="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
mkdir -p "${LOGDIR}"
LOG_PATH="${LOGDIR}/gpu${GPU}_pair7_tmath_${TAG}.log"

if [[ ! -d "${MODEL_A}" ]]; then
  echo "Sender model path does not exist: ${MODEL_A}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_B}" ]]; then
  echo "Receiver model path does not exist: ${MODEL_B}" >&2
  exit 1
fi

has_done_run() {
  local dir=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${dir}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_rekv() {
  local win=$1
  local ratio=$2
  local out="${ROOT}/${TASK}/mtc_receiver"
  local run_name="recv_w${win}_r${ratio}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${TASK} ReKV w${win} r=${ratio} already has per_sample.jsonl ===="
    return
  fi

  echo "==== [pair7 GPU${GPU}] ${TASK} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${TASK}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_brekv() {
  local win=$1
  local tau=$2
  local scale=$3
  local out="${ROOT}/${TASK}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${TASK} B-ReKV w${win} tau=${tau} scale=${scale} already has per_sample.jsonl ===="
    return
  fi

  echo "==== [pair7 GPU${GPU}] ${TASK} B-ReKV w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${TASK}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

(
  echo "######## pair7 TMATH GPU${GPU} START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "TASK=${TASK}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"

  for win in 8 16; do
    for ratio in 0.3 0.5 0.7; do
      run_rekv "${win}" "${ratio}"
    done
  done

  run_brekv 8 0.95 0.75
  run_brekv 8 0.95 0.85
  run_brekv 16 0.95 0.90

  echo "######## pair7 TMATH GPU${GPU} DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "pair7 TMATH GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results root -> ${ROOT}/${TASK}"
