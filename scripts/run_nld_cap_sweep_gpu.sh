#!/usr/bin/env bash
set -euo pipefail

# Receiver-aware NLD message-length sweep with paired cost profiling.
# Usage: GPU=7 bash scripts/run_nld_cap_sweep_gpu.sh

cd "$(dirname "$0")/.." || exit 1

GPU=${GPU:?set GPU}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
PAIRS=${PAIRS:-"1 7"}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
CAPS=${CAPS:-"64 128 256 512"}
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
ROOT=${ROOT:-snapshots/nld_cap_sweep_v1}

pair_paths() {
  case "$1" in
    1) echo "pair1_llama31_same /sharedspace/models/Llama-3.1-8B-Instruct /sharedspace/models/Llama-3.1-8B-Instruct" ;;
    7) echo "pair7_qwen25_uncensored_bespoke /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored /sharedspace/models/Bespoke-Stratos-7B" ;;
    *) echo "unsupported pair $1" >&2; return 2 ;;
  esac
}

mkdir -p "${ROOT}/logs"
LOG_FILE="${ROOT}/logs/gpu${GPU}_nld_cap_sweep_$(date +%m%d_%H%M).log"

run_one() {
  local pair_slug=$1 model_a=$2 model_b=$3 task=$4 cap=$5
  local parent="${ROOT}/${pair_slug}/${task}/cap${cap}"
  if compgen -G "${parent}/ra_nld_cap${cap}_*/cost_summary.json" >/dev/null; then
    echo "[skip] ${pair_slug} ${task} cap=${cap}"
    return
  fi
  echo "[GPU${GPU}] ${pair_slug} ${task} RA-NLD cap=${cap}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test_nld --nld_receiver_aware \
    --profile_cost --profile_limit "${PROFILE_LIMIT}" \
    --profile_warmup "${PROFILE_WARMUP}" \
    --nld_max_tokens_model_A_and_B_phase1 "${cap}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --snapshot_path "${parent}" --run_name "ra_nld_cap${cap}"
}

{
  echo "RA-NLD cap sweep: GPU=${GPU} PAIRS=${PAIRS} TASKS=${TASKS} CAPS=${CAPS}"
  for pair in ${PAIRS}; do
    read -r pair_slug model_a model_b <<< "$(pair_paths "${pair}")"
    for task in ${TASKS}; do
      for cap in ${CAPS}; do
        run_one "${pair_slug}" "${model_a}" "${model_b}" "${task}" "${cap}"
      done
    done
  done
} 2>&1 | tee "${LOG_FILE}"
