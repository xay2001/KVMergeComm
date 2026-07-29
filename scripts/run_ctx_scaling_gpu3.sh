#!/usr/bin/env bash
set -uo pipefail

# Context length x evidence sparsity: fixed question + gold evidence, distractor
# padding to {4K, 16K, 48K} tokens. Same-implementation comparison:
#   full_kv / kvcomm(top0.3, 4K-calibrated frozen layers) / recv_layer(0.3) / rekv(w8 r0.3)
#
# Usage: bash scripts/run_ctx_scaling_gpu3.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-3}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/sharedspace/models/Llama-3.1-8B-Instruct}
LENGTHS=${LENGTHS:-"4000 16000 48000"}
FRACTION=${FRACTION:-0.3}
WINDOW=${WINDOW:-8}
ROOT=${ROOT:-snapshots/ctx_scaling_v1}
DATA_DIR="${ROOT}/data"

timestamp=$(date +"%m%d_%H%M")
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu${GPU}_ctx_scaling_${timestamp}.log"

run_com() {
  local length=$1
  local method=$2
  local run_name=$3
  shift 3
  local parent="${ROOT}/ctx${length}/${method}"
  if compgen -G "${parent}/${run_name}_*/per_sample.jsonl" > /dev/null; then
    echo "[skip] ctx${length} ${method}"
    return
  fi
  echo "==== [GPU${GPU}] ctx${length} ${method} ===="
  SYNTH_JSONL_PATH="${DATA_DIR}/hotpotqa_ctx${length}.jsonl" \
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py \
    --test_task synthetic_jsonl \
    --model_A "${MODEL_A}" \
    --model_B "${MODEL_B}" \
    --layer_from 0 --layer_to 31 \
    --snapshot_path "${parent}" \
    --run_name "${run_name}" \
    "$@"
  if [ $? -ne 0 ]; then
    echo "[FAIL] ctx${length} ${method}"
  fi
}

extract_kvcomm_layers() {
  # Parse the frozen KVComm layer list from the 4K calibrated run log.
  local log
  log=$(ls -t "${ROOT}"/ctx4000/kvcomm/kvcomm_top${FRACTION}_*/log.log 2>/dev/null | head -1)
  [ -z "${log}" ] && return 1
  "${PYTHON}" - "$log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.findall(r"New layers list: \[([0-9,\s]+)\]", text)
if not m:
    raise SystemExit(1)
print(" ".join(x.strip() for x in m[-1].split(",")))
PY
}

{
  echo "######## CTX SCALING START $(date '+%F %T') ########"
  echo "LENGTHS=${LENGTHS} FRACTION=${FRACTION} WINDOW=${WINDOW}"

  # 4K KVComm first: calibration is cheap and safe at short context; the frozen
  # layer list is reused at longer lengths (KVComm protocol: tiny calibration).
  run_com 4000 kvcomm "kvcomm_top${FRACTION}" --do_test --top_layers "${FRACTION}" --calib_size 1

  KVCOMM_LAYERS=$(extract_kvcomm_layers)
  echo "KVComm frozen layers (from 4K calibration): ${KVCOMM_LAYERS}"

  for length in ${LENGTHS}; do
    run_com "${length}" skyline "skyline" --do_test_skyline
    run_com "${length}" full_kv "full_kv" --do_test

    if [ "${length}" != "4000" ] && [ -n "${KVCOMM_LAYERS}" ]; then
      run_com "${length}" kvcomm "kvcomm_top${FRACTION}" --do_test --layers_list ${KVCOMM_LAYERS}
    fi

    run_com "${length}" recv_layer "recv_layer_f${FRACTION}" --do_test \
      --receiver_layer_fraction "${FRACTION}" --score_mode receiver \
      --recv_window "${WINDOW}" --query_sketch_mode bf16

    run_com "${length}" rekv "rekv_w${WINDOW}_r${FRACTION}" --do_test \
      --merge --merge_mode evict --score_mode receiver \
      --recv_window "${WINDOW}" --query_sketch_mode bf16 \
      --merge_ratio "${FRACTION}"
  done

  echo "######## CTX SCALING DONE $(date '+%F %T') ########"
} 2>&1 | tee "${LOG_FILE}"
