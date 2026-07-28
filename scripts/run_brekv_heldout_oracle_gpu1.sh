#!/usr/bin/env bash
set -euo pipefail

# Add only the r=0.70 fixed-ReKV oracle points missing from the 1568-run sweep.
# Resumable: a logical run is complete once its per_sample.jsonl exists.

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-1}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/brekv_oracle_r07_v1}
ANCHOR_ROOT=${ANCHOR_ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
LIMIT=${LIMIT:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
PAIR_IDS=${PAIR_IDS:-"1 2 3 4 5 6 7"}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
RUN_ITEMS=${RUN_ITEMS:-}
INCLUDE_DEV=${INCLUDE_DEV:-0}
RATIO=${RATIO:-0.70}
WINDOW=${WINDOW:-8}

MODEL_A_1=${MODEL_A_1:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_B_1=${MODEL_B_1:-${MODEL_A_1}}
MODEL_A_2=${MODEL_A_2:-/NAS/models/Llama-3.2-3B-Instruct}
MODEL_B_2=${MODEL_B_2:-${MODEL_A_2}}
MODEL_A_3=${MODEL_A_3:-/NAS/models/Qwen2.5-7B-Instruct}
MODEL_B_3=${MODEL_B_3:-${MODEL_A_3}}
MODEL_A_4=${MODEL_A_4:-/NAS/models/Falcon3-7B-Instruct}
MODEL_B_4=${MODEL_B_4:-${MODEL_A_4}}
MODEL_A_5=${MODEL_A_5:-/NAS/models/EvolCodeLlama-3.1-8B-Instruct}
MODEL_B_5=${MODEL_B_5:-/NAS/models/ToolACE-2-Llama-3.1-8B}
MODEL_A_6=${MODEL_A_6:-/NAS/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/NAS/models/DeepSeek-R1-Distill-Llama-3B}
MODEL_A_7=${MODEL_A_7:-/NAS/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/NAS/models/Bespoke-Stratos-7B}

pair_name() {
  case "$1" in
    1) echo "pair1_llama31_same" ;;
    2) echo "pair2_llama32_same" ;;
    3) echo "pair3_qwen25_7b_same" ;;
    4) echo "pair4_falcon3_7b_same" ;;
    5) echo "pair5_evolcodellama_toolace" ;;
    6) echo "pair6_llama32_abliterated_deepseek3b" ;;
    7) echo "pair7_qwen25_uncensored_bespoke" ;;
    *) echo "Unknown pair id: $1" >&2; return 2 ;;
  esac
}

pair5_runtime_model() {
  local view="${ROOT}/model_views/$(basename "${MODEL_A_5}")_no_adapter"
  if [[ -f "${MODEL_A_5}/adapter_config.json" && -f "${MODEL_A_5}/model.safetensors.index.json" ]]; then
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${view}"
      local file base
      for file in "${MODEL_A_5}"/*; do
        base=$(basename "${file}")
        case "${base}" in
          adapter_config.json|adapter_model.bin|adapter_model.safetensors) continue ;;
        esac
        ln -sfn "${file}" "${view}/${base}"
      done
    fi
    echo "${view}"
  else
    echo "${MODEL_A_5}"
  fi
}

model_a() {
  case "$1" in
    1) echo "${MODEL_A_1}" ;; 2) echo "${MODEL_A_2}" ;;
    3) echo "${MODEL_A_3}" ;; 4) echo "${MODEL_A_4}" ;;
    5) pair5_runtime_model ;; 6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;; *) return 2 ;;
  esac
}

model_b() {
  case "$1" in
    1) echo "${MODEL_B_1}" ;; 2) echo "${MODEL_B_2}" ;;
    3) echo "${MODEL_B_3}" ;; 4) echo "${MODEL_B_4}" ;;
    5) echo "${MODEL_B_5}" ;; 6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;; *) return 2 ;;
  esac
}

iter_items() {
  local pair task item
  if [[ -n "${RUN_ITEMS}" ]]; then
    for item in ${RUN_ITEMS}; do echo "${item}"; done
  else
    for pair in ${PAIR_IDS}; do
      for task in ${TASKS}; do
        if [[ "${INCLUDE_DEV}" != "1" && "${pair}" == "1" &&
              ( "${task}" == "hotpotqa" || "${task}" == "musique" ) ]]; then
          continue
        fi
        echo "${pair}:${task}"
      done
    done
  fi
}

run_item() {
  local pair=$1 task=$2 pa pb out run_name anchor
  pa=$(model_a "${pair}")
  pb=$(model_b "${pair}")
  out="${ROOT}/$(pair_name "${pair}")/${task}/fairness_rekv"
  run_name="rekv_w${WINDOW}_r${RATIO}"
  local anchor_paths=()
  mapfile -t anchor_paths < <(
    compgen -G "${ANCHOR_ROOT}/$(pair_name "${pair}")/${task}/coverage/cov_t0.95_s0.75_w8_*/per_sample.jsonl" | sort
  )
  [[ ${#anchor_paths[@]} -eq 1 ]] || {
    echo "Expected one canonical anchor for pair${pair}/${task}, found ${#anchor_paths[@]}" >&2
    return 2
  }
  anchor="${anchor_paths[0]}"
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    local existing=""
    local existing_paths=()
    mapfile -t existing_paths < <(
      compgen -G "${out}/${run_name}_*/per_sample.jsonl" | sort
    )
    if [[ ${#existing_paths[@]} -gt 0 ]]; then
      existing="${existing_paths[$((${#existing_paths[@]} - 1))]}"
    fi
    if [[ -n "${existing}" ]] && "${PYTHON}" - "${existing}" "${anchor}" "${LIMIT}" <<'PY'
import json, sys
path, anchor, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(anchor) as handle:
    next(handle)
    anchor_n = sum(1 for _ in handle)
with open(path) as handle:
    meta = json.loads(next(handle)).get("_meta", {})
    n = sum(1 for _ in handle)
valid = n > 0 and int(meta.get("n", -1)) == n
if limit > 0:
    valid = valid and n == min(limit, anchor_n)
else:
    valid = valid and n == anchor_n
raise SystemExit(0 if valid else 1)
PY
    then
      echo "[skip] pair${pair} ${task} ${run_name}"
      return
    fi
  fi
  local -a command=(
    env "CUDA_VISIBLE_DEVICES=${GPU}" "${PYTHON}" com.py
    --test_task "${task}" --do_test --limit "${LIMIT}"
    --model_A "${pa}" --model_B "${pb}"
    --merge --merge_mode evict --merge_sink 4 --merge_recent 8
    --snapshot_path "${out}" --run_name "${run_name}"
    --score_mode receiver --query_sketch_mode bf16
    --recv_window "${WINDOW}" --merge_ratio "${RATIO}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    mkdir -p "${out}"
    echo "[GPU${GPU}] pair${pair} ${task} ${run_name} $(date '+%F %T')"
    "${command[@]}"
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
  for pair in ${PAIR_IDS}; do
    [[ -d "$(model_a "${pair}")" ]] || { echo "Missing model A for pair${pair}" >&2; exit 2; }
    [[ -d "$(model_b "${pair}")" ]] || { echo "Missing model B for pair${pair}" >&2; exit 2; }
  done
  mkdir -p "${ROOT}/logs"
fi

echo "B-ReKV held-out oracle r=${RATIO}; GPU=${GPU}; LIMIT=${LIMIT}; DRY_RUN=${DRY_RUN}"
while IFS=: read -r pair task; do
  [[ -n "${pair}" && -n "${task}" ]] || continue
  run_item "${pair}" "${task}"
done < <(iter_items)
echo "Oracle r=${RATIO} sweep complete: ${ROOT}"
