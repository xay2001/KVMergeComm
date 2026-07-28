#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-3}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/receiver_conditioning_v1}
ANCHOR_ROOT=${ANCHOR_ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
PAIRS=${PAIRS:-"1 6 7"}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
VARIANTS=${VARIANTS:-"correct shuffled unrelated sender_text_receiver_encoder sender_context_q query_free"}
LIMIT=${LIMIT:-0}
SEED=${SEED:-42}
RECV_WINDOW=${RECV_WINDOW:-8}
BUDGET_TOLERANCE=${BUDGET_TOLERANCE:-0.001}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}

MODEL_A_1=${MODEL_A_1:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_B_1=${MODEL_B_1:-${MODEL_A_1}}
MODEL_A_6=${MODEL_A_6:-/NAS/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/NAS/models/DeepSeek-R1-Distill-Llama-3B}
MODEL_A_7=${MODEL_A_7:-/NAS/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/NAS/models/Bespoke-Stratos-7B}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

pair_slug() {
  case "$1" in
    1) echo "pair1_llama31_same" ;;
    6) echo "pair6_llama32_abliterated_deepseek3b" ;;
    7) echo "pair7_qwen25_uncensored_bespoke" ;;
    *) echo "Unknown pair id: $1" >&2; return 2 ;;
  esac
}

model_a() {
  case "$1" in
    1) echo "${MODEL_A_1}" ;;
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) return 2 ;;
  esac
}

model_b() {
  case "$1" in
    1) echo "${MODEL_B_1}" ;;
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) return 2 ;;
  esac
}

unrelated_task() {
  case "$1" in
    hotpotqa) echo "musique" ;;
    musique) echo "multifieldqa_en" ;;
    multifieldqa_en) echo "hotpotqa" ;;
    *) echo "No unrelated-task mapping for $1" >&2; return 2 ;;
  esac
}

anchor_path() {
  local slug=$1 task=$2
  local pattern="${ANCHOR_ROOT}/${slug}/${task}/coverage/cov_t0.95_s0.75_w8_*/per_sample.jsonl"
  local anchors=()
  mapfile -t anchors < <(compgen -G "${pattern}" | sort)
  if [[ ${#anchors[@]} -ne 1 ]]; then
    echo "Expected exactly one canonical budget anchor for ${slug}/${task}, found ${#anchors[@]}" >&2
    return 2
  fi
  echo "${anchors[0]}"
}

has_done_run() {
  local out=$1 run_name=$2 anchor=$3 variant=$4
  [[ "${SKIP_EXISTING}" == "1" ]] || return 1
  "${PYTHON}" - "${out}" "${run_name}" "${anchor}" "${variant}" "${LIMIT}" <<'PY'
import glob
import json
import os
import sys

out, run_name, anchor, variant, limit = sys.argv[1:]
paths = glob.glob(os.path.join(out, f"{run_name}_*", "per_sample.jsonl"))
if not paths:
    raise SystemExit(1)
path = max(paths, key=os.path.getmtime)

def meta_and_n(filename):
    with open(filename) as handle:
        first = json.loads(next(handle))
        rows = sum(1 for _ in handle)
    return first.get("_meta", {}), rows

anchor_meta, anchor_n = meta_and_n(anchor)
run_meta, run_n = meta_and_n(path)
requested = int(limit)
expected_n = min(requested, anchor_n) if requested > 0 else anchor_n
valid = (
    run_n == expected_n
    and int(run_meta.get("n", -1)) == expected_n
    and run_meta.get("query_condition_mode") == variant
)
raise SystemExit(0 if valid else 1)
PY
}

run_variant() {
  local pair=$1 task=$2 variant=$3
  local slug pa pb anchor out run_name score_mode unrelated
  slug=$(pair_slug "${pair}")
  pa=$(model_a "${pair}")
  pb=$(model_b "${pair}")
  anchor=$(anchor_path "${slug}" "${task}")
  out="${ROOT}/${slug}/${task}/${variant}"
  run_name="rc_${variant}_replay_t095_s075_w${RECV_WINDOW}"
  score_mode="receiver"
  unrelated=""
  if [[ "${variant}" == "query_free" ]]; then
    score_mode="value_norm"
  elif [[ "${variant}" == "unrelated" ]]; then
    unrelated=$(unrelated_task "${task}")
  fi

  if has_done_run "${out}" "${run_name}" "${anchor}" "${variant}"; then
    echo "==== [skip] ${slug} ${task} ${variant} ===="
    return
  fi

  local args=(
    --test_task "${task}"
    --do_test
    --limit "${LIMIT}"
    --seed "${SEED}"
    --model_A "${pa}"
    --model_B "${pb}"
    --merge
    --merge_mode evict
    --merge_sink 4
    --merge_recent 8
    --score_mode "${score_mode}"
    --recv_window "${RECV_WINDOW}"
    --query_sketch_mode bf16
    --budget_mode uniform
    --merge_ratio 0.3
    --query_condition_mode "${variant}"
    --query_condition_seed "${SEED}"
    --budget_replay_from "${anchor}"
    --budget_replay_tolerance "${BUDGET_TOLERANCE}"
    --snapshot_path "${out}"
    --run_name "${run_name}"
  )
  if [[ -n "${unrelated}" ]]; then
    args+=(--query_unrelated_task "${unrelated}")
  fi

  echo "==== [GPU${GPU}] ${slug} ${task} ${variant} anchor=${anchor} $(date '+%F %T') ===="
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q com.py' "${GPU}" "${PYTHON}"
    printf ' %q' "${args[@]}"
    printf '\n'
  else
    mkdir -p "${out}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py "${args[@]}"
  fi
}

validate_inputs() {
  [[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
  local pair task pa pb slug
  for pair in ${PAIRS}; do
    slug=$(pair_slug "${pair}")
    pa=$(model_a "${pair}")
    pb=$(model_b "${pair}")
    [[ -d "${pa}" ]] || { echo "Missing sender model: ${pa}" >&2; exit 2; }
    [[ -d "${pb}" ]] || { echo "Missing receiver model: ${pb}" >&2; exit 2; }
    for task in ${TASKS}; do
      anchor_path "${slug}" "${task}" > /dev/null
    done
  done
}

validate_inputs
if [[ "${DRY_RUN}" == "1" ]]; then
  for pair in ${PAIRS}; do
    for task in ${TASKS}; do
      for variant in ${VARIANTS}; do
        run_variant "${pair}" "${task}" "${variant}"
      done
    done
  done
  exit 0
fi

mkdir -p "${ROOT}/logs"
tag=$(date +"%m%d_%H%M")
log_file="${ROOT}/logs/gpu${GPU}_receiver_conditioning_${tag}.log"

{
  echo "######## Receiver conditioning START $(date '+%F %T') ########"
  echo "GPU=${GPU} PAIRS=${PAIRS} TASKS=${TASKS} VARIANTS=${VARIANTS} LIMIT=${LIMIT}"
  for pair in ${PAIRS}; do
    for task in ${TASKS}; do
      for variant in ${VARIANTS}; do
        run_variant "${pair}" "${task}" "${variant}"
      done
    done
  done
  "${PYTHON}" scripts/analyze_receiver_conditioning.py \
    --root "${ROOT}" \
    --out-dir "${ROOT}/analysis"
  echo "######## Receiver conditioning DONE $(date '+%F %T') ########"
} 2>&1 | tee "${log_file}"

echo "Results: ${ROOT}"
