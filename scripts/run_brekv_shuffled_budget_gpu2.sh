#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-2}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/brekv_shuffled_budget_v1}
ANCHOR_ROOT=${ANCHOR_ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
PAIR_IDS=${PAIR_IDS:-"1 2 3 4 5 6 7"}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
LIMIT=${LIMIT:-0}
SEED=${SEED:-42}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}
INCLUDE_DEV=${INCLUDE_DEV:-0}

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

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

pair_slug() {
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
    mkdir -p "${view}"
    local file base
    for file in "${MODEL_A_5}"/*; do
      base=$(basename "${file}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors) continue ;;
      esac
      ln -sfn "${file}" "${view}/${base}"
    done
    echo "${view}"
  else
    echo "${MODEL_A_5}"
  fi
}

model_a() {
  case "$1" in
    1) echo "${MODEL_A_1}" ;;
    2) echo "${MODEL_A_2}" ;;
    3) echo "${MODEL_A_3}" ;;
    4) echo "${MODEL_A_4}" ;;
    5) pair5_runtime_model ;;
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) return 2 ;;
  esac
}

model_b() {
  case "$1" in
    1) echo "${MODEL_B_1}" ;;
    2) echo "${MODEL_B_2}" ;;
    3) echo "${MODEL_B_3}" ;;
    4) echo "${MODEL_B_4}" ;;
    5) echo "${MODEL_B_5}" ;;
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) return 2 ;;
  esac
}

is_dev_cell() {
  [[ "$1" == "1" && ( "$2" == "hotpotqa" || "$2" == "musique" ) ]]
}

anchor_path() {
  local slug=$1 task=$2
  local anchors=()
  mapfile -t anchors < <(
    compgen -G "${ANCHOR_ROOT}/${slug}/${task}/coverage/cov_t0.95_s0.75_w8_*/per_sample.jsonl" | sort
  )
  [[ ${#anchors[@]} -eq 1 ]] || {
    echo "Expected one canonical anchor for ${slug}/${task}, found ${#anchors[@]}" >&2
    return 2
  }
  echo "${anchors[0]}"
}

has_done_run() {
  local out=$1 run_name=$2 anchor=$3
  [[ "${SKIP_EXISTING}" == "1" ]] || return 1
  "${PYTHON}" - "${out}" "${run_name}" "${anchor}" "${LIMIT}" <<'PY'
import glob, json, os, sys
out, run_name, anchor, limit = sys.argv[1:]
paths = glob.glob(os.path.join(out, f"{run_name}_*", "per_sample.jsonl"))
if not paths:
    raise SystemExit(1)
path = max(paths, key=os.path.getmtime)
def read(path):
    with open(path) as handle:
        meta = json.loads(next(handle)).get("_meta", {})
        n = sum(1 for _ in handle)
    return meta, n
anchor_meta, anchor_n = read(anchor)
meta, n = read(path)
requested = int(limit)
expected = min(requested, anchor_n) if requested > 0 else anchor_n
valid = (
    n == expected
    and int(meta.get("n", -1)) == expected
    and meta.get("budget_replay_mode") == "shuffled"
)
raise SystemExit(0 if valid else 1)
PY
}

run_cell() {
  local pair=$1 task=$2
  local slug pa pb anchor out run_name
  slug=$(pair_slug "${pair}")
  pa=$(model_a "${pair}")
  pb=$(model_b "${pair}")
  anchor=$(anchor_path "${slug}" "${task}")
  out="${ROOT}/${slug}/${task}/shuffled_budget"
  run_name="brekv_shuffled_budget_t095_s075_w8"
  if has_done_run "${out}" "${run_name}" "${anchor}"; then
    echo "==== [skip] ${slug} ${task} shuffled-budget ===="
    return
  fi
  local args=(
    --test_task "${task}" --do_test --limit "${LIMIT}" --seed "${SEED}"
    --model_A "${pa}" --model_B "${pb}"
    --merge --merge_mode evict --merge_sink 4 --merge_recent 8
    --score_mode receiver --recv_window 8 --query_sketch_mode bf16
    --budget_mode uniform --merge_ratio 0.3
    --query_condition_mode correct
    --budget_replay_from "${anchor}"
    --budget_replay_mode shuffled
    --budget_replay_seed "${SEED}"
    --budget_replay_tolerance 0.001
    --snapshot_path "${out}" --run_name "${run_name}"
  )
  echo "==== [GPU${GPU}] ${slug} ${task} shuffled-budget $(date '+%F %T') ===="
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q com.py' "${GPU}" "${PYTHON}"
    printf ' %q' "${args[@]}"
    printf '\n'
  else
    mkdir -p "${out}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py "${args[@]}"
  fi
}

[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
for pair in ${PAIR_IDS}; do
  pa=$(model_a "${pair}")
  pb=$(model_b "${pair}")
  [[ -d "${pa}" ]] || { echo "Missing sender model: ${pa}" >&2; exit 2; }
  [[ -d "${pb}" ]] || { echo "Missing receiver model: ${pb}" >&2; exit 2; }
done

if [[ "${DRY_RUN}" == "1" ]]; then
  for pair in ${PAIR_IDS}; do
    for task in ${TASKS}; do
      if [[ "${INCLUDE_DEV}" != "1" ]] && is_dev_cell "${pair}" "${task}"; then
        continue
      fi
      run_cell "${pair}" "${task}"
    done
  done
  exit 0
fi

mkdir -p "${ROOT}/logs"
tag=$(date +"%m%d_%H%M")
log_file="${ROOT}/logs/gpu${GPU}_shuffled_budget_${tag}.log"
{
  echo "######## B-ReKV shuffled-budget START $(date '+%F %T') ########"
  echo "GPU=${GPU} PAIR_IDS=${PAIR_IDS} TASKS=${TASKS} LIMIT=${LIMIT} SEED=${SEED}"
  for pair in ${PAIR_IDS}; do
    for task in ${TASKS}; do
      if [[ "${INCLUDE_DEV}" != "1" ]] && is_dev_cell "${pair}" "${task}"; then
        continue
      fi
      run_cell "${pair}" "${task}"
    done
  done
  echo "######## B-ReKV shuffled-budget DONE $(date '+%F %T') ########"
} 2>&1 | tee "${log_file}"

echo "Results: ${ROOT}"
