#!/usr/bin/env bash
set -euo pipefail

# Small raw-response probe for Table 8 pair #9 near-zero behavior.
# It saves question / gold answers / raw response / score for manual diagnosis.

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
LIMIT=${LIMIT:-5}
TASKS=${TASKS:-"countries tipsheets hotpotqa musique qasper"}
OUT_DIR=${OUT_DIR:-snapshots/diagnostics/pair9_outputs_gpu${GPU}}

CUDA_VISIBLE_DEVICES=${GPU} python scripts/diagnose_pair9_outputs.py \
  --device cuda:0 \
  --model_a /sharedspace/models/Llama-3.1-SuperNova-Lite \
  --model_b /sharedspace/models/DeepSeek-R1-Distill-Llama-8B \
  --tasks ${TASKS} \
  --limit "${LIMIT}" \
  --out_dir "${OUT_DIR}"
