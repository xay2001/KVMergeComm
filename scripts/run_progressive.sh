#!/bin/bash
# Step-2b: online progressive communication.
#   For each sample, B generates at EVERY budget rung and records
#   {score, budget, uncertainty(entropy/margin)} -> per_sample_prog.jsonl.
#   The stop-threshold theta is then swept offline (no extra GPU):
#       python scripts/analyze_progressive_online.py snapshots/<task>/progressive --tau 0.5
#
# Cost ~ (#rungs) x a normal eval pass. Keep the ladder small (4 rungs default).
# 用法:  TASK=hotpotqa GPU=0 bash scripts/run_progressive.sh
#        TASK=musique GPU=1 LADDER=0.05,0.1,0.2,0.3,0.5 bash scripts/run_progressive.sh
set -e
TASK=${TASK:-hotpotqa}
GPU=${GPU:-0}
WIN=${WIN:-16}
LADDER=${LADDER:-0.1,0.2,0.3,0.5}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/${TASK}/progressive

echo "==== [${TASK} GPU${GPU}] progressive ladder=${LADDER} win=${WIN} ===="
CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task ${TASK} --do_test \
    --model_A ${MODEL} --model_B ${MODEL} \
    --merge --merge_mode evict --score_mode receiver --recv_window ${WIN} \
    --merge_sink 4 --merge_recent 8 \
    --progressive --prog_ladder ${LADDER} \
    --snapshot_path ${OUT} --run_name prog_w${WIN}_$(echo ${LADDER} | tr ',' '-')

echo "==== done -> ${OUT}.  analyze with:"
echo "  python scripts/analyze_progressive_online.py ${OUT} --tau 0.5 ===="
