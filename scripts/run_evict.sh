#!/bin/bash
# Token selection, drop-only, value-norm scoring (r=0.1..0.9)
# usage:  TASK=2wikimqa GPU=0 bash scripts/run_evict.sh
set -e
TASK=${TASK:-hotpotqa}
GPU=${GPU:-0}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/${TASK}/mtc_evict
for R in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "==== [${TASK} | GPU ${GPU}] evict(value-norm) r=${R} ===="
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task ${TASK} --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --merge --merge_mode evict --score_mode value_norm \
        --merge_ratio ${R} --merge_sink 4 --merge_recent 8 \
        --snapshot_path ${OUT} --run_name evict_r${R}
done
