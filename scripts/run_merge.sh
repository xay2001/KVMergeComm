#!/bin/bash
# MtC normalized value-merge sweep (r=0.1..0.9) on hotpotqa
# usage:  bash scripts/run_merge.sh
set -e
GPU=${GPU:-0}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/hotpotqa/mtc_merge
for R in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "==== [GPU ${GPU}] merge r=${R} ===="
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task hotpotqa --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --merge --merge_mode merge --merge_ratio ${R} --merge_sink 4 --merge_recent 8 \
        --snapshot_path ${OUT} --run_name merge_r${R}
done
