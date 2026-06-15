#!/bin/bash
# RASC: receiver-aware token selection w/ observation window (r=0.1..0.9) on hotpotqa
# usage:  bash scripts/run_receiver.sh            (默认 WIN=8)
#         WIN=16 bash scripts/run_receiver.sh     (换窗口)
set -e
GPU=${GPU:-0}
WIN=${WIN:-8}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/hotpotqa/mtc_receiver
for R in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "==== [GPU ${GPU}] receiver win=${WIN} r=${R} ===="
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task hotpotqa --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --merge --merge_mode evict --score_mode receiver --recv_window ${WIN} \
        --merge_ratio ${R} --merge_sink 4 --merge_recent 8 \
        --snapshot_path ${OUT} --run_name recv_w${WIN}_r${R}
done
