#!/bin/bash
# KVComm layer-drop baselines + Full upper bound (top 1.0)
# usage:  TASK=2wikimqa GPU=0 bash scripts/run_baseline.sh
set -e
TASK=${TASK:-hotpotqa}
GPU=${GPU:-0}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/${TASK}/kvcomm
for T in 0.3 0.5 0.7 1.0; do
    echo "==== [${TASK} | GPU ${GPU}] KVComm top_layers=${T} ===="
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task ${TASK} --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --top_layers ${T} \
        --snapshot_path ${OUT} --run_name kvcomm_top${T}
done
