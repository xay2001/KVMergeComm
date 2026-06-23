#!/bin/bash
# 牌2 Step-3: dump single-shot budget-prediction features (Pass-1 only, no generation).
#   Fast (prefill + receiver scoring, no decode). Output: per_sample_feat.jsonl.
#   Must use the SAME dataset default N as the probe runs so idx aligns with the
#   oracle labels in snapshots/<task>/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl.
# 用法:  TASK=hotpotqa GPU=2 bash scripts/run_features.sh
set -e
TASK=${TASK:-hotpotqa}
GPU=${GPU:-2}
WIN=${WIN:-16}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/${TASK}/features

echo "==== [${TASK} GPU${GPU}] pass1 feature dump (win=${WIN}) ===="
CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task ${TASK} --do_test \
    --model_A ${MODEL} --model_B ${MODEL} \
    --merge --merge_mode evict --score_mode receiver --recv_window ${WIN} \
    --merge_sink 4 --merge_recent 8 \
    --dump_pass1_features \
    --snapshot_path ${OUT} --run_name feat_w${WIN}

echo "==== done -> ${OUT}.  train predictor with:"
echo "  python scripts/learn_budget_predictor.py --tasks ${TASK} ===="
