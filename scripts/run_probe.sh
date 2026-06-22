#!/bin/bash
# Step-0 探针:为 budget-headroom 验证重跑 receiver-w16 的 r 扫描(低预算区加密)。
#   会在 snapshots/${TASK}/mtc_receiver/ 下新建带时间戳的 run,并生成 per_sample.jsonl。
# 用法:  TASK=musique GPU=0 bash scripts/run_probe.sh
#   之后:  python scripts/analyze_oracle.py snapshots/${TASK} --method receiver_w16 --tau 0.5
set -e
TASK=${TASK:-musique}
GPU=${GPU:-0}
WIN=${WIN:-16}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/${TASK}/mtc_receiver
# 低预算区加密采样:难任务在 r<=0.3 处变化最剧烈,oracle 主要落在这里
for R in 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.7; do
    echo "==== [${TASK} | GPU ${GPU}] PROBE receiver win=${WIN} r=${R} ===="
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task ${TASK} --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --merge --merge_mode evict --score_mode receiver --recv_window ${WIN} \
        --merge_ratio ${R} --merge_sink 4 --merge_recent 8 \
        --snapshot_path ${OUT} --run_name probe_recv_w${WIN}_r${R}
done
echo "==== probe done. analyze with: python scripts/analyze_oracle.py snapshots/${TASK} --method receiver_w16 --tau 0.5 ===="
