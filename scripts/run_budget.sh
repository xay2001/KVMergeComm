#!/bin/bash
# Step-1 预算分配实验:在 receiver-w16 (ReKV) 基础上对比四种预算模式。
#   uniform      每层固定 merge_ratio(原版 ReKV,作为基线)
#   layer        总预算固定=merge_ratio,按层重要性 softmax 分配(等预算,只改分配)
#   query        每条 query 按重要性熵自适应总预算 [budget_min, budget_max],层间均匀
#   query+layer  自适应总预算 + 层间分配(最终方法)
# 全部用 evict(只丢不合并)+ receiver 打分,与已验证主方法一致。
# 每个 run 都会落 per_sample.jsonl(含实际 budget / query_budget),便于等预算对比。
#
# 用法:  TASK=hotpotqa GPU=0 bash scripts/run_budget.sh
#   之后:  python scripts/analyze_budget.py snapshots/${TASK}
set -e
TASK=${TASK:-hotpotqa}
GPU=${GPU:-0}
WIN=${WIN:-16}
MODEL=/sharedspace/models/Llama-3.1-8B-Instruct
OUT=snapshots/${TASK}/budget

common() {
    CUDA_VISIBLE_DEVICES=${GPU} python com.py \
        --test_task ${TASK} --do_test \
        --model_A ${MODEL} --model_B ${MODEL} \
        --merge --merge_mode evict --score_mode receiver --recv_window ${WIN} \
        --merge_sink 4 --merge_recent 8 \
        --snapshot_path ${OUT} "$@"
}

# --- A. 等预算:uniform vs layer(检验"层间分配"本身的收益)---
for R in 0.2 0.3 0.5; do
    echo "==== [${TASK} GPU${GPU}] uniform r=${R} ===="
    common --budget_mode uniform --merge_ratio ${R} --run_name uniform_r${R}
    echo "==== [${TASK} GPU${GPU}] layer  r=${R} ===="
    common --budget_mode layer   --merge_ratio ${R} --budget_tau 1.0 --run_name layer_r${R}
done

# --- B. query 自适应总预算(检验"每条 query 不该同预算")---
# 用不同 [min,max] 区间扫出几个平均预算点,和 uniform 曲线做等预算对比
for RANGE in "0.05 0.3" "0.1 0.5" "0.15 0.7"; do
    set -- ${RANGE}; LO=$1; HI=$2
    echo "==== [${TASK} GPU${GPU}] query [${LO},${HI}] ===="
    common --budget_mode query       --budget_min ${LO} --budget_max ${HI} --run_name query_${LO}_${HI}
    echo "==== [${TASK} GPU${GPU}] query+layer [${LO},${HI}] ===="
    common --budget_mode query+layer --budget_min ${LO} --budget_max ${HI} --budget_tau 1.0 --run_name querylayer_${LO}_${HI}
done

echo "==== budget sweep done -> ${OUT} ===="
