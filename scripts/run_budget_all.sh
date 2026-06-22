#!/bin/bash
# Step-1 预算分配:7 个数据集分到 GPU0 / GPU1 并行跑全套 budget 实验(WIN=16)。
#   每个数据集内部跑:uniform/layer (r=0.2/0.3/0.5) + query/query+layer (3 个区间)。
#   结果落 snapshots/<task>/budget/,逐样本含实际 budget。
# 用法:  bash scripts/run_budget_all.sh
#   之后:  python scripts/analyze_budget.py snapshots/hotpotqa snapshots/musique ...
set -u
export WIN=16

# GPU0 与 GPU1 的任务队列(按难度大致均衡)
GPU0_TASKS=(hotpotqa countries multifieldqa_en tmath)
GPU1_TASKS=(musique tipsheets twowikimqa)

run_queue() {
    local gpu=$1; shift
    for task in "$@"; do
        echo "######## [GPU${gpu}] START ${task} $(date '+%H:%M:%S') ########"
        TASK=${task} GPU=${gpu} bash scripts/run_budget.sh
        echo "######## [GPU${gpu}] DONE  ${task} $(date '+%H:%M:%S') ########"
    done
}

mkdir -p snapshots
run_queue 0 "${GPU0_TASKS[@]}" > snapshots/budget_all_gpu0.out 2>&1 &
P0=$!
run_queue 1 "${GPU1_TASKS[@]}" > snapshots/budget_all_gpu1.out 2>&1 &
P1=$!

echo "GPU0 (${GPU0_TASKS[*]}) -> snapshots/budget_all_gpu0.out  pid=${P0}"
echo "GPU1 (${GPU1_TASKS[*]}) -> snapshots/budget_all_gpu1.out  pid=${P1}"
wait ${P0} ${P1}
echo "==== ALL budget sweeps done ===="
