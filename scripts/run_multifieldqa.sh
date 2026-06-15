#!/bin/bash
# multifieldqa_en 全套方法,4/5 两张卡并行  (数据来自 LongBench,需先 bash scripts/download_datasets.sh)
#   GPU 4: baseline(top0.3/0.5/0.7/1.0) + merge(r0.1-0.9) + evict(r0.1-0.9)
#   GPU 5: receiver w8(r0.1-0.9) + receiver w16(r0.1-0.9)
# 用法:  bash scripts/run_multifieldqa.sh
#   实时进度:  tail -f snapshots/multifieldqa_en/run_gpu4.out   (或 run_gpu5.out)
set -e
export TASK=multifieldqa_en
mkdir -p snapshots/multifieldqa_en

echo "[GPU4] baseline + merge + evict   ->  snapshots/multifieldqa_en/run_gpu4.out"
( GPU=4 bash scripts/run_baseline.sh
  GPU=4 bash scripts/run_merge.sh
  GPU=4 bash scripts/run_evict.sh ) > snapshots/multifieldqa_en/run_gpu4.out 2>&1 &
P0=$!

echo "[GPU5] receiver w8 + w16          ->  snapshots/multifieldqa_en/run_gpu5.out"
( GPU=5 WIN=8  bash scripts/run_receiver.sh
  GPU=5 WIN=16 bash scripts/run_receiver.sh ) > snapshots/multifieldqa_en/run_gpu5.out 2>&1 &
P1=$!

wait ${P0} ${P1}
echo "==== multifieldqa_en done -> snapshots/multifieldqa_en/ ===="
