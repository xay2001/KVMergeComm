#!/bin/bash
# twowikimqa(2WikiMQA)全套方法,1/7 两张卡并行  (数据来自 LongBench)
#   GPU 1: baseline(top0.3/0.5/0.7/1.0) + merge(r0.1-0.9) + evict(r0.1-0.9)
#   GPU 7: receiver w8(r0.1-0.9) + receiver w16(r0.1-0.9)
# 用法:  bash scripts/run_twowikimqa.sh
#   实时进度:  tail -f snapshots/twowikimqa/run_gpu1.out   (或 run_gpu7.out)
set -e
export TASK=twowikimqa
mkdir -p snapshots/twowikimqa

echo "[GPU1] baseline + merge + evict   ->  snapshots/twowikimqa/run_gpu1.out"
( GPU=1 bash scripts/run_baseline.sh
  GPU=1 bash scripts/run_merge.sh
  GPU=1 bash scripts/run_evict.sh ) > snapshots/twowikimqa/run_gpu1.out 2>&1 &
P0=$!

echo "[GPU7] receiver w8 + w16          ->  snapshots/twowikimqa/run_gpu7.out"
( GPU=7 WIN=8  bash scripts/run_receiver.sh
  GPU=7 WIN=16 bash scripts/run_receiver.sh ) > snapshots/twowikimqa/run_gpu7.out 2>&1 &
P1=$!

wait ${P0} ${P1}
echo "==== twowikimqa done -> snapshots/twowikimqa/ ===="
