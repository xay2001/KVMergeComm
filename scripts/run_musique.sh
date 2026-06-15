#!/bin/bash
# musique 全套方法,0/1 两张卡并行  (需先 bash scripts/download_datasets.sh 下载 dgslibisey/MuSiQue)
#   GPU 0: baseline(top0.3/0.5/0.7/1.0) + merge(r0.1-0.9) + evict(r0.1-0.9)
#   GPU 1: receiver w8(r0.1-0.9) + receiver w16(r0.1-0.9)
# 用法:  bash scripts/run_musique.sh
#   实时进度:  tail -f snapshots/musique/run_gpu0.out   (或 run_gpu1.out)
set -e
export TASK=musique
mkdir -p snapshots/musique

echo "[GPU0] baseline + merge + evict   ->  snapshots/musique/run_gpu0.out"
( GPU=0 bash scripts/run_baseline.sh
  GPU=0 bash scripts/run_merge.sh
  GPU=0 bash scripts/run_evict.sh ) > snapshots/musique/run_gpu0.out 2>&1 &
P0=$!

echo "[GPU1] receiver w8 + w16          ->  snapshots/musique/run_gpu1.out"
( GPU=1 WIN=8  bash scripts/run_receiver.sh
  GPU=1 WIN=16 bash scripts/run_receiver.sh ) > snapshots/musique/run_gpu1.out 2>&1 &
P1=$!

wait ${P0} ${P1}
echo "==== musique done -> snapshots/musique/ ===="
