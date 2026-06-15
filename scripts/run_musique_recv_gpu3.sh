#!/bin/bash
# musique receiver-aware 单独跑在 3 号卡 (从 run_musique.sh 的 GPU1 部分拆出来)
#   GPU 3: receiver w8(r0.1-0.9) + receiver w16(r0.1-0.9)
# 用法:  bash scripts/run_musique_recv_gpu3.sh
#   实时进度:  tail -f snapshots/musique/run_gpu3.out
set -e
export TASK=musique
mkdir -p snapshots/musique

echo "[GPU3] receiver w8 + w16          ->  snapshots/musique/run_gpu3.out"
( GPU=3 WIN=8  bash scripts/run_receiver.sh
  GPU=3 WIN=16 bash scripts/run_receiver.sh ) > snapshots/musique/run_gpu3.out 2>&1 &
P0=$!

wait ${P0}
echo "==== musique receiver done -> snapshots/musique/ ===="
