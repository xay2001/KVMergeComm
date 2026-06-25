#!/usr/bin/env bash
set -u

cd /home/xay/KVComm || exit 1

mkdir -p logs

# Clean failed/old queue logs. Snapshot results are not deleted.
rm -f logs/cov_countries.log \
      logs/cov_tipsheets.log \
      logs/cov_tmath.log \
      logs/cov_qasper.log \
      logs/cov_2wiki_table.log \
      logs/cov_hotpot_table_extra.log \
      logs/gpu2_table_queue.log \
      logs/gpu5_table_queue.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU2 queue: Countries -> Tipsheets -> 2WikiM-QA
(
  echo "######## [GPU2] Countries table coverage $(date '+%F %T') ########"
  TASK=countries GPU=2 WIN=8  TAUS="0.95" SCALES="0.75 0.85" bash scripts/run_coverage.sh
  TASK=countries GPU=2 WIN=16 TAUS="0.95" SCALES="0.90" bash scripts/run_coverage.sh

  echo "######## [GPU2] Tipsheets table coverage $(date '+%F %T') ########"
  TASK=tipsheets GPU=2 WIN=8  TAUS="0.95" SCALES="0.75 0.85" bash scripts/run_coverage.sh
  TASK=tipsheets GPU=2 WIN=16 TAUS="0.95" SCALES="0.90" bash scripts/run_coverage.sh

  echo "######## [GPU2] 2WikiM-QA table coverage $(date '+%F %T') ########"
  TASK=twowikimqa GPU=2 WIN=8  TAUS="0.95" SCALES="0.75 0.85" bash scripts/run_coverage.sh
  TASK=twowikimqa GPU=2 WIN=16 TAUS="0.95" SCALES="0.90" bash scripts/run_coverage.sh

  echo "######## [GPU2] DONE $(date '+%F %T') ########"
) > logs/gpu2_table_queue.log 2>&1 &

P2=$!

# GPU5 queue: QASPER -> TMATH -> HotpotQA extra
(
  echo "######## [GPU5] QASPER fixed RASC w16 first $(date '+%F %T') ########"
  TASK=qasper GPU=5 WIN=16 bash scripts/run_receiver.sh

  echo "######## [GPU5] QASPER table coverage $(date '+%F %T') ########"
  TASK=qasper GPU=5 WIN=8  TAUS="0.95" SCALES="0.75 0.85" bash scripts/run_coverage.sh
  TASK=qasper GPU=5 WIN=16 TAUS="0.95" SCALES="0.90" bash scripts/run_coverage.sh

  echo "######## [GPU5] TMATH table coverage $(date '+%F %T') ########"
  TASK=tmath GPU=5 WIN=8  TAUS="0.95" SCALES="0.75 0.85" bash scripts/run_coverage.sh
  TASK=tmath GPU=5 WIN=16 TAUS="0.95" SCALES="0.90" bash scripts/run_coverage.sh

  echo "######## [GPU5] HotpotQA missing table extra $(date '+%F %T') ########"
  TASK=hotpotqa GPU=5 WIN=8 TAUS="0.95" SCALES="0.85" bash scripts/run_coverage.sh

  echo "######## [GPU5] DONE $(date '+%F %T') ########"
) > logs/gpu5_table_queue.log 2>&1 &

P5=$!

echo "GPU2 queue pid=${P2} -> logs/gpu2_table_queue.log"
echo "GPU5 queue pid=${P5} -> logs/gpu5_table_queue.log"
