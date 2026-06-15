#!/bin/bash
# Download eval datasets into ./datasets/<name> via hfd.sh + hf-mirror (offline loading).
# dataloader/local_loader.py 会优先用 datasets/<name>,没有才联网。
# usage:  bash scripts/download_datasets.sh
set -e
cd /home/xay/KVComm
export HF_ENDPOINT=https://hf-mirror.com

# 2wikimqa + multifieldqa_en (同一仓库的不同 config)
./datasets/hfd.sh Xnhyacinth/LongBench --dataset --local-dir datasets/LongBench

# musique
./datasets/hfd.sh dgslibisey/MuSiQue --dataset --local-dir datasets/MuSiQue

# qasper (需要 trust_remote_code,仓库自带加载脚本)
./datasets/hfd.sh tau/scrolls --dataset --local-dir datasets/scrolls

echo "==== done. datasets/{LongBench,MuSiQue,scrolls} ready ===="
