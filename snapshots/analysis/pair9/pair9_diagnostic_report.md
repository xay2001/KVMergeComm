# Pair #9 异常诊断

基于现有 `per_sample.jsonl` 的分数分布诊断；这些文件不保存模型原始回答，因此输出文本诊断需要额外抽样重跑。

## 数据集级摘要

| Dataset | Runs | Mean of run means | Best run mean | Runs near zero (<=0.01) |
|---|---:|---:|---:|---:|
| 2wikimqa | 9 | 0.0050 | 0.0050 | 9 |
| countries | 9 | 0.0000 | 0.0000 | 9 |
| hotpotqa | 9 | 0.0000 | 0.0000 | 9 |
| multifieldqa_en | 9 | 0.0148 | 0.0200 | 0 |
| musique | 9 | 0.0000 | 0.0000 | 9 |
| qasper | 9 | 0.0011 | 0.0020 | 9 |
| tipsheets | 9 | 0.0000 | 0.0000 | 9 |
| tmath | 9 | 0.3246 | 0.3279 | 0 |

## 结论

- Pair #9 的异常不是单个 run 的偶发问题；多个非 `tmath` 数据集上大量 run 的均分接近 0。
- Raw-output probe 已完成。输出不是乱码或明显 template 污染，`as_run_local_path` 与 `forced_deepseek_think` 输出基本一致；问题更像 evidence grounding / multi-hop reasoning 不稳定。
- KVComm top=0.3 limit=50 probe 也很差：`countries=0.120`, `tipsheets=0.560`, `hotpotqa=0.060`, `musique=0.020`, `qasper=0.000`。这说明 near-zero 不是 ReKV/B-ReKV 独有问题，而是 SuperNova -> DeepSeek-Llama-8B 这个 hard heterogeneous pair 的 KV compatibility / long-context grounding 问题。
- 处理决定：pair #9 暂缓作为正向对比，不再继续投入 GPU 补跑；论文中最多作为 hard negative / limitation 附录说明。
