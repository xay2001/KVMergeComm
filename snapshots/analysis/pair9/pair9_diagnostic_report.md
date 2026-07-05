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
- 现有 per-sample 文件只能确认分数异常，不能确认是 prompt/template、chat special tokens、模型输出格式，还是模型能力/对齐问题。
- 下一步应抽样重跑少量样本并保存 raw prompt / raw response / parsed answer，用于定位异常来源。
