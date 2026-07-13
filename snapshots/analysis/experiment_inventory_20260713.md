# 实验结果总盘点（2026-07-13）

本文件是当前实验数据的协议感知入口。所有结果先按协议代际分组，再讨论
完成度和数值；禁止仅凭目录名或 `score_mode=receiver` 判断为 Query-Sketch。

机器可读清单：`snapshots/analysis/experiment_inventory_20260713.csv`。

## 1. 口径：必须区分的五类结果

| 协议标签 | 识别规则 | 允许用途 |
|---|---|---|
| `query_sketch_bf16_v1` / `int8_v1` / `token_ids_v1` | `_meta.protocol_version` 明确存在，并记录 sketch/通信字段 | 当前可部署单源主协议、准确率、预算、通信量 |
| `query_sketch_bf16_multi_source_v1` | Multi-Source 中 B 发 Q sketch，A1/A2 各自在本地 Q×K 后压缩 | 新 Table 10 主协议 |
| `full_kv_oracle_v1` / `full_kv_oracle_multi_source_v1` | Receiver 在选择前获得 full KV | 仅作 oracle upper bound |
| `query_agnostic_kv_v1` | ValueNorm/Evict、Random | Query-aware fairness 对照 |
| `legacy_implicit_receiver_v0` | 历史 `score_mode=receiver`，但无协议和通信元数据 | 历史准确率、原型证据；不能称为新 Query-Sketch |

另有 `query_sketch_bf16_v0_pre_instrumentation`：结果位于 Query-Sketch
目录但缺少 v1 元数据，只能作准确率参考，不能作通信成本证据。

## 2. 当前总状态

- 全仓库有 2,074 个带 `per_sample.jsonl` 的结果目录。
- 另有 691 个只有 `log.log`、没有 `per_sample.jsonl` 的目录；其中许多是
  cost-only/profile 设计，也包含部分中断和早期 sweep，不能统一视作失败。
- 619/2,074 个结果明确记录 `protocol_version`；其余 1,455 个主要是历史实验。
- 当前没有实验 worker 在运行。
- Query-Sketch 配置冻结和 Stage 3 核心审稿矩阵已经完成。
- Stage 5 的 Table 6 / Table 8 / Table 10 Query-Sketch 重跑尚未启动。

## 3. 新协议：已经完成的核心证据

### 3.1 全局 B-ReKV 配置冻结

Root：`snapshots/query_sketch_config_freeze/`

- 逻辑矩阵：2 pairs × 3 tasks ×（5 fixed ReKV + 6 B-ReKV）= 66/66。
- 物理文件：131 个，包含两轮重复运行；分析脚本按 logical config 取最新结果。
- 每个 run 使用相同的前 100 个样本。
- 全部结果为 `query_sketch_bf16_v1`。
- 最终接受的冻结配置：`B-ReKV-t0.98-s1-w8`。
- 平均实际预算：0.5919。
- 相对 matched-budget fixed ReKV：平均分差 +0.0250，最差 +0.0100，
  6/6 单元持平或获胜。

注意：这里“冻结完成”指配置选择完成，不表示该配置已经在 Table 1 的
3 pairs × 8 tasks 主矩阵跑完；后者目前仍是 0/24。

### 3.2 Stage 3：Query-aware fairness、Pareto 和预算分布

Root：`snapshots/stage3_core_reviewer_query_sketch/`

- 360/360 runs 完成，原始结果完整。
- 234 个 receiver-aware run 为 `query_sketch_bf16_v1`。
- 126 个 ValueNorm/Evict、Random run 为 `query_agnostic_kv_v1`。
- 覆盖 pair #1/#6/#7 与 `hotpotqa`、`musique`、`multifieldqa_en`。
- 已重建完整分析：
  - `analysis/all_runs.csv`：360 行；
  - `analysis/query_fairness_matched_budget.csv`：27 行；
  - `analysis/brekv_accuracy_budget_pareto.csv`：162 行；
  - `analysis/brekv_budget_distribution.csv`：9 行；
  - `analysis/calibrated_vs_strict_adaptive.csv`：9 行。

#### Matched-budget fairness

Stage 3 的 canonical 比较点是 `t=0.95, scale=0.75, w=8`，它是用于
fairness/Pareto 的统一观察点，和最终冻结的 `t=0.98, scale=1.0, w=8`
不是同一个配置。

Canonical B-ReKV 在 9 个 pair-task 单元上：

- 相对 ValueNorm/Evict：9/9 获胜；
- 相对 Random：9/9 获胜；
- 相对 matched-budget fixed ReKV：3/9 获胜、6/9 落后。

因此可以稳健声称“query-aware selection 明显优于 query-agnostic
ValueNorm/Random”；不能声称 canonical B-ReKV 在所有 pair/task 都优于
matched-budget fixed ReKV。B-ReKV 的主要价值是动态预算和 Pareto，而不是
每个单元都提升绝对分数。

代表性 matched-budget 分差（B-ReKV − baseline）：

| Pair / Task | vs ReKV | vs Evict | vs Random | B-ReKV 实际预算 |
|---|---:|---:|---:|---:|
| #1 HotpotQA | -0.0122 | +0.0367 | +0.1992 | 0.2754 |
| #1 MultiFieldQA | +0.0067 | +0.1715 | +0.1003 | 0.3354 |
| #6 MuSiQue | +0.0263 | +0.1115 | +0.1725 | 0.2519 |
| #7 HotpotQA | -0.0326 | +0.1827 | +0.2059 | 0.3853 |
| #7 MultiFieldQA | -0.0759 | +0.2189 | +0.1569 | 0.3585 |

#### B-ReKV 不是固定比例的变体

同一全局超参下，每条 query 的实际预算明显变化：

| Pair / Task | Mean | Std | P10–P90 | 不同预算值 |
|---|---:|---:|---:|---:|
| #1 HotpotQA | 0.2754 | 0.0202 | 0.2487–0.3000 | 486/500 |
| #1 MuSiQue | 0.2745 | 0.0243 | 0.2427–0.3028 | 500/500 |
| #6 MultiFieldQA | 0.3948 | 0.0466 | 0.3395–0.4457 | 149/150 |
| #7 HotpotQA | 0.3853 | 0.0240 | 0.3540–0.4129 | 488/500 |
| #7 MuSiQue | 0.3441 | 0.0304 | 0.3094–0.3797 | 496/500 |

这直接证明 B-ReKV 在固定 `(tau, scale, window)` 下仍按 query 分配不同预算。

#### Strict Adaptive

- Stage 3 中 Strict 仅在 pair #1 的 3 个任务上运行，共 9 runs。
- 它是“保证 coverage”的机制消融，不是主方法。
- Pair #6 的早期 strict root 有 9 个结果，但缺少 v1 协议元数据，不能用于
  新协议通信成本结论。

### 3.3 Query-Sketch 专项实验

#### Oracle gap

Root：`snapshots/query_sketch_oracle_gap/`，18/18 matched cells 完成。
该批是 cost/profile 输出，主要证据在 `cost_summary.json`，没有
`per_sample.jsonl` 属于设计口径，不是运行失败。

- ReKV：Query-Sketch 相对 Full-KV Oracle 平均分差 -0.0378；
  通信量平均减少 61.0%。
- B-ReKV：平均分差 +0.0067；通信量平均减少 35.1%。
- Query-Sketch latency 略低，峰值显存略低。

#### Sketch 表示与窗口

Root：`snapshots/query_sketch_representation_ablation/`，72 runs 完成。

- BF16-w8：平均分 0.5467，B→A sketch 约 1,696 KB。
- INT8-w8：平均分 0.5500，B→A sketch 约 849 KB。
- INT8-w8 把 sketch payload 约减半，平均分没有下降。
- token_ids payload 极小，但最佳分数明显低于 BF16/INT8。
- BF16/INT8 都以 w8 最稳，扩大到 w16/w32 没有稳定收益。

#### 机制消融

- Layer aggregation：identity 0.4896，优于 last 0.3580、mean 0.4004、
  top4 0.4400。
- Score function：receiver 0.4896、receiver_recency 0.4936，
  明显高于 value_norm 0.3596 和 random 0.3027。

## 4. Table 1：新旧结果必须分开

### 4.1 历史主矩阵

- Pair #6：72/72 完成，`legacy_implicit_receiver_v0`。
- Pair #7：72/72 完成，`legacy_implicit_receiver_v0`。
- Pair #8：启动失败/不完整，暂缓。

代表历史结果：

- Pair #6 HotpotQA：best ReKV 0.718；best B-ReKV 0.674。
- Pair #6 MuSiQue：best ReKV 0.396；best B-ReKV 0.384。
- Pair #6 MultiFieldQA：best B-ReKV 0.493，略高于 best ReKV 0.487。

这些结果仍可作为历史准确率证据，但论文表必须写明 legacy/pre-protocol。

### 4.2 Query-Sketch v1 重跑

| Pair | Fixed ReKV | Frozen B-ReKV | 协议状态 |
|---|---:|---:|---|
| #1 | 48/48 | 0/8 | fixed ReKV 全部为 v1 |
| #6 | 72 个 unique cell | 0/8 | 仅 5 个物理文件有 v1；68 个为 v0 pre-instrumentation |
| #7 | 45/48 | 0/8 | v1；缺 tmath w16 × 3 |

当前不能生成最终 Query-Sketch Table 1：冻结 B-ReKV 主块 0/24，pair #6
缺完整 v1 instrumentation，pair #7 尚缺 3 个 fixed ReKV runs。

## 5. Table 6：五个 extended tasks

### 5.1 历史结果：90/90 完成

Pair #6/#7 × HotpotQA-full、Qasper-full、MuSiQue-full、SAMSum、RepoBench，
每个任务 9 个配置。全部属于 `legacy_implicit_receiver_v0`。

| Pair / Task | Best ReKV | Best B-ReKV | B-ReKV 实际预算 |
|---|---:|---:|---:|
| #6 HotpotQA-full | 0.7579 | 0.7153 | 0.3685 |
| #6 Qasper-full | 0.3459 | 0.3384 | 0.3429 |
| #6 MuSiQue-full | 0.4344 | 0.4096 | 0.2501 |
| #6 SAMSum | 0.2886 | 0.2609 | 0.3034 |
| #6 RepoBench | 0.3549 | 0.3402 | 0.1089 |
| #7 HotpotQA-full | 0.5754 | 0.4917 | 0.3675 |
| #7 Qasper-full | 0.2329 | 0.2109 | 0.3451 |
| #7 MuSiQue-full | 0.3736 | 0.3103 | 0.3250 |
| #7 SAMSum | 0.3387 | 0.3150 | 0.3825 |
| #7 RepoBench | 0.3530 | 0.3400 | 0.1594 |

Pair #7 RepoBench 已从早期 OOM 中恢复，9/9 最终完成。

### 5.2 新协议重跑

Table 6 Query-Sketch 目标为 90 runs，目前 0/90，尚未启动。

## 6. Table 8：模型泛化

历史 paper-table block：

- Pair #2/#3/#4/#5：各 72/72 完成，均为 legacy。
- Pair #1：主 paper block 完成，另有大量历史 dense sweep。
- Pair #9：72/72 完成，但 QA/多跳分数接近零，只作为 hard negative /
  limitation；同时存在 original 和 corrected 两个完整 root，正式引用前需固定来源。
- Pair #8：Falcon checkpoint 问题，继续暂缓。

Query-Sketch 重跑目标：

- Pair #2/#3/#4/#5 使用冻结配置，共 224 runs；
- 当前 0/224，尚未启动；
- Pair #9 不重跑正向主表，Pair #8 暂缓。

## 7. Table 10：Multi-Source

### 7.1 历史 18/18

Root：`snapshots/table10_multi_source_rekv/`。

- HotpotQA best：0.6800（w16 r=0.5）。
- MuSiQue best：0.4620（w8 r=0.7）。
- 2WikiMQA best：0.4300。

但旧实现让 receiver 在选择前获得 A1||A2 full KV，必须标为
`legacy_implicit_receiver_multi_source_v0` / oracle-style prototype，不能称为
真正 Multi-Source Query-Sketch，也不能与单源新协议直接比较。

### 7.2 真正 Query-Sketch

新实现已经把流程改为：B 发送 bf16 Q sketch，A1/A2 各自用本地 key
评分并压缩，然后只发送压缩 KV。目标 root：
`snapshots/table10_multi_source_query_sketch/`，当前 0/18，尚未启动。

## 8. 历史 supporting evidence

以下历史实验已经完成，但主要属于 legacy/pre-protocol：

- Pair #1/#6/#7 旧 Query Fairness；
- B-ReKV coverage robustness / Pareto；
- budget headroom、progressive、budget predictor 等正负链路；
- score-function、layer aggregation、sink/recent、positional coherence；
- failure cases、task sensitivity、interpretability；
- NLD vs ReKV cost。

它们可以支撑方法动机、机制和负结果叙事，但凡涉及“可部署协议通信量”的
结论，应优先引用新 Query-Sketch 专项实验。

## 9. 数据质量与文档风险

1. Stage 3 旧 `analysis/all_runs.csv` 曾只写出 14/360；现已修复分析脚本并
   重建为 360/360。
2. Config freeze 有 35 组重复 logical config；正式分析使用 latest run，
   原始重复文件保留用于追溯。
3. 70% 历史 `per_sample` 没有 `protocol_version`，必须按 root 和时间标记 legacy。
4. Pair #9 有两个完整 root，正式论文只能选择一个 canonical source。
5. 若干文档引用不存在的 PNG；当前环境没有 matplotlib，因此 CSV/Markdown
   已生成，但图需要在含 matplotlib 的环境重新生成。
6. `manifest/experiments.csv` 尚未完整覆盖新 Query-Sketch roots，不能单独作为
   当前完成度来源。

## 10. 当前可写与不可写的结论

可以写：

- Query-aware selection 在 matched budget 下稳定优于 ValueNorm/Random。
- B-ReKV 的 per-query 预算分布有明显方差，不是固定比例换名。
- 冻结配置在 6 个 calibration cells 上 matched-budget 6/6 胜/平。
- INT8 sketch 可减半 B→A payload，且平均分不降。
- Legacy Table 6/8/10 已完整，但必须明确 pre-protocol。

暂时不能写：

- “Query-Sketch Table 1 已完整”：冻结 B-ReKV 仍是 0/24。
- “Table 6/8/10 已用新协议重跑”：三个新矩阵均为 0。
- “Canonical B-ReKV 在所有 matched-budget 单元都优于 fixed ReKV”：实际为 3/9。
- 把旧 Table 10 当成真正 Multi-Source Query-Sketch。
