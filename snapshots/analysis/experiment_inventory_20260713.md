# 实验结果总盘点（更新至 2026-07-14）

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
- Query-Sketch 配置搜索网格和 Stage 3 核心审稿矩阵已经完成；原冻结判定
  存在 matched-budget 越界问题，需撤回并重新确认。
- 2026-07-14 本机 fast-node 队列 123/123 完成：Table 6 为 35/70、
  Table 8 为 28/224、Table 10 为 18/18、Table 1 main B-ReKV 为 16/24、
  正式 cost 为 12/18。详见 `fast_node_completion_20260714.md`。

## 3. 新协议：已经完成的核心证据

### 3.1 B-ReKV 配置搜索：网格完成，冻结判定无效

Root：`snapshots/query_sketch_config_freeze/`

- 逻辑矩阵：2 pairs × 3 tasks ×（5 fixed ReKV + 6 B-ReKV）= 66/66。
- 物理文件：131 个，包含两轮重复运行；分析脚本按 logical config 取最新结果。
- 每个 run 使用相同的前 100 个样本。
- 全部结果为 `query_sketch_bf16_v1`。
- 原分析器选择了 `B-ReKV-t0.98-s1-w8`，平均实际预算 0.5919。
- 但它的 6 个预算均为 0.54–0.67，全部超过 fixed ReKV 校准曲线的实际
  上界约 0.42。
- `analyze_query_sketch_config_freeze.py` 对越界预算直接返回最高 fixed-r
  端点，而不是标记 `above_range`；因此“平均 +0.025、最差 +0.010、
  6/6 胜/平”不是 matched-budget 结果。
- `selection.json` 当前应视为失效的历史分析产物，不能驱动 Table 1/8/cost
  主矩阵。

当前建议：

- 正文低预算主 operating point：`t=0.95, scale=0.75, w=8`；
- 中预算 Pareto/稳健点：`t=0.95, scale=0.85, w=8`；
- `t=0.98, scale=1.0, w=8`：仅作为 high-fidelity/high-budget 消融；
- 若要重新冻结，fixed ReKV 校准网格至少补到实际预算 0.70，并在越界时拒绝验收。

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
fairness/Pareto 的统一观察点，和原分析器选出的高预算
`t=0.98, scale=1.0, w=8` 不是同一个配置。

Canonical B-ReKV 在 9 个 pair-task 单元上：

- 相对 ValueNorm/Evict：9/9 获胜；
- 相对 Random：9/9 获胜；
- 相对 matched-budget fixed ReKV：3/9 获胜、6/9 落后。

因此可以稳健声称“query-aware selection 明显优于 query-agnostic
ValueNorm/Random”；不能声称 canonical B-ReKV 在所有 pair/task 都优于
matched-budget fixed ReKV。B-ReKV 的主要价值是动态预算和 Pareto，而不是
每个单元都提升绝对分数。

#### Full 7×8 matched-budget expansion（2026-07-18 阶段性）

Root：`snapshots/full_matched_budget_fairness_query_sketch/`

- 进度：**1443/1568（92.0%）**；GPU1 仍在跑剩余项。
- 已完成可分析单元：**51** 个 pair-task（每单元 ReKV + ValueNorm + Random 插值）。
- 缺块：`pair6/tmath`，以及 `pair7` 的 `tipsheets/qasper/multifieldqa_en/tmath`。
- 主配置仍为 `t0.95-s0.75-w8`；fixed ratio 网格 `0.10..0.60`。
- 详细报告：`snapshots/analysis/full_matched_budget_fairness_20260718.md`

在已完成 51 个单元上（B-ReKV − matched baseline）：

| Baseline | wins | mean Δ | 结论 |
|---|---:|---:|---|
| matched ReKV | 26/51 | +0.0056 | 基本持平 |
| ValueNorm/Evict | 47/51 | +0.0771 | 明显更强 |
| Random | 50/51 | +0.1182 | 明显更强 |
| best-per-task fixed ReKV | 1/51 | -0.0937 | 事后最优预算上界；非公平主结论 |
| global fixed `r=0.6` | 5/51 | -0.0874 | 全局高预算；非免调参设定 |

论文口径应写：B-ReKV 的价值是**无需逐任务选预算**，在 matched-budget
下接近 ReKV 并显著超过 query-agnostic 基线；不要把 vs best-fixed 写成
“B-ReKV 更弱”的主结论。

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
- 旧高预算 B-ReKV：平均分差 +0.0067；通信量平均减少 35.1%。
- Query-Sketch latency 略低，峰值显存略低。

2026-07-14 已补当前主配置 `t0.95-s0.75-w8` 的 Pair #1/#7 六单元：

- 平均 score gap 为 -0.1300；
- 平均通信节省为 59.62%；
- Pair #6 尚缺，因此为 12/18 physical runs；
- QS 与 Oracle 动态预算不同，该值不是严格 matched-budget selector gap。

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
| #1 | 48/48 | 8/8 | 56/56，全部为 v1 |
| #6 | 1/48 | 0/8 | 其余早期结果为 v0 pre-instrumentation |
| #7 | 48/48 | 8/8 | 56/56，全部为 v1 |

当前 Table 1 为 113/168：Pair #1/#7 完成，Pair #6 尚缺 47 个 fixed
ReKV 和 8 个 main B-ReKV，因此仍不能生成最终三 pair 汇总。

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

Table 6 Query-Sketch 主矩阵已统一为每任务 6 个 fixed ReKV +
`t0.95-s0.75-w8` 一个 B-ReKV，共 70 runs，目前 35/70；Pair #7
五个任务全部完成，Pair #6 尚缺 35。
旧三点 B-ReKV 中的 `s0.85-w8` 和 `s0.90-w16` 不再计入主矩阵。

## 6. Table 8：模型泛化

历史 paper-table block：

- Pair #2/#3/#4/#5：各 72/72 完成，均为 legacy。
- Pair #1：主 paper block 完成，另有大量历史 dense sweep。
- Pair #9：72/72 完成，但 QA/多跳分数接近零，只作为 hard negative /
  limitation；同时存在 original 和 corrected 两个完整 root，正式引用前需固定来源。
- Pair #8：Falcon checkpoint 问题，继续暂缓。

Query-Sketch 重跑目标：

- Pair #2/#3/#4/#5 使用冻结配置，共 224 runs；
- 当前 28/224；Pair #5 前四任务完成，剩余 196；
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
`snapshots/table10_multi_source_query_sketch/`，当前 18/18 完成。

- HotpotQA best：0.6680（w16-r0.7）。
- MuSiQue best：0.4500（w8-r0.7）。
- 2WikiMQA best：0.4400（w8-r0.7）。

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
- 配置搜索已完成；低预算 `t0.95-s0.75-w8` 有完整 Stage 3 fairness、
  Pareto 和预算分布证据。
- INT8 sketch 可减半 B→A payload，且平均分不降。
- 真正 Multi-Source Query-Sketch 已完成 18/18，三任务 best 为
  0.6680 / 0.4500 / 0.4400。
- Table 6 Pair #7 的五个 extended tasks 已完成新协议重跑；main B-ReKV
  平均以约 51% 更低预算换取 0.0793 的平均分数下降。
- Legacy Table 6/8/10 已完整，但必须明确 pre-protocol。

暂时不能写：

- “Query-Sketch Table 1 已完整”：当前为 113/168，Pair #6 尚缺 55。
- “`t0.98-s1-w8` 已通过 matched-budget 冻结”：其预算全部超出 fixed-r
  校准网格，原 6/6 结论来自端点截断。
- “Table 6/8 已完整用新协议重跑”：当前分别为 35/70 和 28/224。
- “Canonical B-ReKV 在所有 matched-budget 单元都优于 fixed ReKV”：实际为 3/9。
- 把旧 Table 10 当成真正 Multi-Source Query-Sketch。
