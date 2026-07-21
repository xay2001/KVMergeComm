# Full Matched-Budget Fairness 最终报告（Query-Sketch v1）

## 1. 实验完成与口径

- 完成度：**1568/1568（100%）**
- 覆盖：7 个模型对 × 8 个任务 = 56 个 pair-task 单元
- 每个单元：9 个 fixed-r ReKV + 9 个 ValueNorm/Evict + 9 个 Random + 1 个主 B-ReKV
- B-ReKV 主配置：`tau=0.95, scale=0.75, recv_window=8`
- fixed-r 网格：`0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60`
- Query-Sketch runs：560，协议 `query_sketch_bf16_v1`
- query-agnostic runs：1008，协议 `query_agnostic_kv_v1`
- matched-budget 插值：168/168 成功，0 个超出 fixed-r 网格
- 无重复逻辑 run，无缺失单元
- manifest 另保留 2 个 2026-07-15 早期被终止的 `r0.10` 目录（`status=unknown`）；它们没有 `per_sample.jsonl`，不计入 1568 个有效 run

结果根目录：

`snapshots/full_matched_budget_fairness_query_sketch/`

## 2. 最重要结论

### 2.1 同实际预算：B-ReKV 与 ReKV 基本持平

在 56 个 pair-task 单元中：

| 对照 | B-ReKV 胜/负 | 平均分差 | 中位数分差 |
|---|---:|---:|---:|
| matched-budget ReKV | 28 / 28 | **+0.0036** | **+0.0005** |
| matched-budget ValueNorm/Evict | 51 / 5 | **+0.0923** | **+0.0643** |
| matched-budget Random | 54 / 2 | **+0.1267** | **+0.1068** |

这组结果直接回答公平性问题：

1. B-ReKV 并不是单纯用更小预算换更低分；在相同实际 KV budget 下，它与 ReKV 平均基本持平。
2. 接收方查询感知是必要的：同预算下 B-ReKV 明显优于 ValueNorm 和 Random。
3. B-ReKV 的主要价值不是每个任务都超过事后调优的 fixed ReKV，而是无需逐任务选择 `r`，仍能维持同预算 ReKV 的平均质量。

### 2.2 不能写成“B-ReKV 普遍超过 best fixed ReKV”

若允许每个任务事后从 9 个 fixed-r 点中挑最高分：

- B-ReKV：1 胜、54 负、1 平
- 平均分差：`-0.0889`
- 中位数分差：`-0.0765`

这不是 matched-budget 比较。best-fixed 通常选择更高预算点，回答的是“如果可以针对每个任务单独调预算，最高准确率是多少”，而不是“相同通信预算下 selector 是否更好”。

因此论文中应将 best-fixed 作为 accuracy upper envelope / 逐任务调参上界，而不是主要公平对照。

## 3. 按模型对分析

以下均为 B-ReKV 减 matched baseline 的八任务宏平均。

| Pair | vs ReKV W/L | Δ ReKV | vs ValueNorm W/L | Δ ValueNorm | vs Random W/L | Δ Random |
|---|---:|---:|---:|---:|---:|---:|
| #1 Llama-3.1-8B same | 5/3 | +0.0229 | 8/0 | +0.0620 | 8/0 | +0.1321 |
| #2 Llama-3.2-3B same | 6/2 | +0.0131 | 8/0 | +0.0595 | 8/0 | +0.0941 |
| #3 Qwen2.5-7B same | 1/7 | -0.0155 | 4/4 | +0.0321 | 7/1 | +0.0805 |
| #4 Falcon3-7B same | 2/6 | -0.0131 | 8/0 | +0.0598 | 8/0 | +0.0971 |
| #5 EvolCodeLlama → ToolACE | 5/3 | +0.0118 | 8/0 | +0.0949 | 8/0 | +0.1532 |
| #6 Abliterated-3B → DeepSeek-3B | 6/2 | +0.0137 | 7/1 | +0.0761 | 8/0 | +0.0901 |
| #7 Qwen-Uncensored → Bespoke | 3/5 | -0.0078 | 8/0 | +0.2615 | 7/1 | +0.2396 |

解释：

- #1/#2/#5/#6：B-ReKV 对 matched ReKV 平均为正，支持跨同构和异构模型的免调预算价值。
- #3/#4：对 ReKV 略负，但平均差只有约 `-0.015` / `-0.013`；同时仍显著优于 query-agnostic 基线。
- #7：对 ReKV 略负，但对 ValueNorm/Random 增益最大，说明困难异构 pair 更依赖查询感知 selector。
- 不能只报告总平均而隐藏 #3/#4/#7；这些模型对是方法边界的重要证据。

## 4. 按任务分析

| Task | vs ReKV W/L | Δ ReKV | Δ ValueNorm | Δ Random |
|---|---:|---:|---:|---:|
| countries | 6/1 | +0.0436 | +0.1385 | +0.1954 |
| tipsheets | 6/1 | +0.0716 | +0.2351 | +0.2200 |
| hotpotqa | 0/7 | -0.0259 | +0.0490 | +0.1697 |
| qasper | 3/4 | -0.0132 | +0.0482 | +0.0734 |
| musique | 2/5 | -0.0139 | +0.0448 | +0.1370 |
| multifieldqa_en | 3/4 | -0.0263 | +0.1384 | +0.1128 |
| twowikimqa | 3/4 | -0.0100 | +0.0785 | +0.0978 |
| tmath | 5/2 | +0.0026 | +0.0056 | +0.0075 |

主要模式：

- `countries` 和 `tipsheets` 是 B-ReKV 对 matched ReKV 最稳定的正向任务。
- `hotpotqa` 为 0/7，说明当前 coverage 配置在证据组合型多跳任务上不如同预算固定 ReKV selector；但仍明显优于 ValueNorm/Random。
- `qasper/musique/multifieldqa_en/twowikimqa` 对 ReKV 小幅负或接近，表明“动态分配预算”不保证每个数据集提升 selector 质量。
- `tmath` 各 selector 差异很小，说明该任务对 KV token 选择策略不敏感，不能作为 query-aware 优势的主要证据。

## 5. 关键成功与失败案例

### 对 matched ReKV 最大增益

- #1 / tipsheets：`+0.1833`
- #3 / countries：`+0.1453`
- #4 / tipsheets：`+0.1377`
- #5 / tipsheets：`+0.1348`
- #6 / countries：`+0.0957`

### 对 matched ReKV 最大退化

- #7 / multifieldqa_en：`-0.0759`
- #4 / countries：`-0.0752`
- #3 / multifieldqa_en：`-0.0641`
- #3 / twowikimqa：`-0.0601`
- #4 / multifieldqa_en：`-0.0600`

### ValueNorm 的少数反例

B-ReKV 仅在 5/56 单元低于 ValueNorm，其中四个来自 #3：

- #3 / hotpotqa：`-0.0948`
- #3 / musique：`-0.0920`
- #3 / qasper：`-0.0466`
- #3 / tmath：`-0.0051`
- #6 / tmath：`-0.0032`

这表明 Qwen2.5 same-model pair 是 receiver-aware scoring 的明确困难点，值得在 failure analysis 中单列，而不是仅报告整体 51/56。

## 6. 预算行为与“不是固定比例变体”

56 个 B-ReKV 单元的任务级实际预算：

- 均值：`0.3215`
- 中位数：`0.3373`
- 范围：`0.1490–0.4357`
- per-query budget std 均值：`0.0223`
- 每个单元 unique budget 中位数：`357.5`
- unique budget 范围：`93–500`

因此 B-ReKV 并非固定 `r` 的重命名：

1. 不同 pair-task 的平均预算跨度接近 3 倍。
2. 同一 pair-task 内存在大量不同的 per-query budget。
3. coverage 配置会根据 query/attention 分布改变保留量。

## 7. 固定比例曲线如何解读

全 56 单元宏平均：

| ReKV fixed ratio | 平均实际预算 | ReKV 平均分 | B-ReKV − ReKV | B-ReKV 胜数 |
|---:|---:|---:|---:|---:|
| 0.10 | 0.1481 | 0.2375 | +0.1634 | 53/56 |
| 0.15 | 0.1879 | 0.2891 | +0.1118 | 49/56 |
| 0.20 | 0.2311 | 0.3329 | +0.0680 | 48/56 |
| 0.25 | 0.2775 | 0.3675 | +0.0334 | 36/56 |
| 0.30 | 0.3244 | 0.3968 | +0.0040 | 27/56 |
| 0.35 | 0.3723 | 0.4226 | -0.0217 | 16/56 |
| 0.40 | 0.4207 | 0.4493 | -0.0484 | 12/56 |
| 0.50 | 0.5172 | 0.4715 | -0.0706 | 5/56 |
| 0.60 | 0.6137 | 0.4836 | -0.0828 | 5/56 |

B-ReKV 平均预算 `0.3215` 与 fixed `r=0.30` 的实际预算 `0.3244` 最接近；两者宏平均分差仅 `+0.0040`。这与逐单元插值结果 `+0.0036` 一致。

`r=0.6` 的分数更高，但其平均实际预算约为 B-ReKV 的 1.91 倍，不能作为 matched-budget 主对照。

## 8. 推荐论文表述

可以写：

> Across 7 model pairs and 8 tasks, B-ReKV matches fixed ReKV at the same realized KV budget (28/56 wins; mean Δ +0.0036), while substantially outperforming query-agnostic ValueNorm (51/56; +0.0923) and Random selection (54/56; +0.1267). This supports B-ReKV primarily as a query-adaptive, tuning-free budget allocator rather than an accuracy-dominating replacement for task-wise budget search.

不能写：

- “B-ReKV 普遍优于 fixed ReKV。”
- “B-ReKV 达到了逐任务最优 ReKV 的准确率。”
- “更高 fixed-r 与 B-ReKV 的比较是 matched-budget。”
- “所有任务都受益于动态预算。”HotpotQA 明确是 0/7。

## 9. 产出文件

- 自动短报告：`snapshots/full_matched_budget_fairness_query_sketch/analysis/REPORT.md`
- 全部 run：`analysis/all_runs.csv`
- 168 行 matched 对照：`analysis/query_fairness_matched_budget.csv`
- 56 行预算分布：`analysis/brekv_budget_distribution.csv`
- B-ReKV vs best fixed：`analysis/brekv_vs_best_fixed_rekv.csv`
- 最终汇总 JSON：`analysis/final_summary.json`

本报告为最终 1568/1568 版本，替代
`snapshots/analysis/full_matched_budget_fairness_20260718.md`
中的 92% 阶段性统计。
