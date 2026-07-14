# ReKV / B-ReKV 实验清单

这份文档是论文实验的总 checklist。目标是把“已经完成的证据”和“还需要补的实验”分开记录，并说明每个实验为什么重要。后续每跑完一批实验，就更新这里的状态。

命名约定：

- **ReKV**：Receiver-aware KV Communication，固定预算的 receiver-aware token selection。
- **B-ReKV**：budget-aware ReKV family，目前实现是 receiver-attention coverage 动态预算。
- **KVComm**：原论文的 layer-level KV communication baseline。除非特别说明，KVComm baseline 可以直接引用原论文结果，不需要本地复现。

## 0. 当前总状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 主表 ReKV 覆盖 | 已完成扩展覆盖 | Table 1 pair #6/#7 完整；Table 8 pair #1/#2/#3/#4/#5 完整可用；pair #9 已完成但暂缓作为正向对比。 |
| B-ReKV 稳健性 | 已完成核心覆盖 | pair #1 的 MuSiQue / HotpotQA / MultiFieldQA-en Pareto 与 budget distribution 已完成；pair #6/#7 的 HotpotQA / MuSiQue 小网格已完成，并已生成 summary / Pareto 图。 |
| Cost profiling | pair #1/#6/#7 已完成 | pair #1 有 8 个主数据集 cost 表；pair #6/#7 full cost profile 已完成，并已生成 HotpotQA / MuSiQue / MultiFieldQA-en 论文子表。 |
| Query-aware fairness | pair #1/#6/#7 已完成 | pair #1 完成 ReKV vs Evict vs Random 和 query sketch window ablation；pair #6/#7 完成 ReKV/Evict/Random/B-ReKV 扩展。 |
| 可解释性 | pair #1 已完成 | answer overlap、清洗后的 qualitative examples、token bar 图、deletion ablation、HotpotQA supporting-facts overlap 已完成。 |
| 机制消融 | 主要机制已完成 | Sink/recent token ablation 已完成；Table 11 positional coherence 可跑部分已完成 8 个主任务；B-ReKV-S 因 shift-back 实现限制暂缓；score-function ablation 与 layer aggregation ablation 已完成并生成 summary/figures。 |
| Table 1 pair #8 Falcon | 暂缓/最后处理 | `Falcon3-7B-Instruct-abliterated` checkpoint 目前不可获得或目录不完整；不阻塞主线，放到最后可选处理。 |
| Table 6 / 10 / 11 | Table 6/10 完成，Table 11 可跑部分完成 | Table 6 pair #6/#7 均完成 5 个 extended tasks x 9 runs；pair #7 RepoBench 已在 98 GB GPU 上补齐。Table 10 三任务共 18 runs 已完成。Table 11 已完成 ReKV normal / ReKV-S / B-ReKV normal x 8 主任务，B-ReKV-S 暂缓。 |

## 0.1 2026-07-14 新 Query-Sketch 协议重跑状态

上面的“已完成”大多指历史实验资产，其中 receiver-aware 部分多数属于旧隐式
Full-KV Oracle 口径。论文最终表必须以本节的新协议状态为准，不能直接混用。

| 新协议模块 | 当前状态 | 口径 |
|---|---|---|
| B-ReKV 配置搜索 | 已冻结 | 高预算冻结作废；正文主点 `t=0.95, s=0.75, w=8` |
| Query-Sketch vs 显式 Oracle | 已完成（pair #6 canonical） | 三任务平均通信降约 `65%`，score gap `-0.08` |
| Sketch 表示与窗口 | 已完成 | 72 runs；INT8-w8 最优轻量折中 |
| Table 1 七点主矩阵 | **已完成 24/24** | pair #1/#6/#7 × 8 tasks；canonical B-ReKV 平均预算 `0.30/0.32/0.37` |
| Table 6 Query-Sketch | **已完成 10/10** | pair #6/#7 × 5 extended tasks |
| Table 8 Query-Sketch | **已完成 32/32** | pair #2/#3/#4/#5 × 8 tasks |
| Table 10 Query-Sketch | 已完成 | 18/18；best 0.668 / 0.450 / 0.440 |
| 新协议正式 Cost / Efficiency | **已完成 18/18** | pair #1/#6/#7 × 3 tasks × ReKV/B-ReKV |
| NLD vs Query-Sketch v1 | 已完成 | ReKV/B-ReKV 准确率高约 +0.25~+0.41，latency 快约 2.8×–7.0× |
| 新协议 Score Function | 已完成 | Pair #6 三任务 × 5 score modes |
| 新协议 Layer Aggregation | 已完成 | Pair #6 三任务 × 5 aggregations |

严格区分：

- `query_sketch_bf16_v1` / `int8_v1` / `token_ids_v1`：当前可部署协议。
- `full_kv_oracle_v1`：显式性能上界，只进入 oracle-gap。
- `query_agnostic_kv_v1`：ValueNorm / Random 公平性对照。
- `query_sketch_bf16_v0_pre_instrumentation` 与历史隐式 Oracle：不进入最终主表。

最终汇总：

```text
snapshots/analysis/query_sketch_final_20260714/REPORT.md
snapshots/analysis/nld_vs_rekv_query_sketch_v1/
scripts/summarize_query_sketch_final.py
```

仍暂缓：Pair #8 Falcon、Pair #9 SuperNova、B-ReKV-S、Head-wise B-ReKV。

## 0.2 2026-07-05 历史审计记录

这次审计基于 git 状态、`snapshots/` 产物数量和队列日志：

- 审计开始时 git 工作区无可见未提交改动；本次审计只更新记录文档。大量实验产物可能被 `.gitignore` 忽略，因此以实际 `snapshots/` 产物为准。
- 当前没有检测到本用户的 `com.py` / 队列脚本进程在运行；GPU 上剩余进程不是本实验队列。
- Pair #6/#7 full cost profile 已全部完成：
  - `snapshots/cost_profile/table1_pair6_llama32_abliterated_deepseek3b_full/cost_table.csv`
  - `snapshots/cost_profile/table1_pair7_qwen25_uncensored_bespoke_full/cost_table.csv`
  - 每个 pair 都有 8 tasks x 10 methods = 80 个 `cost_summary.json`。
  - 论文子表已整理：`snapshots/analysis/cost/pair6_pair7_cost_focus_hotpotqa_musique_multifieldqa.csv`。
- Pair #6/#7 B-ReKV robustness 小网格已完成：
  - `snapshots/table1_pair6_llama32_abliterated_deepseek3b/{hotpotqa,musique}/coverage/`
  - `snapshots/table1_pair7_qwen25_uncensored_bespoke/{hotpotqa,musique}/coverage/`
  - 每个 pair 的 HotpotQA / MuSiQue 均有 canonical 点 + 小网格产物。
  - 汇总表和 Pareto 图已生成：`snapshots/analysis/robustness/pair6_pair7_brekv_robustness_summary.csv` 和 `snapshots/analysis/robustness/pair{6,7}_{hotpotqa,musique}_brekv_pareto.png`。
- Sink/recent token ablation 已完成：
  - 结果目录：`snapshots/mechanism/pair1_llama31_same/sink_recent/`
  - 覆盖 8 个主任务，每个任务 ReKV/B-ReKV x 4 个 sink/recent 组合，共 64 个 `per_sample.jsonl`。
- Positional coherence 队列状态：
  - 结果目录：`snapshots/mechanism/pair1_llama31_same/positional_coherence/`
  - 已完成 8 个主任务的 ReKV normal、ReKV-S、B-ReKV normal。
  - 汇总：`snapshots/analysis/mechanism/positional_coherence_summary.md`。
  - B-ReKV-S 在 `--shift_back` + coverage budget 下触发 `models.py:get_short_past_key_values` 的 `assert len(lengths) <= 2`，队列停止；诊断见 `snapshots/analysis/mechanism/brekv_shiftback_diagnosis.md`。
  - `scripts/run_gpu7_mechanism_extended_full_queue.sh` 已默认 `RUN_BREKV_SHIFT=0` 绕开 B-ReKV-S，避免后续 Table 6 被阻塞。
- Table 8 pair #4/#5/#9 的 paper-table 队列已完成；pair #9 结果多数接近 0，分数分布、raw-output 和 KVComm probe 诊断见 `snapshots/analysis/pair9/pair9_diagnostic_report.md`。KVComm top=0.3 limit=50 probe 在 `hotpotqa=0.060`, `musique=0.020`, `qasper=0.000`，说明不是 ReKV 独有失败；pair #9 暂缓作为正向对比。

## 1. 论文表格主线

### 1.1 Table 1：主文 fine-tuned model pairs

目的：在 KVComm 主文的 fine-tuned model pairs 上，对比 ReKV / B-ReKV 与 KVComm 论文报告的 baseline。

每个 dataset 的 canonical block：

- ReKV-w8：`r=0.3`, `r=0.5`, `r=0.7`
- ReKV-w16：`r=0.3`, `r=0.5`, `r=0.7`
- B-ReKV：`cov_t0.95_s0.75_w8`
- B-ReKV：`cov_t0.95_s0.85_w8`
- B-ReKV：`cov_t0.95_s0.90_w16`

| Pair | 结果目录 | 状态 | 剩余工作 |
|---|---|---|---|
| #6 Llama3.2-abliterated -> DeepSeek3B | `snapshots/table1_pair6_llama32_abliterated_deepseek3b/` | 已完成 | 无 |
| #7 Qwen2.5-uncensored -> Bespoke | `snapshots/table1_pair7_qwen25_uncensored_bespoke/` | 已完成 | 无 |
| #8 Falcon fine-tuned pair | `snapshots/table1_pair8_falcon3_ultraset_abliterated/` | 暂缓/最后处理 | `Falcon3-7B-Instruct-abliterated` checkpoint 当前不可获得或本地目录不完整；若后续找到替代 checkpoint，再跑 8 datasets x 9 runs。 |

建议：

- 论文主表先以 pair #6/#7 作为主要证据。
- pair #8 不阻塞当前主线；只有在最后找到可用 checkpoint 或需要 Falcon-family substitute 时再处理。

### 1.2 Table 8：Appendix model pairs

目的：证明 ReKV / B-ReKV 不只在主文 pair 上有效，也能泛化到更多 model pairs。

| Pair | 结果目录 | 状态 | 剩余工作 |
|---|---|---|---|
| #1 Llama3.1 same-model | `snapshots/<dataset>/` | 已完成 | 5 个 fixed ReKV 小洞已补齐：HotpotQA `w8/w16 r=0.7`；QASPER `w8 r=0.3/0.5/0.7`。 |
| #2 Llama3.2 same-model | `snapshots/table8_pair2_llama32_same/` | 已完成 | 无 |
| #3 Qwen2.5 same-model | `snapshots/table8_pair3_qwen25_7b_same/` | 已完成 | 无 |
| #4 Falcon3 same-model | `snapshots/table8_pair4_falcon3_7b_same/` | 已完成 | 8 datasets x 9 paper-table runs 完整。 |
| #5 EvolCodeLlama -> ToolACE | `snapshots/table8_pair5_evolcodellama_toolace/` | 已完成 | 8 datasets x 9 paper-table runs 完整。 |
| #8 Falcon fine-tuned pair | `snapshots/table1_pair8_falcon3_ultraset_abliterated/` | 暂缓/最后处理 | blocker 同 Table 1 pair #8；原 receiver checkpoint 不可获得时可写入 limitation。 |
| #9 SuperNova -> DeepSeek-Llama-8B | `snapshots/table8_pair9_supernova_deepseek_llama8b/` | 已完成但暂缓 | 8 datasets x 9 paper-table runs 完整；除 `tmath` 外多数任务结果接近 0。raw-output / KVComm probe 显示更像 hard heterogeneous pair 的 KV 兼容问题，不作为正向对比。 |

建议：

- pair #1/#2/#3/#4/#5 已可作为 Table 8 appendix 正向覆盖；pair #9 只作为 hard negative / limitation 记录，暂缓放入正向对比。
- pair #8 仍受 Falcon receiver checkpoint blocker 影响；原 checkpoint 不可获得时可写入 limitation。

### 1.3 Table 6：Extended Tasks

目的：在 KVComm extended datasets 上验证鲁棒性。

状态：已完成。早期 GPU7 OOM 缺口已在 98 GB GPU 上补齐。

说明：

- 原始串行脚本：`scripts/run_gpu7_mechanism_extended_full_queue.sh`。
- 后续拆卡脚本：`scripts/run_table6_pair7_remaining_gpu{4,5,6,7}.sh`。
- Pair #6 已完成 5 个 extended tasks x 9 paper-style runs：
  - `hotpotqa_full`
  - `qasper_full`
  - `musique_full`
  - `samsum`
  - `repobench`
- Pair #7 已完成 5 个 extended tasks x 9 paper-style runs：
  - `hotpotqa_full`
  - `qasper_full`
  - `musique_full`
  - `samsum`
  - `repobench`
- Pair #7 RepoBench 早期在 48 GB GPU7 上因 receiver-attention `softmax` OOM；
  后续已在 98 GB GPU 0–3 上拆分补齐 9/9，每个配置 1000 样本。
- 汇总与图：
  - `snapshots/analysis/latest_experiments/table6_extended_summary.csv`
  - `snapshots/analysis/latest_experiments/table6_extended_status.csv`
  - `snapshots/analysis/latest_experiments/table6_extended_best_by_pair_task_family.csv`
  - `snapshots/analysis/latest_experiments/figures/table6_pair6_extended_best.png`
  - `snapshots/analysis/latest_experiments/figures/table6_pair7_extended_best.png`
- 当前命令形态：

```bash
RUN_SINK_RECENT=0 RUN_POSITIONAL=1 RUN_BREKV_SHIFT=0 RUN_TABLE6=1 GPU=7 \
  bash scripts/run_gpu7_mechanism_extended_full_queue.sh
```

Pair #6 关键结果：

- `hotpotqa_full`：best ReKV `0.7579`，best B-ReKV `0.7153`，B-ReKV 平均预算约 `0.3685`。
- `musique_full`：best ReKV `0.4344`，best B-ReKV `0.4096`，B-ReKV 平均预算约 `0.2501`。
- `qasper_full`：best ReKV `0.3459`，best B-ReKV `0.3384`。
- `samsum`：best ReKV `0.2886`，best B-ReKV `0.2609`。
- `repobench`：best ReKV `0.3549`，best B-ReKV `0.3402`，B-ReKV 平均预算约 `0.1089`。

Pair #7 当前关键结果：

- `hotpotqa_full`：best ReKV `0.5754`，best B-ReKV `0.4917`。
- `musique_full`：best ReKV `0.3736`，best B-ReKV `0.3103`。
- `qasper_full`：best ReKV `0.2329`，best B-ReKV `0.2109`。
- `samsum`：best ReKV `0.3387`，best B-ReKV `0.3150`。
- `repobench`：best ReKV `0.3530`，best B-ReKV `0.3400`；后者实际预算
  `0.1594`，约为 ReKV-w8 r=0.3 的 49%，绝对分数仅低 `0.0085`。

结论：Pair #6/#7 extended tasks 已完整支撑“ReKV/B-ReKV 不只在主任务有效”的附录鲁棒性结论。RepoBench 的 48 GB OOM 仍应作为实现层显存限制记录，但不再是实验缺口。

### 1.4 Table 9：Random Selection Ablation

目的：回答 receiver-aware token selection 是否真的优于 random / query-agnostic token selection。

已完成：

- Pair #1 的 `hotpotqa`, `musique`, `multifieldqa_en`。
- 方法：ReKV、Evict/ValueNorm、Random-token。
- Query windows：`recv_window={4,8,16,32,all}`。
- 产物：`snapshots/query_fairness/pair1_llama31_same/query_fairness.csv`。

Pair #6/#7 扩展已完成：

- 任务：`hotpotqa`, `musique`, `multifieldqa_en`。
- 方法：Evict/ValueNorm、Random-token、ReKV `w8/w16 r=0.3`、B-ReKV 三个 canonical 点。
- 产物：`snapshots/query_fairness/pair6_pair7_query_fairness_brekv_summary.csv`。

关键结果：

- Pair #6:
  - HotpotQA: Evict `0.516`, Random `0.406`, best ReKV `0.668`, best B-ReKV `0.674`。
  - MuSiQue: Evict `0.236`, Random `0.182`, best ReKV `0.362`, best B-ReKV `0.384`。
  - MultiFieldQA-en: Evict `0.327`, Random `0.320`, best ReKV `0.467`, best B-ReKV `0.493`。
- Pair #7:
  - HotpotQA: Evict `0.124`, Random `0.118`, best ReKV `0.396`, best B-ReKV `0.446`。
  - MuSiQue: Evict `0.144`, Random `0.080`, best ReKV `0.298`, best B-ReKV `0.308`。
  - MultiFieldQA-en: Evict `0.080`, Random `0.140`, best ReKV `0.393`, best B-ReKV `0.393`。

结论：跨 fine-tuned pairs 后，query-aware ReKV/B-ReKV 仍显著强于 query-agnostic Evict/Random，能直接回应 “ReKV 只是因为用了 query / 不公平” 的审稿风险。

### 1.5 Table 10：Multi-Source KV Communication

目的：多个 sender 给一个 receiver 通信 KV。

状态：未做。

可能的最小实验：

- 任务：`hotpotqa`, `musique`。
- 对比：
  - sender1 only
  - sender2 only
  - naive concat selected KV
  - ReKV per-sender top tokens
  - B-ReKV per-sender coverage budget

优先级：低到中。做出来很加分，但属于新方向，可能需要代码改动。

### 1.6 Table 11：Positional Coherence / KVComm-S

KVComm-S 是什么：

- **KVComm-S** 是 KVComm 的 positional-coherence 消融。
- 正常 KVComm 在注入 selected KV layers 时会维护 position shift / coherence。
- KVComm-S 去掉这部分 coherence 处理。
- 该表用于证明 KV cache communication 需要位置一致性，不是随便拼 KV。

状态：部分完成 / 当前阻塞。

可做的 ReKV 版本：

- ReKV normal vs ReKV-S：破坏或关闭 position-shift coherence。
- B-ReKV normal vs B-ReKV-S。
- 任务：`hotpotqa`, `musique`。
- 先用 pair #1。

当前实际进展：

- 脚本：`scripts/run_gpu7_mechanism_extended_full_queue.sh`。
- 结果目录：`snapshots/mechanism/pair1_llama31_same/positional_coherence/`。
- 已完成 8 个主任务的 ReKV normal、ReKV-S、B-ReKV normal。
- 汇总产物：`snapshots/analysis/mechanism/positional_coherence_summary.md`。
- 主要观察：
  - HotpotQA: ReKV normal `0.6960` -> ReKV-S `0.6120`。
  - MuSiQue: `0.4800` -> `0.3440`。
  - MultiFieldQA-en: `0.5067` -> `0.4267`。
  - 2Wiki: `0.4050` -> `0.2600`。
  - QASPER: `0.3440` -> `0.2900`。
  - 这支持 positional coherence 对长文 / 多跳任务很重要。
- B-ReKV-S 在 `--shift_back` + coverage budget 下触发 `AssertionError`：
  - 日志：`snapshots/mechanism/logs/gpu7_mechanism_extended_full_0703_1144.log`。
  - 报错位置：`models.py:get_short_past_key_values` 中 `assert len(lengths) <= 2`。
- 已确认 B-ReKV-S 报错来自 coverage budget 造成多层 KV 长度档位超过 shift-back 当前实现假设；目前先默认绕开 B-ReKV-S。
- 后续若要补 B-ReKV-S，需要专门修复 shift-back 的 dynamic-cache 长度处理；当前论文先不把它作为必跑项。

优先级：中。适合机制附录，但不如 fairness / deletion / cost 紧急。

## 2. 审稿风险实验

### 2.1 Query-Aware Fairness

审稿人可能质疑：

- ReKV 用了 receiver query 信息，而 KVComm 是 query-blind。
- ReKV 的提升可能只是因为多拿了 query 信息。

已完成：

- Pair #1 的 `hotpotqa`, `musique`, `multifieldqa_en`。
- ReKV vs Evict/ValueNorm vs Random-token。
- Query sketch window ablation：`4/8/16/32/all`。
- Query sketch 通信开销统计。

推荐补充：

- 在 pair #6 和 pair #7 上重复轻量版 fairness。
- 任务：`hotpotqa`, `musique`, `multifieldqa_en`。
- 设置：
  - ReKV `r=0.3`, `recv_window=8/16`
  - Evict/ValueNorm `r=0.3`
  - Random-token `r=0.3`

优先级：高。

### 2.2 Cost / Efficiency

审稿人可能质疑：

- receiver scoring 和 query sketch 会带来额外开销。
- 通信节省可能被 generation latency 掩盖。

已完成：

- Pair #1 的 8 个主任务完整 cost profile。
- 产物：`snapshots/cost_profile/pair1_llama31_same_all8_full/cost_table.csv`。
- 已包含 KV payload、timing、memory、output tokens。
- Pair #6/#7 的 full cost profile 已完成，覆盖 8 个主任务和 10 个 method block：
  - `snapshots/cost_profile/table1_pair6_llama32_abliterated_deepseek3b_full/cost_table.csv`
  - `snapshots/cost_profile/table1_pair7_qwen25_uncensored_bespoke_full/cost_table.csv`
  - 日志：`snapshots/cost_profile/logs/gpu2_pair6_pair7_full_cost_0702_1256.log`，最终 `DONE 2026-07-04 04:21:54`。

后续整理：

- 从 pair #6/#7 的 full cost 表中抽取论文需要的轻量子集：
  - 任务：`hotpotqa`, `musique`, `multifieldqa_en`。
  - 方法：ReKV-w8/w16 `r=0.3`，B-ReKV `w8 t0.95 s0.75/s0.85`。

优先级：已完成实验，剩余是表格整理和写作。

### 2.3 B-ReKV Robustness / Pareto

审稿人可能质疑：

- B-ReKV 只是 cherry-pick 了 `tau=0.95, scale=0.75`。

已完成：

- Pair #1 的 `musique`, `hotpotqa`, `multifieldqa_en`。
- 已扫 window / tau / scale。
- Pareto 图：
  - `snapshots/musique/coverage_pareto.png`
  - `snapshots/hotpotqa/coverage_pareto.png`
  - `snapshots/multifieldqa_en/coverage_pareto.png`
- Budget distribution：
  - `snapshots/brekv_budget_distribution.png`
  - `snapshots/brekv_budget_distribution_summary.csv`
- Pair #6/#7 小网格已完成。
  - 任务：`hotpotqa`, `musique`。
  - 结果目录：
    - `snapshots/table1_pair6_llama32_abliterated_deepseek3b/{hotpotqa,musique}/coverage/`
    - `snapshots/table1_pair7_qwen25_uncensored_bespoke/{hotpotqa,musique}/coverage/`
  - 网格：
  - `window={8,16}`
  - `tau={0.90,0.95,0.98}`
  - `scale={0.65,0.75,0.85}`

后续整理：需要给 pair #6/#7 小网格生成或适配 Pareto 图 / summary 表。

### 2.4 Interpretability / Evidence

审稿人可能质疑：

- attention 不是 explanation。
- selected tokens 不一定是因果证据。

已完成：

- Pair #1 的 answer-term overlap。
- 清洗后的 qualitative examples 和 token bar figures：
  - `snapshots/interpretability/pair1_llama31_same/cleaned/clean_interpretability_examples.md`
  - `snapshots/interpretability/pair1_llama31_same/cleaned/*_clean_top_tokens.png`

Deletion ablation 已完成：

- 脚本：`scripts/run_deletion_ablation_gpu2.sh`。
- 产物：
  - `snapshots/deletion_ablation/pair1_llama31_same/deletion_ablation_summary_w8_r0.3_k20.csv`
  - `snapshots/deletion_ablation/pair1_llama31_same/{hotpotqa,musique,multifieldqa_en}/deletion_ablation_w8_r0.3_k20.jsonl`
- 设置：pair #1, `recv_window=8`, `r=0.3`, 每任务 50 samples, 删除 top-20 content tokens。
- 结果：
  - HotpotQA: base `0.78`; 删除 ReKV tokens 后 `0.42`, drop `0.36`; 删除 Evict/Random 后 drop `0.16/0.14`。
  - MuSiQue: base `0.58`; 删除 ReKV tokens 后 `0.22`, drop `0.36`; 删除 Evict/Random 后 drop `0.14/0.04`。
  - MultiFieldQA-en: base `0.48`; 删除 ReKV tokens 后 `0.24`, drop `0.24`; 删除 Evict/Random 后 drop `0.00/-0.02`。
- 结论：删除 ReKV-selected tokens 的性能下降最大，说明 ReKV 选到的 token 更接近实际证据，不只是 lexical overlap。

推荐补充：

1. Supporting-facts overlap。
   - 优先 HotpotQA。
   - 测 selected tokens 是否落在 supporting sentences 内。

优先级：supporting facts 中高。

## 3. 方法消融

### 3.1 Sink / Recent Token Ablation

问题：

- ReKV 是不是靠固定保留 sink/recent prompt tokens 才有效？

状态：已完成。

产物：

- 脚本：`scripts/run_gpu7_mechanism_extended_full_queue.sh`。
- 日志：`snapshots/mechanism/logs/gpu7_mechanism_extended_full_0703_1144.log`。
- 结果目录：`snapshots/mechanism/pair1_llama31_same/sink_recent/`。
- 覆盖 8 个主任务；每个任务 ReKV-w8 `r=0.3` 和 B-ReKV `w8 t0.95 s0.75`，四组 sink/recent，共 64 个 `per_sample.jsonl`。

设置：

- `sink=0, recent=0`
- `sink=4, recent=0`
- `sink=0, recent=8`
- `sink=4, recent=8`，默认设置

任务：

- `hotpotqa`
- `musique`

方法：

- ReKV-w8 `r=0.3`
- B-ReKV `w8 t0.95 s0.75`

优先级：中。

### 3.2 Receiver Window 跨模型对消融

已完成：

- Pair #1 query fairness 已有 `recv_window={4,8,16,32,all}`。

推荐补充：

- Pair #6/#7：
  - `hotpotqa`
  - `musique`
  - `multifieldqa_en`
  - `recv_window={4,8,16,32,all}`
  - `r=0.3`

优先级：高。如果 pair #6/#7 fairness extension 已覆盖，就不需要单独再跑。

### 3.3 Layer Aggregation Ablation

问题：

- 哪些 receiver-attention layers 对 token scoring 最有用？

实际设置：

- `identity`：原始 ReKV，A 的每层使用对应 B 层 attention。
- `last`：所有 A 层使用 B 最后一层 attention。
- `mean`：所有 A 层使用 B 全层平均 attention。
- `top4`：使用 4 个最集中 B 层的平均 attention。
- `last4`：使用 B 最后 4 层平均 attention。

状态：已完成。

产物：

- 脚本：`scripts/run_layer_aggregation_ablation_gpu5.sh`。
- 日志：`snapshots/layer_aggregation_ablation/logs/gpu5_layer_aggregation_ablation_0708_1912.log`。
- 结果目录：`snapshots/layer_aggregation_ablation/`。
- 汇总：
  - `snapshots/analysis/latest_experiments/layer_aggregation_summary.csv`
  - `snapshots/analysis/latest_experiments/layer_aggregation_best_by_task_method.csv`
  - `snapshots/analysis/latest_experiments/figures/layer_aggregation_heatmap.png`

关键结果：

- HotpotQA：`identity 0.700` 最强，`top4 0.682`、`last4 0.668`、`mean 0.662`，`last 0.516` 明显下降。
- MuSiQue：`identity 0.480` 最强，`top4 0.474`、`mean 0.466` 接近，`last 0.290` 明显下降。
- Tipsheets/Countries：`identity` 或 `mean` 最稳。
- 2WikiMQA / MultiFieldQA-en 上 `top4/last4/mean` 有小幅正向，但整体没有推翻原始 paired-layer 设计。

结论：ReKV 的原始 paired-layer receiver attention 是一个稳健默认选择；只用最后层 attention 不稳定，容易在多跳任务上丢证据。

### 3.4 Score Function Ablation

问题：

- 纯 receiver attention 是否足够？要不要加入 value magnitude / recency？

实际设置：

- `value_norm`：query-agnostic ValueNorm / Evict。
- `random`：随机 token baseline。
- `receiver`：原始 ReKV receiver attention。
- `receiver_x_value_norm`：receiver attention × value norm。
- `receiver_recency`：receiver attention + recency prior。

状态：已完成。

产物：

- 脚本：`scripts/run_score_function_ablation_gpu6.sh`。
- 日志：`snapshots/score_function_ablation/logs/gpu6_score_function_ablation_0708_1902.log`。
- 结果目录：`snapshots/score_function_ablation/`。
- 汇总：
  - `snapshots/analysis/latest_experiments/score_function_summary.csv`
  - `snapshots/analysis/latest_experiments/score_function_best_by_pair_task_method.csv`
  - `snapshots/analysis/latest_experiments/figures/score_function_ablation_best.png`

关键结果：

- Pair #1：HotpotQA 原始 `receiver 0.746` 最强；MuSiQue `receiver_x_value_norm 0.494` 略高于 `receiver 0.484`；MultiFieldQA-en `receiver_x_value_norm 0.533` 略高于 `receiver 0.527`。
- Pair #6：HotpotQA `receiver 0.702` 最强；MuSiQue `receiver_x_value_norm 0.398` 略高于 `receiver 0.390`；MultiFieldQA-en `receiver/value_norm` 都到 `0.487`。
- Pair #7：HotpotQA `receiver_recency 0.474` 略高；MuSiQue `receiver_x_value_norm 0.342` 略高；MultiFieldQA-en `receiver` 与 `receiver_x_value_norm` 同为 `0.427`。

结论：receiver-aware attention 是主要收益来源；value norm / random 不能解释提升。`receiver_x_value_norm` 和 `receiver_recency` 可作为附录增强点，但正文方法保持原始 `receiver` 更干净。

## 4. 失败案例与边界分析

### 4.1 Failure Cases

问题：

- ReKV 什么时候失败？
- B-ReKV 什么时候 under-budget？
- receiver attention 什么时候过于分散？

推荐分析：

- 挑 3-5 个 case：
  - ReKV 成功但 B-ReKV 失败。
  - B-ReKV 低预算成功。
  - 二者都因 attention diffuse 失败。
- 报告：
  - attention concentration
  - chosen budget
  - selected tokens
  - answer score

状态：已完成首版自动分析。

- 产物：
  - `snapshots/analysis/failure_cases/failure_case_summary.csv`
  - `snapshots/analysis/failure_cases/failure_case_examples.csv`
  - `snapshots/analysis/failure_cases/failure_case_report.md`
  - `snapshots/analysis/figures/failure_rate_heatmap.png`
- 说明：当前 failure examples 基于已有 `per_sample.jsonl`，不含 raw response；适合用来挑选后续人工检查样本。如果论文需要更强 case study，再对少量样本补 raw response 和 selected tokens 展示。

### 4.2 Task-Type Sensitivity

问题：

- 哪些任务类型最适合 ReKV / B-ReKV？

任务分组：

- short factual：`countries`, `tipsheets`
- multi-hop：`hotpotqa`, `musique`, `twowikimqa`
- long document：`qasper`, `multifieldqa_en`
- math/reasoning：`tmath`

状态：已完成首版分析。

- 产物：
  - `snapshots/analysis/task_type_sensitivity/task_type_family_summary.csv`
  - `snapshots/analysis/task_type_sensitivity/task_type_run_summary.csv`
  - `snapshots/analysis/task_type_sensitivity/task_type_sensitivity_report.md`
  - `snapshots/analysis/figures/task_type_sensitivity_bar.png`
- 初步结论：ReKV/B-ReKV 的论文主卖点应集中在 evidence-heavy 的 multi-hop / long-document 任务；simple synthetic 任务容易饱和，不适合作为主叙事。

### 4.3 Natural-Language Passing vs ReKV Cost Comparison

目的：

- 回答“为什么不直接让 sender 用自然语言把答案/摘要发给 receiver？”
- 从 token 消耗、通信 payload、时间、显存、准确率四个角度对比 NLD 与 ReKV/B-ReKV。

状态：已完成。

已新增：

- `eval.py`：`NLDEvaluator.test_cost_profile()`，支持 `--do_test_nld --profile_cost`。
- `com.py`：NLD 分支支持 `--profile_cost`。
- 运行脚本：`scripts/run_nld_vs_rekv_cost_gpu6.sh` / `scripts/run_nld_vs_rekv_cost_gpu7.sh`。
- 汇总与绘图脚本：`scripts/summarize_nld_vs_rekv_cost.py`。
- 结果目录：`snapshots/nld_cost_profile/`。
- 汇总与图：
  - `snapshots/analysis/nld_vs_rekv/nld_vs_rekv_report.md`
  - `snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_summary.csv`
  - `snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_focused.csv`
  - `snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_average_by_pair.csv`
  - `snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_cost_overview.png`
  - `snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_accuracy_by_task.png`

推荐设置：

- Pairs：`#1`, `#6`, `#7`。
- Tasks：`hotpotqa`, `musique`, `multifieldqa_en`。
- NLD phase-1 answer length：`128` tokens。
- 指标：
  - 准确率：`score_mean`。
  - 通信 payload：NLD 的 `nld_text_payload_tokens/bytes` vs ReKV 的 `kv_tokens_sent/kv_bytes_sent`。
  - 时间：`t_total`，NLD 另拆 `t_model_a_phase1`, `t_model_b_phase1`, `t_model_b_refine`。
  - 显存：`peak_mem_gb`。

已运行命令：

```bash
GPU=6 PAIRS="1 6 7" TASKS="hotpotqa musique multifieldqa_en" LIMIT=500 \
  bash scripts/run_nld_vs_rekv_cost_gpu6.sh

python scripts/summarize_nld_vs_rekv_cost.py
```

关键结果：

- Pair #1 平均：NLD `0.1865`，ReKV `0.6135`，B-ReKV `0.6236`；NLD `2.64s/sample`，B-ReKV `1.26s/sample`。
- Pair #6 平均：NLD `0.1453`，ReKV `0.4894`，B-ReKV `0.5147`；NLD `1.42s/sample`，B-ReKV `0.66s/sample`。
- Pair #7 平均：NLD `0.0916`，ReKV `0.3524`，B-ReKV `0.3822`；NLD `3.80s/sample`，B-ReKV `1.73s/sample`。
- 显存：NLD 略低或接近 ReKV/B-ReKV，但准确率下降很大，不足以构成优势。

论文表述：

- NLD 是文本级 communication，需要 A 先生成自然语言，再让 B 带着 A 的答案做二次生成；它的通信 payload 可能字节数小，但总推理 token 和 latency 往往更高，并且会引入 A 的生成错误/幻觉。
- ReKV/B-ReKV 传的是 selected KV evidence，不要求 A 生成最终答案；更符合“context holder / memory server 只传证据，receiver 负责推理和对齐”的系统设定。
- 因此这组实验可作为 setting 合理性的强支撑，而不是只和 KVComm 比。

## 5. 推荐执行顺序

### Stage A：补小表洞

1. Table 8 pair #1 的 5 个 fixed ReKV runs 已完成。
   - HotpotQA: `w8 r=0.7` score `0.736`, `w16 r=0.7` score `0.748`。
   - QASPER: `w8 r=0.3/0.5/0.7` score `0.344/0.342/0.344`。

### Stage B：最高审稿风险实验

2. Pair #6/#7 query fairness extension 已完成。
   - 产物：`snapshots/query_fairness/pair6_pair7_query_fairness_brekv_summary.csv`。

3. Deletion ablation 已完成。
   - 产物：`snapshots/deletion_ablation/pair1_llama31_same/deletion_ablation_summary_w8_r0.3_k20.csv`。

4. Pair #6/#7 lightweight cost profile 已完成整理。
   - 任务：`hotpotqa`, `musique`, `multifieldqa_en`。
   - 产物：`snapshots/analysis/cost/pair6_pair7_cost_focus_hotpotqa_musique_multifieldqa.csv`。

5. Pair #6/#7 B-ReKV small robustness grid 已完成整理。
   - 任务：`hotpotqa`, `musique`。
   - 产物：`snapshots/analysis/robustness/pair6_pair7_brekv_robustness_summary.csv`。

### Stage C：附录 / 机制实验

6. Table 6 extended tasks 正在跑。
   - Pair #6 已完成 5 个 extended tasks x 9 runs。
   - Pair #7 正在 GPU7 跑 `hotpotqa_full`，后续继续完整队列。
7. HotpotQA supporting-facts overlap 已完成。
   - ReKV top-20 supporting-facts rate `0.5148`，Evict `0.0638`，Random `0.0555`。
   - 图：`snapshots/analysis/figures/supporting_overlap_bar.png`。
8. Failure case analysis 已完成首版。
9. Task-type sensitivity 已完成首版。
10. Score-function ablation 已完成。
    - 图：`snapshots/analysis/latest_experiments/figures/score_function_ablation_best.png`。
11. Layer aggregation ablation 已完成。
    - 图：`snapshots/analysis/latest_experiments/figures/layer_aggregation_heatmap.png`。
12. Natural-language passing vs ReKV cost comparison。
    - 脚本已准备；等 GPU7 Table 6 队列完成后再跑最合适。

### Stage D：大型可选实验

13. Table 8 pair #4/#5/#9 完整队列已完成。
    - #4 Falcon3 same-model：`snapshots/table8_pair4_falcon3_7b_same/`。
    - #5 EvolCodeLlama -> ToolACE：`snapshots/table8_pair5_evolcodellama_toolace/`。
    - #9 SuperNova -> DeepSeek-Llama-8B：`snapshots/table8_pair9_supernova_deepseek_llama8b/`；暂缓作为正向对比，分数分布 / raw-output / KVComm probe 诊断见 `snapshots/analysis/pair9/pair9_diagnostic_report.md`。
14. 最后再处理 Table 1/Table 8 pair #8 Falcon；若原 checkpoint 仍不可获得，则标记为 checkpoint unavailable，或改跑 Falcon-family substitute 作为额外鲁棒性实验。
15. Table 10 multi-source ReKV。
16. Head-wise B-ReKV。

## 6. 如果最后没法全部跑完，论文怎么说

如果时间有限，最稳的论文呈现方式是：

- 正文：Table 1 pair #6/#7，Table 8 pair #1/#2/#3，cost，fairness，B-ReKV Pareto，interpretability。
- 附录：Table 8 pair #4/#5，pair #9 hard negative / limitation，budget-aware negative ablations，cleaned examples，budget distribution，additional pair tables。
- Future work：Table 8 pair #8 checkpoint-unavailable case，multi-source，B-ReKV-S dynamic-cache shift-back fix，head-wise B-ReKV。

这样写比较诚实：我们不声称完整复现 KVComm 所有 appendix 实验，但已经覆盖核心方法、跨模型泛化、效率、公平性、鲁棒性和可解释性风险。
