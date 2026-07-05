# ReKV / B-ReKV 实验清单

这份文档是论文实验的总 checklist。目标是把“已经完成的证据”和“还需要补的实验”分开记录，并说明每个实验为什么重要。后续每跑完一批实验，就更新这里的状态。

命名约定：

- **ReKV**：Receiver-aware KV Communication，固定预算的 receiver-aware token selection。
- **B-ReKV**：budget-aware ReKV family，目前实现是 receiver-attention coverage 动态预算。
- **KVComm**：原论文的 layer-level KV communication baseline。除非特别说明，KVComm baseline 可以直接引用原论文结果，不需要本地复现。

## 0. 当前总状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 主表 ReKV 覆盖 | 已完成扩展覆盖 | Table 1 pair #6/#7 完整；Table 8 pair #1/#2/#3/#4/#5/#9 完整。 |
| B-ReKV 稳健性 | pair #1 核心任务已完成 | MuSiQue / HotpotQA / MultiFieldQA-en 的 Pareto 图和 budget distribution 已完成。 |
| Cost profiling | pair #1 已完成 | 8 个主数据集都有 cost 表。pair #6/#7 的 cost 是推荐补充项。 |
| Query-aware fairness | pair #1/#6/#7 已完成 | pair #1 完成 ReKV vs Evict vs Random 和 query sketch window ablation；pair #6/#7 完成 ReKV/Evict/Random/B-ReKV 扩展。 |
| 可解释性 | pair #1 已完成 | answer overlap、清洗后的 qualitative examples、token bar 图、deletion ablation 已完成。 |
| Table 1 pair #8 Falcon | 暂缓/最后处理 | `Falcon3-7B-Instruct-abliterated` checkpoint 目前不可获得或目录不完整；不阻塞主线，放到最后可选处理。 |
| Table 6 / 10 / 11 | 未完成 | extended tasks、multi-source、positional coherence 都是附录/机制级实验。 |

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
| #9 SuperNova -> DeepSeek-Llama-8B | `snapshots/table8_pair9_supernova_deepseek_llama8b/` | 已完成 | 8 datasets x 9 paper-table runs 完整；除 `tmath` 外多数任务结果接近 0，需作为异常/负结果单独解释。 |

建议：

- pair #1/#2/#3/#4/#5/#9 已可作为 Table 8 appendix 覆盖。
- pair #8 仍受 Falcon receiver checkpoint blocker 影响；原 checkpoint 不可获得时可写入 limitation。

### 1.3 Table 6：Extended Tasks

目的：在 KVComm extended datasets 上验证鲁棒性。

状态：未做。

推荐轻量版：

- 选 2-3 个 extended tasks，不跑完整 extended suite。
- 每个任务跑：
  - ReKV-w8 `r=0.3/0.5`
  - ReKV-w16 `r=0.3/0.5`
  - B-ReKV `w8 t0.95 s0.75`
  - B-ReKV `w8 t0.95 s0.85`

优先级：中。属于 appendix robustness，不需要早于核心审稿风险实验。

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

状态：ReKV/B-ReKV 版本未做。

可做的 ReKV 版本：

- ReKV normal vs ReKV-S：破坏或关闭 position-shift coherence。
- B-ReKV normal vs B-ReKV-S。
- 任务：`hotpotqa`, `musique`。
- 先用 pair #1。

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

推荐补充：

- Pair #6/#7 的轻量 cost profile。
- 任务：`hotpotqa`, `musique`, `multifieldqa_en`。
- 方法：
  - ReKV-w8 `r=0.3`
  - ReKV-w16 `r=0.3`
  - B-ReKV `w8 t0.95 s0.75`
  - B-ReKV `w8 t0.95 s0.85`

优先级：高。它能支撑跨模型 pair 下的 overhead 结论。

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

推荐补充：

- Pair #6/#7 小网格。
- 任务：`hotpotqa`, `musique`。
- 网格：
  - `window={8,16}`
  - `tau={0.90,0.95,0.98}`
  - `scale={0.65,0.75,0.85}`

优先级：中高。

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

设置：

- all layers
- last 8 layers
- first half
- second half
- middle layers

任务：

- `hotpotqa`
- `musique`

优先级：中低。适合机制附录，但可以后放。

### 3.4 Score Function Ablation

问题：

- 纯 receiver attention 是否足够？要不要加入 value magnitude / recency？

设置：

- receiver attention
- value norm
- receiver attention x value norm
- receiver attention + recency prior

任务：

- `hotpotqa`
- `musique`
- `multifieldqa_en`

优先级：中。用于解释为什么选择当前 score。

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

优先级：中。主要是分析，基本不需要 GPU。

### 4.2 Task-Type Sensitivity

问题：

- 哪些任务类型最适合 ReKV / B-ReKV？

任务分组：

- short factual：`countries`, `tipsheets`
- multi-hop：`hotpotqa`, `musique`, `twowikimqa`
- long document：`qasper`, `multifieldqa_en`
- math/reasoning：`tmath`

状态：

- 可直接基于已有结果分析。

优先级：高，写论文时要做，不需要新跑。

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

4. Pair #6/#7 lightweight cost profile。
   - 任务：`hotpotqa`, `musique`, `multifieldqa_en`。

5. Pair #6/#7 B-ReKV small robustness grid。
   - 任务：`hotpotqa`, `musique`。

### Stage C：附录 / 机制实验

6. HotpotQA supporting-facts overlap。
7. Sink/recent token ablation。
8. Failure case analysis。
9. Table 6 extended tasks 轻量版。
10. Table 11 positional coherence ablation。

### Stage D：大型可选实验

11. Table 8 pair #4/#5/#9 完整队列已完成。
    - #4 Falcon3 same-model：`snapshots/table8_pair4_falcon3_7b_same/`。
    - #5 EvolCodeLlama -> ToolACE：`snapshots/table8_pair5_evolcodellama_toolace/`。
    - #9 SuperNova -> DeepSeek-Llama-8B：`snapshots/table8_pair9_supernova_deepseek_llama8b/`；结果大多接近 0，需检查是否为模型输出/模板/对齐问题。
12. 最后再处理 Table 1/Table 8 pair #8 Falcon；若原 checkpoint 仍不可获得，则标记为 checkpoint unavailable，或改跑 Falcon-family substitute 作为额外鲁棒性实验。
13. Table 10 multi-source ReKV。
14. Head-wise B-ReKV。

## 6. 如果最后没法全部跑完，论文怎么说

如果时间有限，最稳的论文呈现方式是：

- 正文：Table 1 pair #6/#7，Table 8 pair #1/#2/#3，cost，fairness，B-ReKV Pareto，interpretability。
- 附录：Table 8 pair #4/#5/#9，budget-aware negative ablations，cleaned examples，budget distribution，additional pair tables。
- Future work：Table 8 pair #8 checkpoint-unavailable case，multi-source，positional-coherence variants，head-wise B-ReKV。

这样写比较诚实：我们不声称完整复现 KVComm 所有 appendix 实验，但已经覆盖核心方法、跨模型泛化、效率、公平性、鲁棒性和可解释性风险。
