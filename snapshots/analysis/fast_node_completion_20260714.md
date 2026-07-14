# Fast-node Query-Sketch v1 完成与结果报告（2026-07-14）

本报告记录 `scripts/run_remaining_fast_gpu0_1.sh` 在本机完成的全部实验。
统计只接受显式协议元数据和目标参数，不把旧 Oracle、高预算 B-ReKV 或
`v0_pre_instrumentation` 结果混入。

## 1. 完成度与数据质量

- 总目标：123 unique cells。
- 完成：123/123（100%）。
- GPU0：54/54；GPU3：69/69。
- 最后一个有效产物：2026-07-14 09:07:17。
- 99 个 `per_sample.jsonl` 和 24 个 `cost_summary.json` 均可解析。
- 单源协议：`query_sketch_bf16_v1`。
- 多源协议：`query_sketch_bf16_multi_source_v1`。
- Oracle 协议：`full_kv_oracle_v1`。
- 主 B-ReKV 参数：`tau=0.95, scale=0.75, window=8`。
- 有效缺失、有效重复、参数不符：均为 0。

| 模块 | 完成度 | 说明 |
|---|---:|---|
| Table 10 Multi-Source | 18/18 | 三任务 × 两窗口 × 三比例 |
| Table 6 pair #7 | 35/35 | 五任务 ×（六 ReKV + 一 B-ReKV） |
| Table 8 pair #5 前四任务 | 28/28 | 四任务 × 七配置 |
| Table 1 pair #1 | 56/56 | 48 ReKV + 8 main B-ReKV |
| Table 1 pair #7 | 56/56 | 48 ReKV + 8 main B-ReKV |
| Cost v1 pair #1/#7 | 12/12 | 三任务 × 两方法 × 两 pair |
| Main-B Oracle gap pair #1/#7 | 12/12 | 三任务 × QS/Oracle × 两 pair |

三个无最终产物的中断目录均已有唯一有效替代：SAMSum 两次网络/人工中断，
以及 Pair #7 TMath w16-r0.5 一次中断。它们不计入完成度。

## 2. Table 10：真正 Multi-Source Query-Sketch

全部 7200 个样本均为 `query_sketch_bf16_multi_source_v1`。

| Task | Best config | Best score | Actual budget |
|---|---|---:|---:|
| HotpotQA | w16-r0.7 | 0.6680 | 0.7091 |
| MuSiQue | w8-r0.7 | 0.4500 | 0.7095 |
| 2WikiMQA | w8-r0.7 | 0.4400 | 0.7094 |

- 三任务最佳分数宏平均：0.5193。
- w8-r0.7 三任务宏平均：0.5173。
- w8 与 w16 总体接近，ratio 是主要影响因素。
- 相对 legacy oracle-style prototype：HotpotQA -0.012、MuSiQue -0.012、
  2WikiMQA +0.010。两者协议不同，不能当重复实验求平均。

## 3. Table 6 pair #7：Extended Tasks

| Task | Best ReKV score / budget | Main B-ReKV score / budget |
|---|---:|---:|
| HotpotQA-full | 0.5603 / 0.7106 | 0.3951 / 0.3858 |
| MuSiQue-full | 0.3608 / 0.7107 | 0.2321 / 0.3454 |
| Qasper-full | 0.2341 / 0.7107 | 0.1744 / 0.3697 |
| RepoBench | 0.3528 / 0.7107 | 0.3405 / 0.2361 |
| SAMSum | 0.3381 / 0.7105 | 0.3073 / 0.4007 |

- Best ReKV 宏平均：0.3692。
- Main B-ReKV 宏平均：0.2899，平均实际预算 0.3475。
- B-ReKV 相对每任务 best ReKV 平均少 0.0793 分，但预算低 51.09%。
- RepoBench 最稳：B-ReKV 仅少 0.0123 分，预算从 0.7107 降至 0.2361。
- SAMSum B-ReKV 相对同配置 legacy 提升 0.0145；RepoBench 基本持平。
- HotpotQA/MuSiQue-full 的 B-ReKV 精度损失较大，属于方法边界。

## 4. Table 8 pair #5：前四任务

| Task | Best ReKV score / budget | Main B-ReKV score / budget |
|---|---:|---:|
| Countries | 0.5950 / 0.5161 | 0.2700 / 0.2951 |
| Tipsheets | 0.9080 / 0.5156 | 0.8240 / 0.2895 |
| HotpotQA | 0.6220 / 0.7094 | 0.4100 / 0.2886 |
| Qasper | 0.2460 / 0.7094 | 0.2220 / 0.3836 |

- Best ReKV 宏平均：0.5928。
- Main B-ReKV 宏平均：0.4315，平均实际预算 0.3142。
- B-ReKV 平均预算低 48.71%。
- Tipsheets、Qasper 是较稳的低预算点；Countries、HotpotQA 精度损失明显。

## 5. Table 1：Pair #1 和 Pair #7

### Pair #1

| Task | Best ReKV | Main B-ReKV | B budget |
|---|---:|---:|---:|
| Countries | 0.6050 | 0.6150 | 0.2944 |
| Tipsheets | 0.9120 | 0.7120 | 0.2781 |
| HotpotQA | 0.7320 | 0.5600 | 0.2754 |
| Qasper | 0.3300 | 0.3100 | 0.3525 |
| MuSiQue | 0.4640 | 0.3040 | 0.2745 |
| MultiFieldQA | 0.5467 | 0.5000 | 0.3354 |
| 2WikiMQA | 0.4250 | 0.4300 | 0.3397 |
| TMath | 0.3582 | 0.3559 | 0.2648 |

- Best ReKV 宏平均：0.5466。
- Main B-ReKV：0.4734 @ budget 0.3019。
- B-ReKV 平均少 0.0732 分，预算低 50.73%。
- Countries、2WikiMQA 略优于 best ReKV；TMath 基本持平。

### Pair #7

| Task | Best ReKV | Main B-ReKV | B budget |
|---|---:|---:|---:|
| Countries | 0.5000 | 0.4850 | 0.4008 |
| Tipsheets | 0.9540 | 0.9160 | 0.3980 |
| HotpotQA | 0.4820 | 0.3520 | 0.3853 |
| Qasper | 0.2180 | 0.1720 | 0.3684 |
| MuSiQue | 0.3500 | 0.2240 | 0.3441 |
| MultiFieldQA | 0.4200 | 0.3133 | 0.3585 |
| 2WikiMQA | 0.1900 | 0.1700 | 0.3417 |
| TMath | 0.3365 | 0.3294 | 0.3907 |

- Best ReKV 宏平均：0.4313。
- Main B-ReKV：0.3702 @ budget 0.3734。
- B-ReKV 平均少 0.0611 分，预算低 41.49%。
- Countries、Tipsheets、TMath 较稳；多跳任务损失较明显。

## 6. 正式 Query-Sketch cost v1

Cost 只测前 50 个样本并含 3 个 warmup，不能替代完整主表分数。

| Pair / Task | ReKV total bytes | B-ReKV total bytes | ReKV time | B-ReKV time |
|---|---:|---:|---:|---:|
| #1 HotpotQA | 8.09 MB | 7.16 MB | 161.7 ms | 203.1 ms |
| #1 MuSiQue | 16.20 MB | 13.88 MB | 217.9 ms | 242.9 ms |
| #1 MultiFieldQA | 306.23 MB | 326.10 MB | 750.5 ms | 756.9 ms |
| #7 HotpotQA | 4.24 MB | 4.67 MB | 843.1 ms | 1071.3 ms |
| #7 MuSiQue | 8.05 MB | 8.24 MB | 990.3 ms | 1489.8 ms |
| #7 MultiFieldQA | 139.99 MB | 161.64 MB | 2285.6 ms | 2613.3 ms |

- BF16 sketch 的固定 B→A 开销：Pair #1 约 2.10 MB/样本；Pair #7
  约 1.61 MB/样本。
- B-ReKV 不保证总通信量总低于 fixed ReKV r0.3：当实际预算高于 r0.3
  或上下文很长时，A→B KV payload 可能更高。
- Pair #7 的 B-ReKV scoring/timing 开销明显高于 Pair #1。
- MultiFieldQA 通信量大来自长上下文，不是计费异常。

## 7. Main B-ReKV Query-Sketch vs Full-KV Oracle

这里只统计当前主配置 `t0.95-s0.75-w8`，不混入旧高预算 B-ReKV。

| Pair / Task | QS score | Oracle score | Score gap | Communication reduction |
|---|---:|---:|---:|---:|
| #1 HotpotQA | 0.62 | 0.76 | -0.14 | 61.42% |
| #1 MuSiQue | 0.42 | 0.58 | -0.16 | 68.28% |
| #1 MultiFieldQA | 0.44 | 0.52 | -0.08 | 65.45% |
| #7 HotpotQA | 0.50 | 0.58 | -0.08 | 42.23% |
| #7 MuSiQue | 0.24 | 0.44 | -0.20 | 58.38% |
| #7 MultiFieldQA | 0.30 | 0.42 | -0.12 | 61.97% |

- 六单元平均 score gap：-0.1300。
- 六单元平均通信节省：59.62%。
- Pair #1 平均 gap -0.1267、通信节省 65.05%。
- Pair #7 平均 gap -0.1333、通信节省 54.19%。
- QS 与 Oracle 的实际动态预算也不同，因此该 gap 同时包含 query-sketch
  评分近似与预算分配变化，不是严格 matched-budget selector gap。
- 旧报告中的 B-ReKV 平均 gap `+0.0067` 来自旧高预算配置，不能覆盖本结果。

## 8. 更新后的全局完成度

| 主模块 | 当前完成度 | 剩余 |
|---|---:|---:|
| Stage 3 | 360/360 | 0 |
| Table 1 fixed ReKV v1 | 97/144 | Pair #6 缺 47 |
| Table 1 main B-ReKV | 16/24 | Pair #6 缺 8 |
| Table 1 合计 | 113/168 | Pair #6 缺 55 |
| Table 6 v1 | 35/70 | Pair #6 缺 35 |
| Table 8 v1 | 28/224 | Pair #2/#3/#4 + Pair #5 后四任务缺 196 |
| Table 10 v1 | 18/18 | 0 |
| Cost v1 | 12/18 | Pair #6 缺 6 |
| Main-B Oracle gap | 12/18 | Pair #6 缺 6 |

本机任务已全部完成；剩余结果需要从另一台机器运行或同步。
