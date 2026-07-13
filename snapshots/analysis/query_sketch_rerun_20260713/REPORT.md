# Query-Sketch 论文重跑审计（2026-07-13）

## 口径

- `query_sketch_bf16_v1` / `int8_v1` / `token_ids_v1`：本轮可部署协议。
- `full_kv_oracle_v1`：显式 Full-KV 上界，只用于 oracle-gap，不并入主方法结果。
- `query_agnostic_kv_v1`：ValueNorm / Random 对照。
- Pair #6 主矩阵中缺少 protocol metadata 的 68 个结果位于明确的 Query-Sketch root，标记为 `v0_pre_instrumentation`；可用于准确率，但不能用于新通信计费。
- 其他历史 snapshots 不纳入本报告。

## 完成状态

- Table 1 文件覆盖：ReKV 六配置完成 23/24 个 pair-task 单元。
- Table 1 显式 v1 metadata：ReKV 完成 94/144 runs、15/24 个完整单元；Pair #6 的 v0 结果需单列。
- Table 1 冻结 B-ReKV 主矩阵完成 0/24；完整主块 0/24。
- Oracle gap：18/18 个 matched method-pair-task 单元。
- 表示消融：12 个聚合点（2 pairs × 3 tasks × 3 modes × 4 windows 原始共 72 runs）。
- 新协议独立 cost：0 runs；若为 0，说明第二阶段 cost 尚未启动。

## 冻结配置

- 已接受 `B-ReKV-t0.98-s1-w8`：平均预算 0.5919，matched-budget 平均分差 +0.0250，6/6 持平或获胜，最差分差 +0.0100。
- 注意：配置冻结校准已完成；上面的 0/24 指该冻结配置尚未进入 Table 1 三个 pair × 八任务主矩阵。

## Oracle gap（Query-Sketch − Full-KV Oracle）

| method | mean_score_gap_query_minus_oracle | cells |
|---|---|---|
| B-ReKV | 0.0067 | 9 |
| ReKV | -0.0378 | 9 |

通信节省：
| method | mean_communication_reduction_pct | cells |
|---|---|---|
| B-ReKV | 35.1352 | 9 |
| ReKV | 61.0065 | 9 |

时间与显存：
| method | query_time | oracle_time | latency_delta_pct | query_peak_mem_gb | oracle_peak_mem_gb |
|---|---|---|---|---|---|
| ReKV | 1.0584 | 1.1040 | -4.1239 | 24.5451 | 24.7157 |
| B-ReKV | 1.1264 | 1.1449 | -1.6157 | 24.6956 | 24.7549 |

## 表示与窗口消融

| mode | window | cells | mean_score | mean_b_to_a_kb | mean_total_mb | mean_t_total |
|---|---|---|---|---|---|---|
| bf16 | 4 | 6 | 0.4700 | 848.4766 | 97.4456 | 0.7006 |
| bf16 | 8 | 6 | 0.5467 | 1696.4766 | 98.2737 | 0.6915 |
| bf16 | 16 | 6 | 0.5100 | 3392.4766 | 99.9300 | 0.7197 |
| bf16 | 32 | 6 | 0.4667 | 6784.4766 | 103.2425 | 0.6982 |
| int8 | 4 | 6 | 0.4667 | 424.5938 | 97.0317 | 0.7074 |
| int8 | 8 | 6 | 0.5500 | 848.5938 | 97.4457 | 0.6933 |
| int8 | 16 | 6 | 0.4967 | 1696.5938 | 98.2738 | 0.7210 |
| int8 | 32 | 6 | 0.4700 | 3392.5938 | 99.9301 | 0.7118 |
| token_ids | 4 | 6 | 0.2800 | 0.0234 | 96.6170 | 0.7583 |
| token_ids | 8 | 6 | 0.3667 | 0.0391 | 96.6171 | 0.7136 |
| token_ids | 16 | 6 | 0.4467 | 0.0703 | 96.6171 | 0.7173 |
| token_ids | 32 | 6 | 0.4400 | 0.1328 | 96.6171 | 0.6925 |

结论：INT8-w8 将 BF16-w8 的 B→A sketch 字节减半，平均分不降（0.5500 vs 0.5467）；Token IDs 虽几乎消除 B→A payload，但最佳分数明显更低。BF16/INT8 的 w8 都优于更大窗口。

## 机制消融

| family | setting | mean_score | cells |
|---|---|---|---|
| layer_aggregation | identity | 0.4896 | 3 |
| layer_aggregation | last | 0.3580 | 3 |
| layer_aggregation | last4 | 0.4198 | 3 |
| layer_aggregation | mean | 0.4004 | 3 |
| layer_aggregation | top4 | 0.4400 | 3 |
| score_function | random | 0.3027 | 3 |
| score_function | receiver | 0.4896 | 3 |
| score_function | receiver_recency | 0.4936 | 3 |
| score_function | receiver_x_value_norm | 0.4771 | 3 |
| score_function | value_norm | 0.3596 | 3 |

## 当前不能提前写入论文的部分

- 冻结 B-ReKV 的 24 个 Table-1 主单元尚未全部完成前，不生成最终主表平均值。
- `query_sketch_cost_v1` 尚未完整时，不用 Oracle-gap profile 代替正式 cost 表。
- Pair #6 pre-instrumentation 结果不能用于 bytes / timing 结论。

