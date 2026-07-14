# Deployable Query-Sketch 最终实验汇总（2026-07-14）

## 数据口径

- 主结果只纳入 `_meta.protocol_version=query_sketch_bf16_v1`。
- Canonical B-ReKV 固定为 calibrated coverage `tau=0.95, scale=0.75, window=8`。
- `full_kv_oracle_v1` 只用于 Oracle-gap；v0/legacy 不进入平均值。

## 完整度

- 主任务 fixed ReKV 完整单元：24/24；canonical B-ReKV：24/24；完整七点单元：24/24。
- Extended tasks 完整七点单元：10/10。
- Appendix model settings 完整七点单元：32/32。
- Multi-source：3/3 tasks。
- 正式 cost：18/18（3/3 pairs）；canonical B-ReKV Oracle-gap：3/3。

## 主任务模型设置平均

| pair_label | complete_cells | cells | mean_best_rekv_score | mean_brekv_score | mean_brekv_budget | mean_gap_brekv_minus_best |
|---|---|---|---|---|---|---|
| S: Llama-3.1-8B; R: Llama-3.1-8B | 8 | 8 | 0.5466 | 0.4734 | 0.3019 | -0.0732 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | 8 | 8 | 0.5052 | 0.4071 | 0.3185 | -0.0981 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | 8 | 8 | 0.4313 | 0.3702 | 0.3734 | -0.0611 |

## Extended tasks 模型设置平均

| pair_label | complete_cells | cells | mean_best_rekv_score | mean_brekv_score | mean_brekv_budget | mean_gap_brekv_minus_best |
|---|---|---|---|---|---|---|
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | 5 | 5 | 0.4339 | 0.3537 | 0.2609 | -0.0802 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | 5 | 5 | 0.3692 | 0.2899 | 0.3475 | -0.0793 |

## Appendix model settings 平均

| pair_label | complete_cells | cells | mean_best_rekv_score | mean_brekv_score | mean_brekv_budget | mean_gap_brekv_minus_best |
|---|---|---|---|---|---|---|
| S: Llama-3.2-3B; R: Llama-3.2-3B | 8 | 8 | 0.5089 | 0.4102 | 0.3127 | -0.0987 |
| S: Qwen2.5-7B; R: Qwen2.5-7B | 8 | 8 | 0.5087 | 0.3966 | 0.3512 | -0.1121 |
| S: Falcon3-7B; R: Falcon3-7B | 8 | 8 | 0.4780 | 0.3499 | 0.2717 | -0.1281 |
| S: EvolCodeLlama-8B; R: ToolACE-8B | 8 | 8 | 0.5059 | 0.3977 | 0.3214 | -0.1081 |

## Multi-source

| task | runs_done | complete | best_score | best_budget | best_config |
|---|---|---|---|---|---|
| hotpotqa | 6 | True | 0.6680 | 0.7091 | w16-r0.7 |
| musique | 6 | True | 0.4500 | 0.7095 | w8-r0.7 |
| twowikimqa | 6 | True | 0.4400 | 0.7094 | w8-r0.7 |

## 正式 Cost / Efficiency（pair 平均）

| pair_label | cells | mean_score | mean_budget | mean_b_to_a_mb | mean_total_mb | mean_t_total_s | mean_peak_mem_gb |
|---|---|---|---|---|---|---|---|
| S: Llama-3.1-8B; R: Llama-3.1-8B | 6 | 0.5200 | 0.3076 | 2.0005 | 107.7118 | 0.3888 | 31.1574 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | 6 | 0.5300 | 0.3114 | 1.3129 | 102.5360 | 0.5144 | 13.1030 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | 6 | 0.3433 | 0.3441 | 1.5317 | 51.9473 | 1.5489 | 29.4575 |

## 正式 Cost / Efficiency（逐 cell）

| pair_label | task | method | score | budget | b_to_a_mb | a_to_b_mb | total_mb | t_total_s | peak_mem_gb |
|---|---|---|---|---|---|---|---|---|---|
| S: Llama-3.1-8B; R: Llama-3.1-8B | hotpotqa | B-ReKV | 0.6200 | 0.2736 | 2.0005 | 4.8321 | 6.8326 | 0.2031 | 30.0490 |
| S: Llama-3.1-8B; R: Llama-3.1-8B | hotpotqa | ReKV | 0.7200 | 0.3222 | 2.0005 | 5.7109 | 7.7114 | 0.1617 | 30.0502 |
| S: Llama-3.1-8B; R: Llama-3.1-8B | multifieldqa_en | B-ReKV | 0.4400 | 0.3330 | 2.0005 | 308.9944 | 310.9949 | 0.7569 | 33.3063 |
| S: Llama-3.1-8B; R: Llama-3.1-8B | multifieldqa_en | ReKV | 0.4800 | 0.3219 | 2.0005 | 290.0435 | 292.0440 | 0.7505 | 33.2661 |
| S: Llama-3.1-8B; R: Llama-3.1-8B | musique | B-ReKV | 0.4200 | 0.2730 | 2.0005 | 11.2400 | 13.2405 | 0.2429 | 30.1341 |
| S: Llama-3.1-8B; R: Llama-3.1-8B | musique | ReKV | 0.4400 | 0.3218 | 2.0005 | 13.4469 | 15.4474 | 0.2179 | 30.1388 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | hotpotqa | B-ReKV | 0.6200 | 0.2520 | 1.3129 | 3.1751 | 4.4881 | 0.2303 | 12.0459 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | hotpotqa | ReKV | 0.7000 | 0.3249 | 1.3129 | 4.1496 | 5.4625 | 0.2143 | 12.0471 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | multifieldqa_en | B-ReKV | 0.4600 | 0.3962 | 1.3129 | 325.4373 | 326.7502 | 1.0771 | 15.2087 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | multifieldqa_en | ReKV | 0.4800 | 0.3250 | 1.3129 | 255.3604 | 256.6733 | 1.0523 | 15.0636 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | musique | B-ReKV | 0.4200 | 0.2456 | 1.3129 | 8.2250 | 9.5380 | 0.2886 | 12.1235 |
| S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B | musique | ReKV | 0.5000 | 0.3249 | 1.3129 | 10.9908 | 12.3037 | 0.2240 | 12.1293 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | hotpotqa | B-ReKV | 0.5000 | 0.3820 | 1.5317 | 2.9244 | 4.4561 | 1.0713 | 28.4998 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | hotpotqa | ReKV | 0.4600 | 0.3251 | 1.5317 | 2.5116 | 4.0433 | 0.8431 | 28.5004 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | multifieldqa_en | B-ReKV | 0.3000 | 0.3610 | 1.5317 | 152.6167 | 154.1484 | 2.6133 | 31.3180 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | multifieldqa_en | ReKV | 0.3400 | 0.3250 | 1.5317 | 131.9725 | 133.5042 | 2.2856 | 31.2753 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | musique | B-ReKV | 0.2400 | 0.3465 | 1.5317 | 6.3239 | 7.8556 | 1.4898 | 28.5767 |
| S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B | musique | ReKV | 0.2200 | 0.3249 | 1.5317 | 6.1446 | 7.6763 | 0.9903 | 28.5749 |

## Canonical B-ReKV vs Full-KV Oracle

| task | query_score | oracle_score | score_gap_query_minus_oracle | query_budget | communication_reduction_pct | query_time_s | oracle_time_s |
|---|---|---|---|---|---|---|---|
| hotpotqa | 0.6200 | 0.7800 | -0.1600 | 0.2520 | 64.8278 | 0.2241 | 0.2153 |
| musique | 0.4200 | 0.5400 | -0.1200 | 0.2456 | 71.7735 | 0.2253 | 0.2247 |
| multifieldqa_en | 0.4600 | 0.4200 | 0.0400 | 0.3962 | 58.3738 | 1.0716 | 1.1317 |

## 论文结论边界

- 本机 Query-Sketch 主矩阵、扩展表、附录模型设置、Multi-Source、正式 cost 与 canonical Oracle-gap 均已完整。
- Canonical B-ReKV 主任务平均实际预算约 33.1%，明显低于 high-fixed ReKV。
- B-ReKV 的主张是约 30% 左右动态预算折中，以及显著优于 query-agnostic baselines；通常低于 best fixed ReKV，但这是预期 tradeoff。
- NLD 原始 baseline 可复用；ReKV/B-ReKV 侧必须用本报告正式 cost v1 结果替换旧 cost_profile。
