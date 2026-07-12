# Table 6 Pair #7 RepoBench Summary

## Status and Setup

- Status: complete, 9/9 runs.
- Samples: 1000 per run.
- Sender: `/NAS/models/Qwen2.5-7B-Instruct-Uncensored`.
- Receiver: `/NAS/models/Bespoke-Stratos-7B`.
- Dataset: local `datasets/RepoBench`, split `cross_file_first`.
- Metric: RepoBench next-line edit similarity.
- Final queue: `scripts/run_table6_pair7_repobench_4gpu.sh`, GPUs 0–3.
- Run logs: `snapshots/table6_pair7_qwen25_uncensored_bespoke/logs/*0712_1521.log`.

The earlier GPU7 attempt failed at sample 379/1000 because receiver-attention scoring exhausted a 48 GB A6000. The final four-GPU rerun used 98 GB RTX PRO 6000 cards and completed without OOM.

## Full Results

| Method | Window | Ratio / scale | Mean score | Actual KV budget | Query budget |
|---|---:|---:|---:|---:|---:|
| ReKV | 8 | r=0.3 | 0.34851 | 0.325019 | — |
| ReKV | 8 | r=0.5 | 0.35113 | 0.517835 | — |
| **ReKV** | **8** | **r=0.7** | **0.35302** | 0.710703 | — |
| ReKV | 16 | r=0.3 | 0.34872 | 0.325019 | — |
| ReKV | 16 | r=0.5 | 0.34740 | 0.517835 | — |
| ReKV | 16 | r=0.7 | 0.35073 | 0.710703 | — |
| **B-ReKV** | **8** | **t=0.95, s=0.75** | **0.34001** | **0.159448** | 0.128231 |
| B-ReKV | 8 | t=0.95, s=0.85 | 0.33572 | 0.173439 | 0.142756 |
| B-ReKV | 16 | t=0.95, s=0.90 | 0.32132 | 0.182462 | 0.152154 |

Source: `snapshots/analysis/latest_experiments/table6_extended_summary.csv`.

## Findings

1. Best fixed-budget result is ReKV-w8 r=0.7 at 0.35302.
2. Fixed ReKV is relatively insensitive to budget on RepoBench. Moving from w8 r=0.3 to r=0.7 increases actual KV budget from 0.3250 to 0.7107 (2.19x) but improves score by only 0.00451.
3. ReKV-w8 is slightly more reliable than w16: its score increases monotonically with ratio, while w16 r=0.5 dips to 0.34740.
4. Best B-ReKV reaches 0.34001 with only 0.15945 actual KV budget. Relative to ReKV-w8 r=0.3, it uses about 49% as much KV for an absolute score drop of 0.00850.
5. B-ReKV is not monotonic across the tested scale/window settings. The smallest tested configuration, w8 t=0.95 s=0.75, is both the cheapest and the most accurate B-ReKV point.

## Paper-Facing Takeaway

Pair #7 RepoBench completes the Table 6 extended-task matrix. The result supports two claims:

- ReKV transfers to cross-file code completion, with all six fixed-budget settings clustered around 0.35 edit similarity.
- B-ReKV provides a strong communication-efficiency point: 0.3400 score at roughly 15.9% transmitted KV, compared with 0.3485 at 32.5% for the lowest fixed ReKV budget.

For a compact table, report:

| Pair / task | Best ReKV | Best B-ReKV | B-ReKV actual budget |
|---|---:|---:|---:|
| Pair #7 / RepoBench | 0.3530 | 0.3400 | 0.1594 |

## Operational Note

Receiver scoring currently forms the full `[heads, query_length, KV_length]` attention tensor before applying `recv_window`. RepoBench's long code prompts therefore caused approximately 81 GB peak/reserved GPU memory per process. A future memory-safe implementation should slice receiver queries to the final observation window before the QK matrix multiplication and softmax.
