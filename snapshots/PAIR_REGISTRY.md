# KVComm / RASC Pair Registry

This file maps KVComm paper model pairs to local snapshot roots. Use it with
`snapshots/manifest/experiments.csv` when checking which paper table cells are
complete.

## Main Dataset Set

The shared 8-task set is:

`countries`, `tipsheets`, `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `twowikimqa`, `tmath`.

## Pair Mapping

| Paper table | Pair | Snapshot root | Sender | Receiver | Current scope |
|---|---:|---|---|---|---|
| Table 8 | #1 | `snapshots/<dataset>/` | `/sharedspace/models/Llama-3.1-8B-Instruct` | `/sharedspace/models/Llama-3.1-8B-Instruct` | Full sweep: KVComm, merge, evict, RASC, budget, coverage, progressive, features |
| Table 8 | #2 | `snapshots/table8_pair2_llama32_same/` | `/sharedspace/models/Llama-3.2-3B-Instruct` | `/sharedspace/models/Llama-3.2-3B-Instruct` | Paper-table queue: RASC + Coverage-BRASC |
| Table 8 | #3 | `snapshots/table8_pair3_qwen25_7b_same/` | `/sharedspace/models/Qwen2.5-7B-Instruct` | `/sharedspace/models/Qwen2.5-7B-Instruct` | Paper-table queue: RASC + Coverage-BRASC |
| Table 1 | #6 | `snapshots/table1_pair6_llama32_abliterated_deepseek3b/` | `/sharedspace/models/Llama-3.2-3B-Instruct-abliterated` | `/sharedspace/models/DeepSeek-R1-Distill-Llama-3B` | Paper-table queue: RASC + Coverage-BRASC |
| Table 1 | #7 | `snapshots/table1_pair7_qwen25_uncensored_bespoke/` | `/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored` | `/sharedspace/models/Bespoke-Stratos-7B` | Paper-table queue: RASC + Coverage-BRASC |
| Table 1 | #8 | `snapshots/table1_pair8_falcon3_ultraset_abliterated/` | `/sharedspace/models/falcon3-ultraset` | `/sharedspace/models/Falcon3-7B-Instruct-abliterated` | Paper-table queue: RASC + Coverage-BRASC |

## Directory Conventions

Flat dataset roots such as `snapshots/hotpotqa/` are legacy-but-active roots for
Table 8 pair #1. New paper-aligned model pairs use explicit roots:

```text
snapshots/table{paper_table}_pair{pair_id}_{slug}/<dataset>/<method>/<run>/
```

Each canonical run directory should keep:

- `log.log`: Python logging output with `AlignConfig` and final result.
- `per_sample.jsonl`: per-sample scores and budget fields when generated.

Queue-level shell logs under `logs/`, `snapshots/*.out`, or
`snapshots/table*_pair*/logs/` are operational records, not metric sources.
