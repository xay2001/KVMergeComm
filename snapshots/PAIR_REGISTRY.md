# KVComm / ReKV Pair Registry

This file maps KVComm paper model pairs to local snapshot roots. Use it with
`snapshots/manifest/experiments.csv` when checking which paper table cells are
complete.

## Main Dataset Set

The shared 8-task set is:

`countries`, `tipsheets`, `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `twowikimqa`, `tmath`.

## Pair Mapping

| Paper table | Pair | Snapshot root | Sender | Receiver | Current scope |
|---|---:|---|---|---|---|
| Table 8 | #1 | `snapshots/<dataset>/` | `/sharedspace/models/Llama-3.1-8B-Instruct` | `/sharedspace/models/Llama-3.1-8B-Instruct` | Full sweep: KVComm, merge, evict, ReKV, budget, coverage, progressive, features |
| Table 8 | #2 | `snapshots/table8_pair2_llama32_same/` | `/sharedspace/models/Llama-3.2-3B-Instruct` | `/sharedspace/models/Llama-3.2-3B-Instruct` | Paper-table queue: ReKV + B-ReKV |
| Table 8 | #3 | `snapshots/table8_pair3_qwen25_7b_same/` | `/sharedspace/models/Qwen2.5-7B-Instruct` | `/sharedspace/models/Qwen2.5-7B-Instruct` | Paper-table queue: ReKV + B-ReKV |
| Table 8 | #4 | `snapshots/table8_pair4_falcon3_7b_same/` | `/NAS/models/Falcon3-7B-Instruct` | `/NAS/models/Falcon3-7B-Instruct` | Paper-table queue complete: ReKV + B-ReKV |
| Table 8 | #5 | `snapshots/table8_pair5_evolcodellama_toolace/` | `/NAS/models/EvolCodeLlama-3.1-8B-Instruct` | `/NAS/models/ToolACE-2-Llama-3.1-8B` | Paper-table queue complete: ReKV + B-ReKV |
| Table 8 | #9 | `snapshots/table8_pair9_supernova_deepseek_llama8b/` | `/NAS/models/Llama-3.1-SuperNova-Lite` | `/NAS/models/DeepSeek-R1-Distill-Llama-8B` | Paper-table queue complete: ReKV + B-ReKV; near-zero non-`tmath` scores need inspection |
| Table 1 | #6 | `snapshots/table1_pair6_llama32_abliterated_deepseek3b/` | `/sharedspace/models/Llama-3.2-3B-Instruct-abliterated` | `/sharedspace/models/DeepSeek-R1-Distill-Llama-3B` | Paper-table queue: ReKV + B-ReKV |
| Table 1 | #7 | `snapshots/table1_pair7_qwen25_uncensored_bespoke/` | `/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored` | `/sharedspace/models/Bespoke-Stratos-7B` | Paper-table queue: ReKV + B-ReKV |
| Table 1 | #8 | `snapshots/table1_pair8_falcon3_ultraset_abliterated/` | `/sharedspace/models/falcon3-ultraset` | `/sharedspace/models/Falcon3-7B-Instruct-abliterated` | Paper-table queue: ReKV + B-ReKV |

## Directory Conventions

Flat dataset roots such as `snapshots/hotpotqa/` are legacy-but-active roots for
Table 8 pair #1. New paper-aligned model pairs use explicit roots:

```text
snapshots/table{paper_table}_pair{pair_id}_{slug}/<dataset>/<method>/<run>/
```

Each canonical run directory should keep:

- `log.log`: Python logging output with `AlignConfig` and final result.
- `per_sample.jsonl`: per-sample scores and budget fields when generated.
- `cost_summary.json`: aggregate cost fields when cost profiling is enabled.

## Query-Sketch Protocol Roots

Receiver-initiated query-sketch runs use a protocol-specific root rather than
sharing the canonical query-blind root:

```text
snapshots/table{paper_table}_pair{pair_id}_query_sketch_{slug}/<dataset>/<method>/<run>/
```

The existing Table 1 pair #6 root is:

```text
snapshots/table1_pair6_query_sketch_llama32_abliterated_deepseek3b/
```

Use the same naming rule for future pair #1, pair #7, and Table 8 query-sketch
runs. The manifest maps these roots back to the canonical paper pair while
preserving the protocol-specific root path. Runs should record
`protocol_version` in the `_meta` object of `per_sample.jsonl` and/or
`cost_summary.json`; the manifest copies that value into its
`protocol_version` field.

Queue-level shell logs under `logs/`, `snapshots/*.out`, or
`snapshots/table*_pair*/logs/` are operational records, not metric sources.
