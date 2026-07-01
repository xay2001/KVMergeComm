# Snapshots Index

This is the working index for KVComm / RASC experiment assets. It points to the
large narrative document, the pair registry, and the machine-readable run
manifest.

## Primary Files

- `snapshots/RESULTS.md`: narrative experiment record and historical notes.
- `snapshots/PAIR_REGISTRY.md`: paper table / model pair / local path mapping.
- `snapshots/manifest/experiments.csv`: machine-readable run manifest.
- `snapshots/manifest/experiments.json`: JSON version of the same manifest.

Regenerate the manifest after new runs:

```bash
python scripts/build_experiment_manifest.py
```

## Current Manifest Summary

Generated from `snapshots/**/log.log`.

| Paper table | Pair | Snapshot root | Runs indexed | Status |
|---|---:|---|---:|---|
| Table 1 | #6 | `snapshots/table1_pair6_llama32_abliterated_deepseek3b/` | 72 | Complete for 8 datasets x 9 paper-table runs |
| Table 1 | #7 | `snapshots/table1_pair7_qwen25_uncensored_bespoke/` | 64 | Missing most `tmath` paper-table runs |
| Table 1 | #8 | `snapshots/table1_pair8_falcon3_ultraset_abliterated/` | 2 | Started only; current runs are not completed |
| Table 8 | #1 | `snapshots/<dataset>/` | 506 | Main full sweep; `qasper` is sparse |
| Table 8 | #2 | `snapshots/table8_pair2_llama32_same/` | 74 | 72 completed paper-table runs plus 2 incomplete duplicate coverage dirs |
| Table 8 | #3 | `snapshots/table8_pair3_qwen25_7b_same/` | 72 | Complete for 8 datasets x 9 paper-table runs |

## Paper-Table Run Definition

For paper-aligned Table 1 / Table 8 queues, one complete dataset block contains
9 runs:

- `RASC-w8`: `r=0.3`, `r=0.5`, `r=0.7`
- `RASC-w16`: `r=0.3`, `r=0.5`, `r=0.7`
- `Coverage-BRASC`: `cov_t0.95_s0.75_w8`
- `Coverage-BRASC`: `cov_t0.95_s0.85_w8`
- `Coverage-BRASC`: `cov_t0.95_s0.90_w16`

## Table 1: Main KVComm Model Pairs

| Pair | Dataset coverage | Notes |
|---:|---|---|
| #6 | `countries`, `tipsheets`, `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `twowikimqa`, `tmath`: 9/9 each | Complete paper-table queue |
| #7 | Seven datasets: 9/9 each; `tmath`: 1/9 | Finish `tmath` before treating pair #7 as complete |
| #8 | `countries`: 1/9; `multifieldqa_en`: 1/9 | Falcon pair currently has incomplete/failed startup runs |

## Table 8: Appendix Model Pairs

| Pair | Dataset coverage | Notes |
|---:|---|---|
| #1 | Full legacy sweep under `snapshots/<dataset>/` | Includes KVComm, merge, evict, RASC, budget, coverage, progressive, features |
| #2 | 8 datasets have 9 completed paper-table runs | Two extra incomplete coverage directories are indexed as `unknown` |
| #3 | 8 datasets have 9 completed paper-table runs | Complete paper-table queue |

## Method Directory Legend

| Directory | Method family | Notes |
|---|---|---|
| `kvcomm/` | KVComm layer selection | Mostly legacy pair #1 |
| `mtc_merge/` | token-level merge | Query-agnostic merge baseline |
| `mtc_evict/` | token-level evict | Query-agnostic value-norm baseline |
| `mtc_receiver/` | RASC | Receiver-aware token selection; includes probe runs |
| `budget/` | budget negative results | `uniform`, `layer`, `query`, `query+layer` |
| `coverage/` | Coverage-BRASC | receiver-attention evidence coverage budget |
| `progressive/` | online progressive negative result | per-sample progressive traces |
| `features/` | Pass-1 feature dump | budget predictor / LODO studies |
| `logs/` | queue logs | Operational shell logs, not metric sources |

## Known Cleanup / Follow-Up Items

- Archive root `logs/cov_*.log`, `logs/feat_*.log`, and `snapshots/**/*.out` after preserving manifest references.
- Keep all run-level `log.log` and `per_sample.jsonl` files.
- Pair #7 needs `tmath` completion.
- Pair #8 needs model path/startup issue resolution before continuing the queue.
- Pair #1 `qasper` remains sparse relative to other datasets.
