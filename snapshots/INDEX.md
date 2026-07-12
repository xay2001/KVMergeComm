# Snapshots Index

This is the working index for KVComm / ReKV experiment assets. It points to the
large narrative document, the pair registry, and the machine-readable run
manifest.

## Primary Files

- `snapshots/RESULTS.md`: narrative experiment record and historical notes.
- `snapshots/EXPERIMENT_TODO.md`: full paper/appendix/reviewer-risk experiment checklist.
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
| Table 8 | #1 | `snapshots/<dataset>/` | 511 | Main full sweep; paper-table ReKV/B-ReKV block is complete |
| Table 8 | #2 | `snapshots/table8_pair2_llama32_same/` | 74 | 72 completed paper-table runs plus 2 incomplete duplicate coverage dirs |
| Table 8 | #3 | `snapshots/table8_pair3_qwen25_7b_same/` | 72 | Complete for 8 datasets x 9 paper-table runs |
| Table 8 | #4 | `snapshots/table8_pair4_falcon3_7b_same/` | 72 | Complete for 8 datasets x 9 paper-table runs |
| Table 8 | #5 | `snapshots/table8_pair5_evolcodellama_toolace/` | 72 | Complete for 8 datasets x 9 paper-table runs |
| Table 8 | #9 | `snapshots/table8_pair9_supernova_deepseek_llama8b/` | 72 | Complete but deferred from positive comparison; KVComm probe is also poor on QA/multi-hop tasks |
| Table 10 | multi-source | `snapshots/table10_multi_source_rekv/` | 18 | Complete for hotpotqa / musique / twowikimqa x ReKV-w8/w16 x r=0.3/0.5/0.7 |

## Paper-Table Run Definition

For paper-aligned Table 1 / Table 8 queues, one complete dataset block contains
9 runs:

- `ReKV-w8`: `r=0.3`, `r=0.5`, `r=0.7`
- `ReKV-w16`: `r=0.3`, `r=0.5`, `r=0.7`
- `B-ReKV`: `cov_t0.95_s0.75_w8`
- `B-ReKV`: `cov_t0.95_s0.85_w8`
- `B-ReKV`: `cov_t0.95_s0.90_w16`

## Table 1: Main KVComm Model Pairs

| Pair | Dataset coverage | Notes |
|---:|---|---|
| #6 | `countries`, `tipsheets`, `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `twowikimqa`, `tmath`: 9/9 each | Complete paper-table queue |
| #7 | `countries`, `tipsheets`, `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `twowikimqa`, `tmath`: 9/9 each | Complete paper-table queue |
| #8 | `countries`: 1/9; `multifieldqa_en`: 1/9 | Falcon pair currently has incomplete/failed startup runs |

## Table 8: Appendix Model Pairs

| Pair | Dataset coverage | Notes |
|---:|---|---|
| #1 | Full legacy sweep under `snapshots/<dataset>/` | Includes KVComm, merge, evict, ReKV, budget, coverage, progressive, features; paper-table ReKV/B-ReKV block complete |
| #2 | 8 datasets have 9 completed paper-table runs | Two extra incomplete coverage directories are indexed as `unknown` |
| #3 | 8 datasets have 9 completed paper-table runs | Complete paper-table queue |
| #4 | 8 datasets have 9 completed paper-table runs | Falcon3 same-model queue complete |
| #5 | 8 datasets have 9 completed paper-table runs | EvolCodeLlama -> ToolACE queue complete |
| #9 | 8 datasets have 9 completed paper-table runs | SuperNova -> DeepSeek-Llama-8B queue complete, but deferred from Table 8 positive evidence because KVComm baseline probe is also unstable |

## Method Directory Legend

| Directory | Method family | Notes |
|---|---|---|
| `kvcomm/` | KVComm layer selection | Mostly legacy pair #1 |
| `mtc_merge/` | token-level merge | Query-agnostic merge baseline |
| `mtc_evict/` | token-level evict | Query-agnostic value-norm baseline |
| `mtc_receiver/` | ReKV | Receiver-aware token selection; includes probe runs |
| `budget/` | budget negative results | `uniform`, `layer`, `query`, `query+layer` |
| `coverage/` | B-ReKV | receiver-attention evidence coverage budget |
| `progressive/` | online progressive negative result | per-sample progressive traces |
| `features/` | Pass-1 feature dump | budget predictor / LODO studies |
| `logs/` | queue logs | Operational shell logs, not metric sources |

## Known Cleanup / Follow-Up Items

- Archive root `logs/cov_*.log`, `logs/feat_*.log`, and `snapshots/**/*.out` after preserving manifest references.
- Keep all run-level `log.log` and `per_sample.jsonl` files.
- Pair #7 needs `tmath` completion.
- Pair #8 needs model path/startup issue resolution before continuing the queue.
- Pair #1 `qasper` remains sparse relative to other datasets.
- Pair #9 is now deferred from positive comparison. Score-distribution, raw-output, and KVComm probe diagnostics indicate a hard heterogeneous-pair issue rather than a ReKV-specific failure; see `snapshots/analysis/pair9/pair9_diagnostic_report.md`.

## Reviewer-Risk Experiment Status

These experiments were added to support the paper framing beyond raw accuracy.

| Module | Output | Status |
|---|---|---|
| Cost / efficiency | `snapshots/cost_profile/pair1_llama31_same_all8_full/cost_table.csv`, `snapshots/cost_profile/table1_pair6_llama32_abliterated_deepseek3b_full/cost_table.csv`, `snapshots/cost_profile/table1_pair7_qwen25_uncensored_bespoke_full/cost_table.csv`, `snapshots/analysis/cost/pair6_pair7_cost_focus_hotpotqa_musique_multifieldqa.csv` | Done for pair #1/#6/#7; pair #6/#7 paper-focused cost subset generated |
| Coverage robustness / Pareto | `snapshots/coverage_robustness_summary.txt`, `snapshots/{musique,hotpotqa,multifieldqa_en}/coverage_pareto.png`, `snapshots/analysis/robustness/pair6_pair7_brekv_robustness_summary.csv`, `snapshots/analysis/robustness/pair{6,7}_{hotpotqa,musique}_brekv_pareto.png` | Done; pair #6/#7 summary and Pareto plots generated |
| B-ReKV budget adaptivity | `snapshots/brekv_budget_distribution.png`, `snapshots/brekv_budget_distribution_summary.csv` | Done |
| Receiver-initiated fairness | `snapshots/query_fairness/pair1_llama31_same/query_fairness.csv` | Done |
| Interpretability / evidence proxy | `snapshots/interpretability/pair1_llama31_same/interpretability_overlap_summary.csv`, `interpretability_examples.md`, `cleaned/*_clean_top_tokens.png`, `snapshots/deletion_ablation/pair1_llama31_same/deletion_ablation_summary_w8_r0.3_k20.csv`, `snapshots/supporting_overlap/hotpotqa_pair1_full_context/supporting_overlap_summary_top20_w8_r0.3.csv`, `snapshots/analysis/figures/supporting_overlap_bar.png` | Done; supporting-facts overlap added |
| Failure / task sensitivity analysis | `snapshots/analysis/failure_cases/`, `snapshots/analysis/task_type_sensitivity/`, `snapshots/analysis/figures/failure_rate_heatmap.png`, `snapshots/analysis/figures/task_type_sensitivity_bar.png` | Done from existing per-sample outputs |
| Sink/recent token ablation | `snapshots/mechanism/pair1_llama31_same/sink_recent/` | Done for 8 main tasks |
| Positional coherence / ReKV-S | `snapshots/mechanism/pair1_llama31_same/positional_coherence/`, `snapshots/analysis/mechanism/positional_coherence_summary.md`, `snapshots/analysis/mechanism/brekv_shiftback_diagnosis.md` | Done for 8 main tasks for ReKV normal / ReKV-S / B-ReKV normal; B-ReKV-S omitted due to shift-back implementation limit |
| Table 10 Multi-Source ReKV | `snapshots/table10_multi_source_rekv/`, `snapshots/analysis/table10_multi_source_rekv_summary.md` | Done for hotpotqa / musique / twowikimqa; 18 runs complete on GPU3 |
| Table 6 extended tasks | `snapshots/table6_pair6_llama32_abliterated_deepseek3b/`, `snapshots/table6_pair7_qwen25_uncensored_bespoke/`, `snapshots/analysis/latest_experiments/table6_extended_status.csv`, `snapshots/analysis/latest_experiments/table6_pair7_repobench_summary.md` | Pair #6 and pair #7 are complete for all 5 extended tasks x 9 runs; pair #7 RepoBench was recovered on GPUs 0–3 after the earlier GPU7 OOM |
| Score-function ablation | `snapshots/score_function_ablation/`, `snapshots/analysis/latest_experiments/score_function_summary.csv`, `snapshots/analysis/latest_experiments/figures/score_function_ablation_best.png` | Done for pair #1/#6/#7 on HotpotQA / MuSiQue / MultiFieldQA-en; receiver-aware scoring remains strongest overall |
| Receiver-layer aggregation ablation | `snapshots/layer_aggregation_ablation/`, `snapshots/analysis/latest_experiments/layer_aggregation_summary.csv`, `snapshots/analysis/latest_experiments/figures/layer_aggregation_heatmap.png` | Done for pair #1 on 8 main tasks; original paired-layer identity is strongest or near-strongest on most evidence-heavy tasks |
| NLD vs ReKV cost comparison | `snapshots/nld_cost_profile/`, `snapshots/analysis/nld_vs_rekv/nld_vs_rekv_report.md`, `snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_focused.csv`, `snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_cost_overview.png`, `snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_accuracy_by_task.png` | Done for pair #1/#6/#7 on HotpotQA / MuSiQue / MultiFieldQA-en; NLD has much lower accuracy and roughly 2x latency, with similar peak memory |

Main remaining work is now narrower: fold the completed Table 6 and NLD
comparisons into the paper, then implement memory-safe pre-softmax receiver
windowing before any further long-code receiver-scoring runs.
Pair #9 and B-ReKV-S are deferred from positive claims.
