# Scripts Guide

This directory contains three kinds of scripts:

1. Canonical entry points for future runs.
2. Analysis / plotting scripts that read existing outputs.
3. Archived one-off queue wrappers kept for provenance.

Prefer the canonical entry points below when launching new experiments.

## Canonical Run Scripts

### Single-Method Building Blocks

These operate on one dataset and write under `snapshots/<TASK>/...` unless a
caller passes another `--snapshot_path` through a wrapper.

| Script | Purpose | Main env vars |
|---|---|---|
| `run_baseline.sh` | KVComm layer-selection baseline | `TASK`, `GPU` |
| `run_merge.sh` | token-level merge baseline | `TASK`, `GPU` |
| `run_evict.sh` | value-norm evict baseline | `TASK`, `GPU` |
| `run_receiver.sh` | fixed-r RASC sweep | `TASK`, `GPU`, `WIN` |
| `run_dataset.sh` | full legacy pair #1 method suite | `TASK`, `GPU` |

### Paper Table Queues

Use this for new Table 1 / Table 8 model-pair runs:

```bash
TABLE_ID=1 PAIR_ID=7 \
MODEL_A=/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored \
MODEL_B=/sharedspace/models/Bespoke-Stratos-7B \
ROOT=snapshots/table1_pair7_qwen25_uncensored_bespoke \
GPU_LIST="5 6" \
bash scripts/run_paper_table_queue.sh
```

The paper-table queue runs 9 points per dataset:

- RASC `w8/w16 x r={0.3,0.5,0.7}`
- Coverage-BRASC `t0.95_s0.75_w8`
- Coverage-BRASC `t0.95_s0.85_w8`
- Coverage-BRASC `t0.95_s0.90_w16`

Older pair-specific scripts have been archived under
`scripts/archive/deprecated_20260630/` after the unified runner was added.

### Coverage / Budget-Aware Runs

| Script | Purpose |
|---|---|
| `run_probe.sh` | dense fixed-r RASC probes for oracle / coverage comparisons |
| `run_budget.sh` | Step 1 open-loop budget modes |
| `run_budget_all.sh` | legacy 7-dataset budget queue |
| `run_progressive.sh` | Step 2b online progressive runs |
| `run_features.sh` | Pass-1 features for budget predictor |
| `run_coverage.sh` | canonical Coverage-BRASC single-task runner |
| `run_cost_profile.sh` | controlled timing / payload runs for the paper cost table |

Coverage wrapper scripts such as `run_coverage_stage1.sh` are retained only as
historical profiles; new coverage sweeps should call `run_coverage.sh` or the
paper-table queue directly.

## Analysis and Plotting

| Script | Reads | Output |
|---|---|---|
| `build_experiment_manifest.py` | `snapshots/**/log.log`, `per_sample.jsonl` | `snapshots/manifest/experiments.{csv,json}` |
| `analyze_oracle.py` | probe `per_sample.jsonl` | oracle headroom stdout |
| `analyze_budget.py` | `budget/` runs | budget negative-result stdout |
| `sim_progressive.py` | probe scores | offline progressive upper bound |
| `analyze_progressive_online.py` | progressive traces | threshold sweep stdout |
| `learn_stop_policy.py` | progressive traces | learned stop-policy report |
| `learn_budget_predictor.py` | features + probe labels | WITHIN/LODO predictor report |
| `sim_coverage_budget.py` | features + probe labels | offline coverage pre-check |
| `analyze_coverage.py` | fixed probes + coverage runs | coverage Pareto stdout |
| `plot_coverage_pareto.py` | fixed probes + coverage runs | Pareto figure |
| `plot_budget_distribution.py` | one `per_sample.jsonl` | budget distribution figure |
| `analyze_cost_profile.py` | `cost_summary.json` files | Markdown/CSV cost table |

## Data Preparation

- `download_datasets.sh`: main dataset download entry.
- `prepare_qasper.py`: QASPER-specific helper retained for reproducibility.

## Rules for New Scripts

- Do not copy an existing table-pair script just to change model paths. Use
  `run_paper_table_queue.sh`.
- Write paper-aligned runs to `snapshots/table{N}_pair{M}_{slug}/`.
- Keep run-level `log.log` and `per_sample.jsonl`.
- Treat queue-level `logs/*.log` and `*.out` as operational logs; archive them
  after the manifest has been regenerated.
- Regenerate the manifest after every batch:

```bash
python scripts/build_experiment_manifest.py
```
