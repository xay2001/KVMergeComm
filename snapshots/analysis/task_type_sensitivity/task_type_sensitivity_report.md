# Task-Type Sensitivity

Task groups:

- simple: countries, tipsheets
- multi-hop: hotpotqa, musique, twowikimqa
- long document: qasper, multifieldqa_en
- math/reasoning: tmath

Outputs:

- `snapshots/analysis/task_type_sensitivity/task_type_run_summary.csv`
- `snapshots/analysis/task_type_sensitivity/task_type_family_summary.csv`

Use this as the starting point for the paper discussion: ReKV/B-ReKV is strongest on evidence-heavy multi-hop and long-context settings, while simple synthetic tasks can saturate or favor layer-level KVComm.
