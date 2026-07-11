# Table 10 Multi-Source ReKV Summary

## Run Status

- Status: complete.
- Queue log: `snapshots/table10_multi_source_rekv/logs/gpu3_table10_multi_source_rekv_0710_1250.log`.
- Script: `scripts/run_table10_multi_source_rekv_gpu3.sh`.
- Model setup: A1 = A2 = B = `/NAS/models/Llama-3.1-8B-Instruct`.
- Method: two sender contexts transmit KV to one receiver; token selection uses receiver attention (`score_mode=receiver`, `merge_mode=evict`).
- Grid: 3 tasks x 2 receiver windows x 3 keep ratios = 18 completed runs.
- Outputs: all 18 runs have `per_sample.jsonl`.

No `Traceback`, CUDA OOM, or error line was found in the queue log. The queue ended with:

```text
######## Table10 Multi-Source ReKV GPU3 DONE 2026-07-10 13:40:25 ########
```

## Result Matrix

Metric is each evaluator's `communication result` from run-level `log.log`.

| Task | N | ReKV-w8 r=0.3 | ReKV-w8 r=0.5 | ReKV-w8 r=0.7 | ReKV-w16 r=0.3 | ReKV-w16 r=0.5 | ReKV-w16 r=0.7 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hotpotqa | 500 | 0.6600 | 0.6720 | 0.6740 | 0.6600 | **0.6800** | 0.6740 | w16 r=0.5 |
| musique | 500 | 0.4300 | 0.4520 | **0.4620** | 0.4400 | 0.4500 | 0.4560 | w8 r=0.7 |
| twowikimqa | 200 | 0.4250 | **0.4300** | **0.4300** | 0.4150 | **0.4300** | 0.4200 | tie: w8 r=0.5 / w8 r=0.7 / w16 r=0.5 |

## Aggregates

| Task | Best score | w8 average | w16 average | Best window |
|---|---:|---:|---:|---|
| hotpotqa | 0.6800 | 0.6687 | 0.6713 | w16, slight |
| musique | 0.4620 | 0.4480 | 0.4487 | w8 best point, averages tied |
| twowikimqa | 0.4300 | 0.4283 | 0.4217 | w8 |

Mean of per-task best scores: 0.5240.

## Observations

1. The multi-source path is operational end-to-end: the queue completed all planned runs, and every run produced per-sample output.
2. Increasing ratio generally helps on `musique`, especially under `w8`: 0.4300 -> 0.4520 -> 0.4620. This suggests the split-support multi-source setting benefits from retaining more sender KV on the hardest task.
3. `hotpotqa` peaks at the middle budget (`w16 r=0.5` = 0.6800). Higher budget does not improve beyond that, so `r=0.5` is a reasonable default for this setup.
4. `twowikimqa` is flat around 0.4300; larger window does not help and can hurt at `r=0.3` / `r=0.7`.
5. `w16` is not consistently better than `w8` in multi-source ReKV. For a compact Table 10, report both windows or use `w8` as the simpler default plus the `hotpotqa w16 r=0.5` best point.

## Recommended Table 10 Slice

If the paper table needs a small, readable block, use the best-per-task row:

| Task | Best Multi-Source ReKV | Config |
|---|---:|---|
| hotpotqa | 0.6800 | w16 r=0.5 |
| musique | 0.4620 | w8 r=0.7 |
| twowikimqa | 0.4300 | w8 r=0.5 / w8 r=0.7 / w16 r=0.5 |

If a fixed configuration is preferred across tasks, `w8 r=0.7` is simple and strong:

| Task | Multi-Source ReKV w8 r=0.7 |
|---|---:|
| hotpotqa | 0.6740 |
| musique | 0.4620 |
| twowikimqa | 0.4300 |

## Notes

- This table should not be directly compared one-to-one with the single-source Table 8 runs: the multi-agent dataloader splits evidence into `prompt_A1` and `prompt_A2`, changing the input construction and communication pattern.
- The current implementation covers two senders (`num_senders=2`). Extending to arbitrary N senders would require generalizing `models_ms.py`, `eval_ms.py`, and dataloader prompt fields beyond `prompt_A1` / `prompt_A2`.
