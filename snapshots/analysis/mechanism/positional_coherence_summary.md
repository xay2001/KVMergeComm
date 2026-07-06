| Task | ReKV normal | ReKV-S | B-ReKV normal | B-ReKV-S |
|---|---:|---:|---:|---|
| countries | 0.6000 | 0.6000 | 0.6150 | skipped |
| tipsheets | 0.8680 | 0.8620 | 0.8780 | skipped |
| hotpotqa | 0.6960 | 0.6120 | 0.7080 | skipped |
| qasper | 0.3440 | 0.2900 | 0.3320 | skipped |
| musique | 0.4800 | 0.3440 | 0.4820 | skipped |
| multifieldqa_en | 0.5067 | 0.4267 | 0.5200 | skipped |
| twowikimqa | 0.4050 | 0.2600 | 0.4100 | skipped |
| tmath | 0.3408 | 0.3494 | 0.3481 | skipped |

B-ReKV-S is skipped by default because `shift_back + coverage budget` triggers the dynamic-cache length assertion documented in `brekv_shiftback_diagnosis.md`.
