# Archived Operational Logs

These files are queue-level stdout/stderr captures archived on 2026-06-30.
They are not canonical metric sources.

For metrics and reproducibility, use run-level files:

- `snapshots/**/log.log`
- `snapshots/**/per_sample.jsonl`
- `snapshots/manifest/experiments.csv`

The root logs left in `logs/` are the main queue logs still referenced by the
experiment record.
