#!/usr/bin/env python
"""Analyze the same-implementation Layer-vs-Token causal comparison.

Inputs: per_sample.jsonl runs under snapshots/layer_vs_token_v1/ (and
optionally snapshots/ctx_scaling_v1/ with --root).

Outputs (written next to the root):
  - layer_vs_token_summary.csv     per run: score, actual bytes, recovered skyline
  - layer_vs_token_pareto_auc.csv  per method x task: normalized Pareto AUC
  - layer_vs_token_paired.csv      matched-fraction paired diffs + bootstrap CI + win/loss
  - layer_vs_token_report.md       human-readable summary
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

METHOD_LABELS = {
    "skyline": "Skyline (raw text)",
    "full_kv": "Full KV",
    "kvcomm": "KVComm layer (query-free)",
    "random_layer": "Random layer",
    "recv_layer": "Receiver-aware layer",
    "evict": "Query-free token (value-norm)",
    "rekv": "ReKV (receiver token)",
    "rekv_shuffled": "ReKV shuffled query",
}

GRID_METHODS = ["kvcomm", "random_layer", "recv_layer", "evict", "rekv"]


def load_per_sample(path):
    meta, rows = None, []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                rows.append(row)
    return meta, rows


def frac_from_run_name(run_dir):
    import re

    m = re.search(r"(?:_r|_f|top)(\d?\.\d+)", run_dir.name)
    return float(m.group(1)) if m else None


def collect_runs(root):
    """-> {task: {method: [run_info...]}}"""
    import re

    # Deduplicate runs that share the same base run name (timestamp suffix
    # stripped): keep only the most recently modified per_sample.jsonl.
    latest = {}
    for per_sample in root.rglob("per_sample.jsonl"):
        run_dir = per_sample.parent
        base = re.sub(r"_\d{4}_\d{4}$", "", run_dir.name)
        key = (run_dir.parent.parent.name, run_dir.parent.name, base)
        mtime = per_sample.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            latest[key] = (mtime, per_sample)

    runs = defaultdict(lambda: defaultdict(list))
    for _, per_sample in sorted(latest.values(), key=lambda x: str(x[1])):
        run_dir = per_sample.parent
        method_dir = run_dir.parent
        task_dir = method_dir.parent
        method = method_dir.name
        task = task_dir.name
        meta, rows = load_per_sample(per_sample)
        if not rows:
            continue
        scores = {row["idx"]: float(row["score"]) for row in rows}
        bytes_per = {
            row["idx"]: float(row.get("total_communication_bytes") or 0.0)
            for row in rows
        }
        a2b_per = {
            row["idx"]: float(row.get("a_to_b_communication_bytes") or 0.0)
            for row in rows
        }
        info = {
            "run_dir": str(run_dir),
            "method": method,
            "task": task,
            "fraction": frac_from_run_name(run_dir),
            "n": len(rows),
            "score_mean": sum(scores.values()) / len(scores),
            "bytes_mean": sum(bytes_per.values()) / len(bytes_per) if bytes_per else 0.0,
            "a2b_bytes_mean": sum(a2b_per.values()) / len(a2b_per) if a2b_per else 0.0,
            "budget_mean": (
                sum(float(r.get("budget") or 0) for r in rows) / len(rows)
                if any(r.get("budget") is not None for r in rows)
                else None
            ),
            "scores": scores,
            "bytes": bytes_per,
        }
        runs[task][method].append(info)
    # skyline runs live in run dirs without byte fields; also accept
    # skyline logs (skyline uses SkylineEvaluator -> no per_sample.jsonl), so
    # parse skyline result from log.log instead.
    for log in sorted(root.rglob("skyline/*/log.log")):
        import re

        text = log.read_text(errors="ignore")
        # Receiver-side skyline (model B answering from raw text) is the
        # relevant reference for KV communication to B.
        m = re.findall(r"skyline result B: ([0-9.]+)", text)
        if not m:
            continue
        task = log.parent.parent.parent.name
        runs[task]["skyline"].append(
            {
                "run_dir": str(log.parent),
                "method": "skyline",
                "task": task,
                "fraction": None,
                "n": None,
                "score_mean": float(m[-1]),
                "bytes_mean": None,
                "a2b_bytes_mean": None,
                "budget_mean": None,
                "scores": {},
                "bytes": {},
            }
        )
    return runs


def pareto_auc(points, byte_max):
    """Normalized accuracy-bytes AUC: x = bytes / byte_max clipped to [0,1]."""
    pts = sorted((min(b / byte_max, 1.0), s) for b, s in points)
    if len(pts) < 2:
        return None
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    span = pts[-1][0] - pts[0][0]
    return auc / span if span > 0 else None


def bootstrap_ci(diffs, n_boot=2000, seed=42):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="snapshots/layer_vs_token_v1")
    args = parser.parse_args()
    root = Path(args.root)

    runs = collect_runs(root)
    summary_rows = []
    auc_rows = []
    paired_rows = []
    report = ["# Layer-vs-Token same-implementation comparison\n"]

    for task in sorted(runs):
        task_runs = runs[task]
        skyline = max((r["score_mean"] for r in task_runs.get("skyline", [])), default=None)
        full_kv = task_runs.get("full_kv", [])
        full_bytes = full_kv[0]["bytes_mean"] if full_kv else None

        report.append(f"\n## {task}\n")
        report.append(f"- skyline = {skyline}, full_kv bytes = {full_bytes and round(full_bytes/1e6,2)} MB\n")
        report.append("| method | frac | n | score | recovered skyline | total bytes (MB) | A->B bytes (MB) | budget |")
        report.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for method in ["full_kv"] + GRID_METHODS + ["rekv_shuffled"]:
            for info in sorted(task_runs.get(method, []), key=lambda r: (r["fraction"] or 0)):
                recovered = (
                    info["score_mean"] / skyline if skyline else None
                )
                summary_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "fraction": info["fraction"],
                        "n": info["n"],
                        "score": round(info["score_mean"], 6),
                        "skyline": skyline,
                        "recovered_skyline": recovered and round(recovered, 4),
                        "bytes_mean": round(info["bytes_mean"], 1),
                        "bytes_mb": round(info["bytes_mean"] / 1e6, 4),
                        "a2b_mb": round((info.get("a2b_bytes_mean") or 0.0) / 1e6, 4),
                        "budget_mean": info["budget_mean"],
                        "run_dir": info["run_dir"],
                    }
                )
                report.append(
                    f"| {METHOD_LABELS.get(method, method)} | {info['fraction'] or ''} | {info['n']} "
                    f"| {info['score_mean']:.4f} | {recovered and format(recovered, '.1%') or ''} "
                    f"| {info['bytes_mean']/1e6:.2f} | {(info.get('a2b_bytes_mean') or 0)/1e6:.2f} "
                    f"| {info['budget_mean'] and round(info['budget_mean'],3) or ''} |"
                )

        # Pareto AUC over the shared normalized byte axis
        if full_bytes:
            for method in GRID_METHODS:
                pts = [
                    (r["bytes_mean"], r["score_mean"])
                    for r in task_runs.get(method, [])
                    if r["bytes_mean"]
                ]
                auc = pareto_auc(pts, full_bytes)
                if auc is not None:
                    auc_rows.append(
                        {"task": task, "method": method, "pareto_auc": round(auc, 4), "points": len(pts)}
                    )

        # Matched-fraction paired comparisons (same samples, same nominal fraction)
        pairs = [
            ("rekv", "recv_layer"),
            ("rekv", "kvcomm"),
            ("rekv", "evict"),
            ("recv_layer", "kvcomm"),
            ("recv_layer", "random_layer"),
            ("kvcomm", "random_layer"),
        ]
        by_frac = lambda m: {r["fraction"]: r for r in task_runs.get(m, []) if r["fraction"]}
        for m1, m2 in pairs:
            r1s, r2s = by_frac(m1), by_frac(m2)
            for frac in sorted(set(r1s) & set(r2s)):
                a, b = r1s[frac], r2s[frac]
                shared = sorted(set(a["scores"]) & set(b["scores"]))
                if len(shared) < 10:
                    continue
                diffs = [a["scores"][i] - b["scores"][i] for i in shared]
                lo, hi = bootstrap_ci(diffs)
                wins = sum(1 for d in diffs if d > 1e-9)
                losses = sum(1 for d in diffs if d < -1e-9)
                paired_rows.append(
                    {
                        "task": task,
                        "method_a": m1,
                        "method_b": m2,
                        "fraction": frac,
                        "n": len(shared),
                        "mean_diff": round(sum(diffs) / len(diffs), 4),
                        "ci_low": round(lo, 4),
                        "ci_high": round(hi, 4),
                        "wins": wins,
                        "losses": losses,
                        "ties": len(shared) - wins - losses,
                        "bytes_a_mb": round(a["bytes_mean"] / 1e6, 3),
                        "bytes_b_mb": round(b["bytes_mean"] / 1e6, 3),
                    }
                )

        # shuffled-query causal control at matched fraction
        rekv_by_frac = by_frac("rekv")
        for r in task_runs.get("rekv_shuffled", []):
            frac = r["fraction"]
            if frac in rekv_by_frac:
                a, b = rekv_by_frac[frac], r
                shared = sorted(set(a["scores"]) & set(b["scores"]))
                if len(shared) < 10:
                    continue
                diffs = [a["scores"][i] - b["scores"][i] for i in shared]
                lo, hi = bootstrap_ci(diffs)
                paired_rows.append(
                    {
                        "task": task,
                        "method_a": "rekv",
                        "method_b": "rekv_shuffled",
                        "fraction": frac,
                        "n": len(shared),
                        "mean_diff": round(sum(diffs) / len(diffs), 4),
                        "ci_low": round(lo, 4),
                        "ci_high": round(hi, 4),
                        "wins": sum(1 for d in diffs if d > 1e-9),
                        "losses": sum(1 for d in diffs if d < -1e-9),
                        "ties": sum(1 for d in diffs if abs(d) <= 1e-9),
                        "bytes_a_mb": round(a["bytes_mean"] / 1e6, 3),
                        "bytes_b_mb": round(b["bytes_mean"] / 1e6, 3),
                    }
                )

    import csv

    def write_csv(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    out = root
    write_csv(out / "layer_vs_token_summary.csv", summary_rows)
    write_csv(out / "layer_vs_token_pareto_auc.csv", auc_rows)
    write_csv(out / "layer_vs_token_paired.csv", paired_rows)

    if auc_rows:
        report.append("\n## Pareto AUC (normalized bytes)\n")
        report.append("| task | " + " | ".join(GRID_METHODS) + " |")
        report.append("|---|" + "---:|" * len(GRID_METHODS))
        by_task = defaultdict(dict)
        for row in auc_rows:
            by_task[row["task"]][row["method"]] = row["pareto_auc"]
        for task in sorted(by_task):
            report.append(
                f"| {task} | "
                + " | ".join(str(by_task[task].get(m, "")) for m in GRID_METHODS)
                + " |"
            )

    if paired_rows:
        report.append("\n## Paired differences (bootstrap 95% CI)\n")
        report.append("| task | A vs B | frac | mean diff | 95% CI | win/loss/tie | bytes A/B (MB) |")
        report.append("|---|---|---:|---:|---|---|---|")
        for row in paired_rows:
            report.append(
                f"| {row['task']} | {row['method_a']} vs {row['method_b']} | {row['fraction']} "
                f"| {row['mean_diff']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                f"| {row['wins']}/{row['losses']}/{row['ties']} "
                f"| {row['bytes_a_mb']}/{row['bytes_b_mb']} |"
            )

    report_path = out / "layer_vs_token_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
