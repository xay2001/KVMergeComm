#!/usr/bin/env python3
"""Analyze B-ReKV runs against fixed-r ReKV probes.

Inputs:
  fixed curve: snapshots/<task>/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl
  coverage:    snapshots/<task>/coverage/cov_t*_s*_w*/per_sample.jsonl

Usage:
  python scripts/analyze_coverage.py --tasks musique hotpotqa twowikimqa --tau 0.5
"""

import argparse
import glob
import json
import os
import re


PROBE_RE = re.compile(r"(?:probe_)?recv_w16_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")


def read_rows(path):
    rows = {}
    meta = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            rows[int(row["idx"])] = row
    return meta, rows


def load_fixed(task):
    out = {}
    patterns = [
        f"snapshots/{task}/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl",
        f"snapshots/{task}/mtc_receiver/*recv_w16_r*/per_sample.jsonl",
    ]
    for pattern in patterns:
      for path in glob.glob(pattern):
        m = PROBE_RE.search(os.path.dirname(path))
        if not m:
            continue
        r = float(m.group(1))
        if r in out:
            continue
        _, rows = read_rows(path)
        out[r] = {idx: float(row["score"]) for idx, row in rows.items()}
    return out


def load_coverage(task):
    out = []
    latest = {}
    for path in glob.glob(f"snapshots/{task}/coverage/**/per_sample.jsonl", recursive=True):
        base = os.path.basename(os.path.dirname(path))
        m = COV_RE.search(base)
        if not m:
            continue
        key = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
        if key not in latest or os.path.getmtime(path) > os.path.getmtime(latest[key]):
            latest[key] = path

    for (cov_tau, scale, win), path in sorted(latest.items()):
        base = os.path.basename(os.path.dirname(path))
        meta, rows = read_rows(path)
        scores = [float(row["score"]) for row in rows.values()]
        budgets = [float(row.get("budget", meta.get("merge_ratio", 0.0))) for row in rows.values()]
        out.append({
            "name": base,
            "cov_tau": cov_tau,
            "scale": scale,
            "win": win,
            "n": len(rows),
            "acc": sum(1 for s in scores if s >= ARGS.tau) / max(len(scores), 1),
            "score_mean": sum(scores) / max(len(scores), 1),
            "avg_budget": sum(budgets) / max(len(budgets), 1),
        })
    return out


def fixed_curve(fixed, tau):
    common = None
    for scores in fixed.values():
        ks = set(scores)
        common = ks if common is None else common & ks
    common = sorted(common or [])
    curve = []
    for r in sorted(fixed):
        acc = sum(1 for idx in common if fixed[r][idx] >= tau) / max(len(common), 1)
        score_mean = sum(fixed[r][idx] for idx in common) / max(len(common), 1)
        curve.append((r, acc, score_mean))
    return curve


def budget_at_acc(curve, acc):
    pts = sorted((r, a) for r, a, _ in curve)
    pts = sorted(pts, key=lambda x: x[1])
    if not pts:
        return None
    if acc <= pts[0][1]:
        return pts[0][0]
    if acc > pts[-1][1] + 1e-12:
        return None
    for (b0, a0), (b1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1:
            if abs(a1 - a0) < 1e-12:
                return min(b0, b1)
            return b0 + (acc - a0) / (a1 - a0) * (b1 - b0)
    return pts[-1][0]


def pareto(rows):
    front = []
    for row in sorted(rows, key=lambda r: (r["avg_budget"], -r["acc"])):
        if not front or row["acc"] > front[-1]["acc"] + 1e-12:
            front.append(row)
    return front


def report(task):
    fixed = load_fixed(task)
    cov = load_coverage(task)
    print(f"\n=== {task} ===")
    if not fixed:
        print("  [skip] no fixed probe scores")
        return
    curve = fixed_curve(fixed, ARGS.tau)
    print("fixed-r: " + ", ".join(f"r{r:g}:acc{a:.3f}" for r, a, _ in curve))
    if not cov:
        print("  [skip] no coverage runs")
        return

    for row in cov:
        b_eq = budget_at_acc(curve, row["acc"])
        row["b_eq"] = b_eq
        row["saving"] = None if b_eq is None or b_eq <= 0 else (1 - row["avg_budget"] / b_eq) * 100

    print("\ncoverage runs:")
    for row in sorted(cov, key=lambda r: (r["cov_tau"], r["scale"], r["win"])):
        cmp = "acc>fixed-ceiling" if row["b_eq"] is None else f"{row['saving']:+.1f}% vs fixed b{row['b_eq']:.3f}"
        print(
            f"  tau={row['cov_tau']:.2f} scale={row['scale']:<4g} win={row['win']} "
            f"acc={row['acc']:.3f} score={row['score_mean']:.3f} budget={row['avg_budget']:.3f}  {cmp}"
        )

    print("\npareto front:")
    for row in pareto(cov):
        cmp = "acc>fixed-ceiling" if row["b_eq"] is None else f"{row['saving']:+.1f}% vs fixed b{row['b_eq']:.3f}"
        print(
            f"  tau={row['cov_tau']:.2f} scale={row['scale']:<4g} win={row['win']} "
            f"acc={row['acc']:.3f} budget={row['avg_budget']:.3f}  {cmp}"
        )


def main():
    for task in ARGS.tasks:
        report(task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ARGS = ap.parse_args()
    main()
