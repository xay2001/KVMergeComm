#!/usr/bin/env python3
"""Plot fixed-r RASC vs Coverage-BRASC accuracy-budget curve.

Default is MuSiQue because it is the clearest Coverage-BRASC result.

Usage:
  python scripts/plot_coverage_pareto.py --task musique --tau 0.5
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROBE_RE = re.compile(r"probe_recv_w16_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")


def read_rows(path):
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                continue
            rows[int(row["idx"])] = row
    return rows


def load_fixed(task, tau):
    points = []
    for path in glob.glob(f"snapshots/{task}/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl"):
        m = PROBE_RE.search(os.path.dirname(path))
        if not m:
            continue
        r = float(m.group(1))
        rows = read_rows(path)
        scores = [float(row["score"]) for row in rows.values()]
        acc = sum(1 for s in scores if s >= tau) / max(len(scores), 1)
        points.append((r, acc))
    return sorted(points)


def load_coverage(task, tau):
    # Deduplicate repeated runs by keeping the latest path per (tau, scale, win).
    latest = {}
    for path in glob.glob(f"snapshots/{task}/coverage/**/per_sample.jsonl", recursive=True):
        base = os.path.basename(os.path.dirname(path))
        m = COV_RE.search(base)
        if not m:
            continue
        key = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
        if key not in latest or os.path.getmtime(path) > os.path.getmtime(latest[key]):
            latest[key] = path

    points = []
    for (cov_tau, scale, win), path in latest.items():
        rows = read_rows(path)
        scores = [float(row["score"]) for row in rows.values()]
        budgets = [float(row.get("budget", 0.0)) for row in rows.values()]
        acc = sum(1 for s in scores if s >= tau) / max(len(scores), 1)
        avg_budget = sum(budgets) / max(len(budgets), 1)
        points.append({
            "coverage_tau": cov_tau,
            "scale": scale,
            "win": win,
            "acc": acc,
            "budget": avg_budget,
            "label": f"t{cov_tau:.2f} s{scale:g} w{win}",
        })
    return sorted(points, key=lambda p: (p["win"], p["coverage_tau"], p["scale"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="musique")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fixed = load_fixed(args.task, args.tau)
    coverage = load_coverage(args.task, args.tau)
    if not fixed:
        raise SystemExit(f"No fixed probe curve found for {args.task}")
    if not coverage:
        raise SystemExit(f"No coverage runs found for {args.task}")

    plt.figure(figsize=(8.5, 5.8))
    fx, fy = zip(*fixed)
    plt.plot(fx, fy, "o-", color="#4c78a8", lw=2.2, ms=6, label="Fixed-r RASC")

    markers = {8: "s", 16: "^"}
    colors = {8: "#e45756", 16: "#54a24b"}
    for win in sorted({p["win"] for p in coverage}):
        pts = [p for p in coverage if p["win"] == win]
        plt.scatter(
            [p["budget"] for p in pts],
            [p["acc"] for p in pts],
            marker=markers.get(win, "o"),
            s=72,
            alpha=0.88,
            color=colors.get(win, None),
            edgecolors="white",
            linewidths=0.8,
            label=f"Coverage-BRASC (w{win})",
        )

    # Label the strongest MuSiQue points.
    highlights = [
        (0.95, 0.75, 8),
        (0.95, 0.85, 8),
        (0.90, 0.75, 8),
        (0.95, 0.90, 16),
    ]
    for ht, hs, hw in highlights:
        match = [p for p in coverage if p["coverage_tau"] == ht and p["scale"] == hs and p["win"] == hw]
        if not match:
            continue
        p = match[0]
        plt.annotate(
            p["label"],
            (p["budget"], p["acc"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    best_fixed_acc = max(y for _, y in fixed)
    plt.axhline(best_fixed_acc, color="#4c78a8", ls="--", lw=1, alpha=0.45)
    plt.text(0.055, best_fixed_acc + 0.004, f"fixed-r ceiling={best_fixed_acc:.3f}", fontsize=8, color="#4c78a8")

    plt.xlabel("Average KV Budget (kept fraction)")
    plt.ylabel("Accuracy / Score")
    plt.title(f"{args.task}: Fixed-r RASC vs Coverage-BRASC")
    plt.grid(alpha=0.25)
    plt.xlim(0.04, 0.72)
    plt.ylim(max(0, min(fy) - 0.04), min(1.0, max([p["acc"] for p in coverage] + list(fy)) + 0.04))
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    out = args.out or f"snapshots/{args.task}/coverage_pareto.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
