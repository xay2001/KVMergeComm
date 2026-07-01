#!/usr/bin/env python3
"""Plot actual per-sample budget distribution for B-ReKV.

This figure is meant to show where "aware" appears: tau/scale are global
hyperparameters, but the actual transmitted KV budget varies per query because
it is induced by receiver-attention coverage.

Usage:
  python scripts/plot_budget_distribution.py \
    --path snapshots/musique/coverage/cov_t0.95_s0.75_w8_0623_2305/per_sample.jsonl
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_budgets(path):
    budgets, scores = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                continue
            if "budget" in row:
                budgets.append(float(row["budget"]))
            if "score" in row:
                scores.append(float(row["score"]))
    return budgets, scores


def setting_name(path):
    base = os.path.basename(os.path.dirname(path))
    m = re.search(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)", base)
    if not m:
        return base
    return f"B-ReKV (tau={float(m.group(1)):.2f}, scale={float(m.group(2)):g}, w={m.group(3)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--fixed_refs", nargs="*", type=float, default=[0.3, 0.5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budgets, scores = load_budgets(args.path)
    if not budgets:
        raise SystemExit(f"No per-sample budget field found in {args.path}")

    mean = sum(budgets) / len(budgets)
    sorted_b = sorted(budgets)
    def pct(p):
        idx = min(max(int(round((len(sorted_b) - 1) * p)), 0), len(sorted_b) - 1)
        return sorted_b[idx]

    title_task = args.task or os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.path))))
    title = f"{title_task}: Per-query KV budget distribution"
    subtitle = setting_name(args.path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax1.hist(budgets, bins=28, color="#e45756", alpha=0.82, edgecolor="white")
    ax1.axvline(mean, color="#222222", lw=2, label=f"mean={mean:.3f}")
    for r in args.fixed_refs:
        ax1.axvline(r, color="#4c78a8", ls="--", lw=1.5, alpha=0.8, label=f"fixed r={r:g}")
    ax1.set_xlabel("Actual KV budget per query (kept fraction)")
    ax1.set_ylabel("Number of samples")
    ax1.set_title("Budget histogram")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.boxplot(
        budgets,
        vert=False,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#e45756", "alpha": 0.65},
        medianprops={"color": "black", "linewidth": 2},
    )
    for r in args.fixed_refs:
        ax2.axvline(r, color="#4c78a8", ls="--", lw=1.5, alpha=0.8)
    ax2.set_yticks([])
    ax2.set_xlabel("Actual KV budget per query (kept fraction)")
    ax2.set_title("Spread across queries")
    ax2.grid(axis="x", alpha=0.25)
    txt = (
        f"N={len(budgets)}\n"
        f"mean={mean:.3f}\n"
        f"p25={pct(0.25):.3f}\n"
        f"p50={pct(0.50):.3f}\n"
        f"p75={pct(0.75):.3f}\n"
        f"min={min(budgets):.3f}, max={max(budgets):.3f}"
    )
    ax2.text(0.98, 0.05, txt, transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"})

    fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.path), "budget_distribution.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
