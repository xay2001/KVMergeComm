#!/usr/bin/env python3
"""Step-1 budget-allocation analysis.

Reads per_sample.jsonl under snapshots/<dataset>/budget/* (emitted by run_budget.sh)
and compares budget modes against the uniform RASC baseline on two axes:

  - equal-budget gain:  at the *same* achieved KV budget, does layer / query+layer
    allocation give a higher mean score than uniform?
  - equal-accuracy saving:  to reach a given score, how much less budget does the
    adaptive policy need vs the uniform curve (interpolated)?

Each run reports its *achieved* transmitted-KV fraction (mean of per-sample
`budget`), so adaptive runs are placed on the same x-axis as uniform.

Usage:
  python scripts/analyze_budget.py snapshots/hotpotqa
  python scripts/analyze_budget.py snapshots/hotpotqa snapshots/musique
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load_runs(ds_dir):
    """-> list of {mode, label, avg_budget, avg_score, avg_qbudget, n}."""
    runs = []
    for path in glob.glob(os.path.join(ds_dir, "budget", "**", "per_sample.jsonl"), recursive=True):
        meta, scores, budgets, qbudgets = {}, [], [], []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "_meta" in row:
                    meta = row["_meta"]
                    continue
                scores.append(row["score"])
                if "budget" in row:
                    budgets.append(row["budget"])
                if "query_budget" in row:
                    qbudgets.append(row["query_budget"])
        if not scores:
            continue
        mode = meta.get("budget_mode") or "uniform"
        # achieved budget: prefer measured kept-ratio, else nominal merge_ratio
        if budgets:
            avg_budget = sum(budgets) / len(budgets)
        else:
            avg_budget = float(meta.get("merge_ratio") or 0.0)
        runs.append({
            "mode": mode,
            "label": os.path.basename(os.path.dirname(path)),
            "avg_budget": avg_budget,
            "avg_score": sum(scores) / len(scores),
            "avg_qbudget": (sum(qbudgets) / len(qbudgets)) if qbudgets else None,
            "n": len(scores),
        })
    return runs


def interp(curve, x):
    """Linear interpolation of score at budget x on a (budget, score) curve."""
    pts = sorted(curve)
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def budget_for_score(curve, target):
    """Smallest budget on the curve whose interpolated score >= target."""
    pts = sorted(curve)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        lo, hi = min(y0, y1), max(y0, y1)
        if lo <= target <= hi and y1 != y0:
            t = (target - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    for x, y in pts:
        if y >= target:
            return x
    return None


def report(ds_dir):
    runs = load_runs(ds_dir)
    if not runs:
        print(f"  [skip] no budget runs under {ds_dir}/budget")
        return
    name = os.path.basename(ds_dir.rstrip("/"))
    print(f"\n=== {name} | budget allocation ===")
    print(f"{'run':<22} {'mode':<12} {'avg_budget':>10} {'score':>8} {'qbudget':>8} {'n':>5}")
    by_mode = defaultdict(list)
    for r in sorted(runs, key=lambda r: (r["mode"], r["avg_budget"])):
        by_mode[r["mode"]].append(r)
        qb = f"{r['avg_qbudget']:.3f}" if r["avg_qbudget"] is not None else "-"
        print(f"{r['label']:<22} {r['mode']:<12} {r['avg_budget']:>10.3f} "
              f"{r['avg_score']:>8.4f} {qb:>8} {r['n']:>5}")

    uniform = [(r["avg_budget"], r["avg_score"]) for r in by_mode.get("uniform", [])]
    if len(uniform) < 2:
        print("  (need >=2 uniform points for equal-budget / equal-accuracy comparison)")
        return

    print("\n-- vs uniform curve (interp) --")
    for mode in ("layer", "query", "query+layer"):
        for r in by_mode.get(mode, []):
            u_at_b = interp(uniform, r["avg_budget"])
            gain = r["avg_score"] - u_at_b if u_at_b is not None else None
            # b_eq = uniform budget needed to reach this run's score.
            # adaptive used avg_budget; saving = how much less it used than uniform.
            b_eq = budget_for_score(uniform, r["avg_score"])
            saving = (1 - r["avg_budget"] / b_eq) * 100 if (b_eq and b_eq > 0) else None
            g = f"{gain:+.4f}" if gain is not None else "  n/a"
            s = f"{saving:+.1f}%" if saving is not None else "  n/a"
            print(f"  {r['label']:<22} @b={r['avg_budget']:.3f}  "
                  f"score {r['avg_score']:.4f} vs uniform {u_at_b:.4f} ({g})   "
                  f"equal-acc budget {('%.3f'%b_eq) if b_eq else 'n/a'} (saving {s})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="snapshots/<dataset> dirs")
    args = ap.parse_args()
    for ds in args.datasets:
        report(ds)


if __name__ == "__main__":
    main()
