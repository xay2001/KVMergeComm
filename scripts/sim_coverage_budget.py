#!/usr/bin/env python3
"""Offline pre-check for Receiver-Aware Coverage Budget.

This is a zero-GPU simulation. It reuses:

  1) Pass-1 feature dumps:
       snapshots/<task>/features/**/per_sample_feat.jsonl
     which contain rcap50/90/95 = fraction of tokens needed to cover
     50/90/95% of receiver-attention mass.

  2) Probe scores:
       snapshots/<task>/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl
     which contain the actual score at fixed budgets.

For each sample, the policy chooses a dynamic budget from rcapXX, optionally
scales/clamps it, snaps it to the nearest available probe rung, and looks up the
sample score. The output compares this training-free coverage policy against
the fixed-r RASC curve.

Usage:
  python scripts/sim_coverage_budget.py --tasks hotpotqa musique twowikimqa
  python scripts/sim_coverage_budget.py --tasks hotpotqa --features rcap90_mean rcap95_mean --scales 0.8 1.0 1.2
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, Iterable, List, Tuple


R_RE = re.compile(r"probe_recv_w16_r([0-9.]+)")


def load_scores(task: str) -> Dict[float, Dict[int, float]]:
    """Return {budget_r: {sample_idx: score}} for dense receiver-w16 probes."""
    out: Dict[float, Dict[int, float]] = {}
    pattern = f"snapshots/{task}/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl"
    for path in glob.glob(pattern):
        m = R_RE.search(os.path.dirname(path))
        if not m:
            continue
        r = float(m.group(1))
        rows: Dict[int, float] = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "_meta" in row:
                    continue
                rows[int(row["idx"])] = float(row["score"])
        out[r] = rows
    return out


def load_features(task: str) -> Dict[int, Dict[str, float]]:
    """Return {sample_idx: feature_dict} from the latest available feature dump."""
    paths = sorted(
        glob.glob(f"snapshots/{task}/features/**/per_sample_feat.jsonl", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    out: Dict[int, Dict[str, float]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "_meta" in row:
                    continue
                idx = int(row["idx"])
                out[idx] = {k: float(v) for k, v in row.items() if k not in ("idx", "id")}
        if out:
            return out
    return out


def common_indices(scores: Dict[float, Dict[int, float]], feats: Dict[int, Dict[str, float]]) -> List[int]:
    common = set(feats)
    for rows in scores.values():
        common &= set(rows)
    return sorted(common)


def fixed_curve(scores: Dict[float, Dict[int, float]], idxs: Iterable[int], tau: float) -> List[Tuple[float, float]]:
    idxs = list(idxs)
    curve = []
    for r in sorted(scores):
        acc = sum(1 for i in idxs if scores[r][i] >= tau) / max(len(idxs), 1)
        curve.append((r, acc))
    return curve


def interp_budget_at_acc(curve: List[Tuple[float, float]], acc: float):
    """Linear interpolation: cheapest fixed budget reaching a target accuracy."""
    pts = sorted(curve, key=lambda x: x[1])
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


def snap_to_rung(raw_budget: float, rungs: List[float], mode: str) -> float:
    if mode == "ceil":
        for r in rungs:
            if r >= raw_budget - 1e-12:
                return r
        return rungs[-1]
    if mode == "nearest":
        return min(rungs, key=lambda r: abs(r - raw_budget))
    if mode == "floor":
        ans = rungs[0]
        for r in rungs:
            if r <= raw_budget + 1e-12:
                ans = r
        return ans
    raise ValueError(f"unknown snap mode: {mode}")


def simulate_policy(
    scores: Dict[float, Dict[int, float]],
    feats: Dict[int, Dict[str, float]],
    idxs: List[int],
    feature_name: str,
    scale: float,
    min_budget: float,
    max_budget: float,
    tau: float,
    snap: str,
):
    rungs = sorted(scores)
    solved = 0
    budget_sum = 0.0
    raw_sum = 0.0
    rung_hist = {r: 0 for r in rungs}

    for idx in idxs:
        raw = feats[idx][feature_name] * scale
        raw = min(max(raw, min_budget), max_budget)
        r = snap_to_rung(raw, rungs, snap)
        raw_sum += raw
        budget_sum += r
        rung_hist[r] += 1
        solved += 1 if scores[r][idx] >= tau else 0

    n = max(len(idxs), 1)
    return {
        "acc": solved / n,
        "avg_budget": budget_sum / n,
        "avg_raw_budget": raw_sum / n,
        "hist": rung_hist,
    }


def pareto(rows):
    """Keep rows that improve accuracy as budget increases."""
    front = []
    for row in sorted(rows, key=lambda x: (x["avg_budget"], -x["acc"])):
        if not front or row["acc"] > front[-1]["acc"] + 1e-12:
            front.append(row)
    return front


def report_task(task: str, args):
    scores = load_scores(task)
    feats = load_features(task)
    if not scores or not feats:
        print(f"\n=== {task} ===")
        print(f"  [skip] scores={len(scores)} feature_rows={len(feats)}")
        return
    idxs = common_indices(scores, feats)
    if not idxs:
        print(f"\n=== {task} ===")
        print("  [skip] no common sample ids across scores/features")
        return

    feature_names = set(next(iter(feats.values())).keys())
    features = [f for f in args.features if f in feature_names]
    missing = [f for f in args.features if f not in feature_names]

    fixed = fixed_curve(scores, idxs, args.tau)
    print(f"\n=== {task} | N={len(idxs)} | tau={args.tau} | snap={args.snap} ===")
    print("fixed-r: " + ", ".join(f"r{r:g}:acc{acc:.3f}" for r, acc in fixed))
    if missing:
        print("missing features: " + ", ".join(missing))

    rows = []
    for feat in features:
        for scale in args.scales:
            sim = simulate_policy(
                scores=scores,
                feats=feats,
                idxs=idxs,
                feature_name=feat,
                scale=scale,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                tau=args.tau,
                snap=args.snap,
            )
            b_eq = interp_budget_at_acc(fixed, sim["acc"])
            saving = None if b_eq is None or b_eq <= 0 else (1 - sim["avg_budget"] / b_eq) * 100
            row = {
                "feat": feat,
                "scale": scale,
                "acc": sim["acc"],
                "avg_budget": sim["avg_budget"],
                "avg_raw_budget": sim["avg_raw_budget"],
                "b_eq": b_eq,
                "saving": saving,
                "hist": sim["hist"],
            }
            rows.append(row)

    print("\ncoverage-budget policies:")
    for row in sorted(rows, key=lambda x: (x["feat"], x["scale"])):
        if row["b_eq"] is None:
            cmp = "acc>fixed-ceiling"
        else:
            cmp = f"{row['saving']:+.1f}% vs fixed b{row['b_eq']:.3f}"
        print(
            f"  {row['feat']} x{row['scale']:<4g}  "
            f"acc={row['acc']:.3f}  budget={row['avg_budget']:.3f} "
            f"(raw={row['avg_raw_budget']:.3f})  {cmp}"
        )

    print("\npareto front (best coverage policies by avg budget):")
    for row in pareto(rows):
        cmp = "acc>fixed-ceiling" if row["b_eq"] is None else f"{row['saving']:+.1f}% vs fixed b{row['b_eq']:.3f}"
        print(
            f"  {row['feat']} x{row['scale']:<4g}  "
            f"acc={row['acc']:.3f}  budget={row['avg_budget']:.3f}  {cmp}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.5, help="Solved threshold on the task score.")
    ap.add_argument(
        "--features",
        nargs="+",
        default=["rcap50_mean", "rcap90_mean", "rcap95_mean"],
        help="Pass-1 coverage features to use as raw budgets.",
    )
    ap.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25, 1.5, 2.0],
        help="Multipliers applied to rcapXX before clamping/snapping.",
    )
    ap.add_argument("--min_budget", type=float, default=0.05)
    ap.add_argument("--max_budget", type=float, default=0.7)
    ap.add_argument(
        "--snap",
        choices=["ceil", "nearest", "floor"],
        default="ceil",
        help="How to map continuous predicted budgets to available probe rungs.",
    )
    args = ap.parse_args()

    for task in args.tasks:
        report_task(task, args)


if __name__ == "__main__":
    main()
