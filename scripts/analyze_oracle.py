#!/usr/bin/env python3
"""Step-0 budget headroom analysis.

Reads per_sample.jsonl files (emitted by eval.py) under snapshots/<dataset>/...,
joins samples across budgets by index, and quantifies:

  - the distribution of each sample's *oracle minimal budget* (smallest r that
    "solves" it), i.e. whether different queries genuinely need different budgets;
  - how much budget a single fixed-r policy wastes vs the oracle (headroom);
  - budget regret and interval coverage (BAGEN-style calibration baselines).

Usage:
  python scripts/analyze_oracle.py snapshots/musique --method receiver_w16 --tau 0.5
  python scripts/analyze_oracle.py snapshots/twowikimqa snapshots/musique --tau 0.5
"""
import argparse
import json
import os
import re
import glob
from collections import defaultdict

RUN_RE = [
    (re.compile(r"kvcomm_top([0-9.]+)"), "kvcomm"),
    (re.compile(r"merge_r([0-9.]+)"), "merge"),
    (re.compile(r"evict_r([0-9.]+)"), "evict"),
    (re.compile(r"recv_w(\d+)_r([0-9.]+)"), "receiver"),
]


def parse_run(dirname):
    """-> (method, budget) or (None, None)."""
    base = os.path.basename(dirname.rstrip("/"))
    for rgx, name in RUN_RE:
        m = rgx.search(base)
        if not m:
            continue
        if name == "receiver":
            return f"receiver_w{m.group(1)}", float(m.group(2))
        return name, float(m.group(1))
    return None, None


def load_dataset(ds_dir):
    """-> {method: {budget: {idx: score}}}"""
    table = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(ds_dir, "**", "per_sample.jsonl"), recursive=True):
        method, budget = parse_run(os.path.dirname(path))
        if method is None:
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "_meta" in row:
                    continue
                table[method][budget][row["idx"]] = row["score"]
    return table


def oracle_min_budgets(by_budget, tau):
    """For each sample idx, smallest budget whose score >= tau (else None)."""
    budgets = sorted(by_budget.keys())
    # samples present in every budget run (clean join)
    common = None
    for b in budgets:
        ks = set(by_budget[b].keys())
        common = ks if common is None else (common & ks)
    common = sorted(common or [])
    out = {}
    for idx in common:
        omb = None
        for b in budgets:
            if by_budget[b][idx] >= tau:
                omb = b
                break
        out[idx] = omb
    return out, budgets, common


def fixed_policy_solved_rate(by_budget, common, tau, r):
    return sum(1 for idx in common if by_budget[r][idx] >= tau) / max(len(common), 1)


def report(ds_dir, method, tau):
    table = load_dataset(ds_dir)
    if method not in table:
        avail = ", ".join(sorted(table.keys())) or "(none)"
        print(f"  [skip] method '{method}' not found in {ds_dir}. available: {avail}")
        return
    by_budget = table[method]
    omb, budgets, common = oracle_min_budgets(by_budget, tau)
    n = len(common)
    if n == 0:
        print(f"  [skip] no common samples across budgets in {ds_dir}")
        return

    solved = [b for b in omb.values() if b is not None]
    unsolved = n - len(solved)

    print(f"\n=== {os.path.basename(ds_dir.rstrip('/'))} | method={method} | tau={tau} | N={n} ===")
    print(f"budgets available: {budgets}")
    print(f"solvable@maxbudget: {len(solved)}/{n} ({100*len(solved)/n:.1f}%)   unsolved: {unsolved}")

    # distribution of oracle minimal budget
    hist = defaultdict(int)
    for b in omb.values():
        hist["unsolved" if b is None else b] += 1
    print("\noracle minimal budget distribution:")
    cum = 0
    for b in budgets:
        c = hist.get(b, 0)
        cum += c
        print(f"  r={b:<4} : {c:4d}  ({100*c/n:5.1f}%)   cum {100*cum/n:5.1f}%")
    if unsolved:
        print(f"  unsolved : {unsolved:4d}  ({100*unsolved/n:5.1f}%)")

    # fixed-policy vs oracle headroom (over solvable samples)
    if solved:
        avg_oracle = sum(solved) / len(solved)
        # best fixed r = smallest budget reaching >=99% of max solved-rate
        rates = {r: fixed_policy_solved_rate(by_budget, common, tau, r) for r in budgets}
        max_rate = max(rates.values())
        best_fixed = min((r for r in budgets if rates[r] >= 0.99 * max_rate), default=budgets[-1])
        print(f"\nfixed-policy solved-rate by r: " + ", ".join(f"{r}:{rates[r]:.2f}" for r in budgets))
        print(f"best fixed r (>=99% of max rate {max_rate:.2f}): {best_fixed}")
        print(f"avg oracle budget (solvable): {avg_oracle:.3f}   vs fixed {best_fixed}")
        if best_fixed > 0:
            print(f"  -> budget saving vs best-fixed: {100*(best_fixed-avg_oracle)/best_fixed:.1f}%")
        # regret of best fixed policy
        regrets = [best_fixed - omb[idx] for idx in common if omb[idx] is not None]
        if regrets:
            avg_reg = sum(regrets) / len(regrets)
            over = sum(1 for x in regrets if x > 0)
            print(f"  fixed-r regret (best_fixed - oracle): mean {avg_reg:.3f}, "
                  f"over-provisioned {over}/{len(regrets)} ({100*over/len(regrets):.1f}%)")

    # interval coverage examples
    print("\ninterval coverage (oracle <= r_high):")
    for rh in budgets:
        cov = sum(1 for b in solved if b <= rh) / max(len(solved), 1)
        print(f"  r_high={rh:<4}: covers {100*cov:.1f}% of solvable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="snapshots/<dataset> dirs")
    ap.add_argument("--method", default="receiver_w16",
                    help="method family: kvcomm | merge | evict | receiver_w8 | receiver_w16")
    ap.add_argument("--tau", type=float, default=0.5, help="solved threshold on the metric")
    args = ap.parse_args()
    for ds in args.datasets:
        report(ds, args.method, args.tau)


if __name__ == "__main__":
    main()
