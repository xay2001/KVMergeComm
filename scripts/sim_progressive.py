#!/usr/bin/env python3
"""Step-2a: offline simulation of progressive (multi-round) KV communication.

Reuses the per-sample scores already collected at multiple budgets (receiver-w16,
uniform) to simulate a closed-loop policy WITHOUT any extra GPU work:

  start at the lowest budget rung; if the receiver is still "unsure", request the
  next (higher) budget; stop once solved (or the ladder is exhausted).

Offline we do not have the receiver's runtime uncertainty signal, so we bracket the
achievable region with two policies:

  * ORACLE-stop  (ceiling): escalate iff the current rung did not solve the sample.
        This is the best any uncertainty trigger could do -> lower bound on budget /
        rounds at full (solvable) accuracy.
  * FIXED-r      (floor):   no feedback, every sample pays a single fixed budget.
        This is the current RASC baseline.

A real trigger (Step-2b, needs GPU) lives between these two; the oracle-vs-fixed gap
is exactly the prize an uncertainty predictor must capture.

Cost model: top-k selection at higher r is (near-)nested in lower r, so a round only
transmits the *increment*. Total transmitted KV fraction ~= the final rung reached
(incremental). We also report the restart cost (sum of rungs visited) as an upper
bound for protocols that re-send from scratch.

Usage:
  python scripts/sim_progressive.py snapshots/hotpotqa --tau 0.5
  python scripts/sim_progressive.py snapshots/hotpotqa snapshots/musique snapshots/twowikimqa \
      --tau 0.5 --ladder 0.05,0.1,0.15,0.2,0.3,0.5 --cap 3
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

R_RE = re.compile(r"_r([0-9.]+)")


def load_budget_scores(ds_dir):
    """-> {budget: {idx: score}} for receiver-w16 uniform runs (probe + budget/uniform)."""
    patterns = [
        os.path.join(ds_dir, "mtc_receiver", "probe_recv_w16_r*", "per_sample.jsonl"),
        os.path.join(ds_dir, "mtc_receiver", "recv_w16_r*", "per_sample.jsonl"),
        os.path.join(ds_dir, "budget", "uniform_r*", "per_sample.jsonl"),
    ]
    table = defaultdict(dict)
    for pat in patterns:
        for path in glob.glob(pat):
            m = R_RE.search(os.path.basename(os.path.dirname(path)))
            if not m:
                continue
            b = float(m.group(1))
            for line in open(path):
                row = json.loads(line)
                if "_meta" in row:
                    continue
                # prefer denser probe source; don't overwrite if already present
                table[b].setdefault(row["idx"], row["score"])
    return table


def common_idx(table, ladder):
    common = None
    for b in ladder:
        ks = set(table[b].keys())
        common = ks if common is None else (common & ks)
    return sorted(common or [])


def simulate(table, ladder, idxs, tau, cap):
    """ORACLE-stop progressive over `ladder` (ascending), capped at `cap` rounds."""
    rungs = ladder if cap is None else ladder[:cap]
    n = len(idxs)
    solved = 0
    rounds_sum = 0
    final_sum = 0.0      # incremental cost (final rung reached)
    restart_sum = 0.0    # sum of visited rungs
    for idx in idxs:
        visited = 0.0
        hit = None
        for r in rungs:
            visited += r
            if table[r][idx] >= tau:
                hit = r
                break
        rounds_used = (rungs.index(hit) + 1) if hit is not None else len(rungs)
        rounds_sum += rounds_used
        if hit is not None:
            solved += 1
            final_sum += hit
            restart_sum += visited
        else:
            final_sum += rungs[-1]
            restart_sum += visited
    return {
        "n": n,
        "acc": solved / n,
        "avg_rounds": rounds_sum / n,
        "avg_budget_incr": final_sum / n,
        "avg_budget_restart": restart_sum / n,
    }


def fixed_curve(table, ladder, idxs, tau):
    """fixed-r solved-rate per rung."""
    return {r: sum(1 for i in idxs if table[r][i] >= tau) / len(idxs) for r in ladder}


def report(ds_dir, tau, ladder_arg, cap):
    table = load_budget_scores(ds_dir)
    avail = sorted(table.keys())
    name = os.path.basename(ds_dir.rstrip("/"))
    if len(avail) < 2:
        print(f"\n=== {name} ===\n  [skip] need >=2 budgets, found {avail}")
        return
    ladder = [b for b in (ladder_arg or avail) if b in table]
    ladder = sorted(set(ladder))
    idxs = common_idx(table, ladder)
    if not idxs:
        print(f"\n=== {name} ===\n  [skip] no common samples across {ladder}")
        return

    sim = simulate(table, ladder, idxs, tau, cap)
    fixed = fixed_curve(table, ladder, idxs, tau)
    max_rate = max(fixed.values())
    # cheapest fixed-r matching the progressive accuracy (>=99% of it)
    target = 0.99 * sim["acc"]
    best_fixed = min((r for r in ladder if fixed[r] >= target), default=ladder[-1])

    print(f"\n=== {name} | tau={tau} | N={sim['n']} ===")
    print(f"ladder: {ladder}" + (f"  (cap={cap} rounds)" if cap else ""))
    print(f"fixed-r solved-rate: " + ", ".join(f"{r}:{fixed[r]:.2f}" for r in ladder))
    print(f"\nORACLE-progressive (ceiling):")
    print(f"  accuracy           : {sim['acc']:.3f}  (max fixed rate {max_rate:.3f})")
    print(f"  avg rounds         : {sim['avg_rounds']:.2f}")
    print(f"  avg budget (incr)  : {sim['avg_budget_incr']:.3f}")
    print(f"  avg budget (restart): {sim['avg_budget_restart']:.3f}")
    print(f"\nvs FIXED-r baseline (floor):")
    print(f"  cheapest fixed-r matching acc {sim['acc']:.3f}: r={best_fixed}")
    if best_fixed > 0:
        save = (1 - sim["avg_budget_incr"] / best_fixed) * 100
        print(f"  -> progressive avg budget {sim['avg_budget_incr']:.3f} vs fixed {best_fixed} "
              f"= {save:+.1f}% budget saving at equal accuracy")
    # round distribution
    dist = defaultdict(int)
    for idx in idxs:
        rungs = ladder if cap is None else ladder[:cap]
        ru = len(rungs)
        for k, r in enumerate(rungs):
            if table[r][idx] >= tau:
                ru = k + 1
                break
        dist[ru] += 1
    print("  rounds distribution: " + ", ".join(
        f"{k}r:{100*dist[k]/sim['n']:.0f}%" for k in sorted(dist)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="snapshots/<dataset> dirs")
    ap.add_argument("--tau", type=float, default=0.5, help="solved threshold on the metric")
    ap.add_argument("--ladder", type=str, default=None,
                    help="comma list of budget rungs, e.g. 0.05,0.1,0.15,0.2,0.3,0.5 (default: all available)")
    ap.add_argument("--cap", type=int, default=None, help="max rounds (truncate ladder)")
    args = ap.parse_args()
    ladder = [float(x) for x in args.ladder.split(",")] if args.ladder else None
    for ds in args.datasets:
        report(ds, args.tau, ladder, args.cap)


if __name__ == "__main__":
    main()
