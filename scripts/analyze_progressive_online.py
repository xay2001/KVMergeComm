#!/usr/bin/env python3
"""Step-2b analysis: sweep the stop-threshold theta over recorded progressive runs.

run_progressive.sh records, for each sample and each budget rung, the answer score
plus uncertainty signals (entropy / top-2 margin). Given a stop rule

    confident(rung)  iff   uncertainty <= theta      (entropy signals)
                     iff   uncertainty >= theta      (margin signals)

we replay the policy: walk the ladder low->high, stop at the first confident rung,
else exhaust it. Because each rung's generation is independent, replaying offline
gives the EXACT accuracy / avg-budget / avg-rounds the online policy would achieve
(only wall-clock latency needs a真早停 run -> Step 3).

Reports the accuracy-budget Pareto front across theta, plus the ORACLE ceiling
(stop when solved) and FIXED-r floor (each single rung) for reference.

Usage:
  python scripts/analyze_progressive_online.py snapshots/hotpotqa/progressive --tau 0.5
  python scripts/analyze_progressive_online.py snapshots/hotpotqa/progressive --signal margin_first
"""
import argparse
import glob
import json
import os

SIGNALS = {
    "ent_first": "le",     # confident if entropy <= theta
    "ent_mean": "le",
    "margin_first": "ge",  # confident if margin >= theta
    "margin_mean": "ge",
}


def load(run_dir):
    paths = glob.glob(os.path.join(run_dir, "**", "per_sample_prog.jsonl"), recursive=True)
    if not paths:
        return None, None
    meta, rows = {}, []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "_meta" in r:
                    meta = r["_meta"]
                    continue
                rows.append(r)
        if rows:
            break
    return meta, rows


def simulate(rows, signal, theta, tau):
    """Replay the stop policy; return (acc, avg_budget, avg_rounds)."""
    cmp_ge = SIGNALS[signal] == "ge"
    n = len(rows)
    solved = 0
    bud_sum = 0.0
    rounds_sum = 0
    for row in rows:
        rungs = row["rungs"]
        chosen = rungs[-1]
        used = len(rungs)
        for k, rung in enumerate(rungs):
            u = rung[signal]
            confident = (u >= theta) if cmp_ge else (u <= theta)
            if confident:
                chosen = rung
                used = k + 1
                break
        rounds_sum += used
        bud_sum += chosen["budget"] if chosen["budget"] is not None else chosen["r"]
        if chosen["score"] >= tau:
            solved += 1
    return solved / n, bud_sum / n, rounds_sum / n


def oracle(rows, tau):
    """Stop at the first rung that solves the sample (ceiling)."""
    n = len(rows); solved = 0; bud = 0.0; rd = 0
    for row in rows:
        rungs = row["rungs"]; chosen = rungs[-1]; used = len(rungs)
        for k, rung in enumerate(rungs):
            if rung["score"] >= tau:
                chosen = rung; used = k + 1; break
        rd += used
        bud += chosen["budget"] if chosen["budget"] is not None else chosen["r"]
        if chosen["score"] >= tau:
            solved += 1
    return solved / n, bud / n, rd / n


def fixed_points(rows, tau):
    """Per-rung fixed-r accuracy / budget (the no-feedback floor)."""
    ladder = [rg["r"] for rg in rows[0]["rungs"]]
    out = []
    for k, r in enumerate(ladder):
        acc = sum(1 for row in rows if row["rungs"][k]["score"] >= tau) / len(rows)
        b = sum((row["rungs"][k]["budget"] or r) for row in rows) / len(rows)
        out.append((r, b, acc))
    return out


def pareto_front(points):
    """points: list of (theta, acc, budget, rounds). Keep non-dominated (high acc, low budget)."""
    front = []
    for p in sorted(points, key=lambda x: x[2]):  # by budget asc
        if not front or p[1] > front[-1][1] + 1e-9:  # strictly better acc
            front.append(p)
    return front


def report(run_dir, tau, signal):
    meta, rows = load(run_dir)
    if not rows:
        print(f"  [skip] no per_sample_prog.jsonl under {run_dir}")
        return
    name = meta.get("dataset", os.path.basename(run_dir.rstrip("/")))
    ladder = meta.get("ladder", [rg["r"] for rg in rows[0]["rungs"]])
    print(f"\n=== {name} | progressive | signal={signal} | tau={tau} | N={len(rows)} ===")
    print(f"ladder: {ladder}")

    # fixed-r floor
    print("\nFIXED-r (floor): " + ", ".join(
        f"r{r}:acc{acc:.3f}@b{b:.3f}" for r, b, acc in fixed_points(rows, tau)))
    # oracle ceiling
    oacc, obud, ornd = oracle(rows, tau)
    print(f"ORACLE (ceiling): acc {oacc:.3f}  avg_budget {obud:.3f}  avg_rounds {ornd:.2f}")

    # theta sweep
    vals = sorted(rung[signal] for row in rows for rung in row["rungs"])
    grid = [vals[int(q * (len(vals) - 1))] for q in
            [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]]
    grid = sorted(set(round(v, 4) for v in grid))
    pts = []
    print(f"\ntheta sweep ({signal}, confident if "
          f"{'>=' if SIGNALS[signal]=='ge' else '<='} theta):")
    print(f"{'theta':>8} {'acc':>7} {'avg_budget':>11} {'avg_rounds':>11}")
    for th in grid:
        acc, bud, rnd = simulate(rows, signal, th, tau)
        pts.append((th, acc, bud, rnd))
        print(f"{th:>8.3f} {acc:>7.3f} {bud:>11.3f} {rnd:>11.2f}")

    front = pareto_front(pts)
    print("\nPareto front (theta -> acc @ budget, rounds):")
    for th, acc, bud, rnd in front:
        # budget saving vs cheapest fixed-r matching this acc
        fp = fixed_points(rows, tau)
        match = min((b for r, b, a in fp if a >= 0.99 * acc), default=None)
        save = f"{(1-bud/match)*100:+.1f}% vs fixed b{match:.3f}" if match else "n/a"
        print(f"  theta={th:.3f}  acc={acc:.3f}  budget={bud:.3f}  rounds={rnd:.2f}   ({save})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="snapshots/<task>/progressive dirs")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--signal", default="ent_first", choices=list(SIGNALS.keys()))
    args = ap.parse_args()
    for d in args.run_dirs:
        report(d, args.tau, args.signal)


if __name__ == "__main__":
    main()
