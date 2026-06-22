#!/usr/bin/env python3
"""Step-2b decisive test: can a LEARNED combination of uncertainty signals drive the
progressive stop decision better than any single hand-thresholded signal?

For each (sample, budget-rung) we have 6 signals (entropy/margin first&mean,
ctx_mass, ctx_conc) + the rung budget r, and the answer score. We train a logistic
regression to predict P(this rung solves the sample), using GROUP k-fold (a sample's
rungs never split across train/test -> no leakage), and collect out-of-fold
probabilities. The progressive policy then stops at the first rung whose predicted
P(solved) >= theta'. We sweep theta' to trace the accuracy-budget Pareto front and
compare against the fixed-r floor and the oracle ceiling.

Verdict:
  - learned controller clearly above the best single signal & toward oracle  -> budget-aware lives (learnable controller).
  - learned controller still hugging the fixed-r line                        -> output-side signals are fundamentally insufficient.

Usage:
  python scripts/learn_stop_policy.py snapshots/hotpotqa/progressive --tau 0.5
"""
import argparse
import glob
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEATURES = ["ent_first", "ent_mean", "margin_first", "margin_mean", "ctx_mass", "ctx_conc", "r"]


def load(run_dir):
    paths = glob.glob(os.path.join(run_dir, "**", "per_sample_prog.jsonl"), recursive=True)
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


def build_matrix(rows, tau):
    X, y, groups, sample_rung = [], [], [], []
    for si, row in enumerate(rows):
        for ri, rung in enumerate(row["rungs"]):
            X.append([rung.get(f, 0.0) for f in FEATURES])
            y.append(1 if rung["score"] >= tau else 0)
            groups.append(si)
            sample_rung.append((si, ri))
    return np.array(X, float), np.array(y, int), np.array(groups), sample_rung


def oof_probs(X, y, groups, n_splits=5):
    prob = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    for tr, te in gkf.split(X, y, groups):
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    # refit on all to expose coefficients
    clf.fit(X, y)
    coef = clf.named_steps["logisticregression"].coef_[0]
    return prob, dict(zip(FEATURES, coef))


def fixed_points(rows, tau):
    ladder = [rg["r"] for rg in rows[0]["rungs"]]
    out = []
    for k, r in enumerate(ladder):
        acc = sum(1 for row in rows if row["rungs"][k]["score"] >= tau) / len(rows)
        b = sum((row["rungs"][k]["budget"] or r) for row in rows) / len(rows)
        out.append((r, b, acc))
    return out


def budget_at_acc(fp, acc):
    pts = sorted(fp, key=lambda x: x[2])
    if acc <= pts[0][2]:
        return pts[0][1]
    if acc > pts[-1][2] + 1e-9:
        return None
    for (r0, b0, a0), (r1, b1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0:
            t = (acc - a0) / (a1 - a0)
            return b0 + t * (b1 - b0)
    return pts[-1][1]


def oracle(rows, tau):
    n = len(rows); solved = 0; bud = 0.0; rd = 0
    for row in rows:
        rungs = row["rungs"]; chosen = rungs[-1]; used = len(rungs)
        for k, rg in enumerate(rungs):
            if rg["score"] >= tau:
                chosen = rg; used = k + 1; break
        rd += used; bud += chosen["budget"] or chosen["r"]
        solved += 1 if chosen["score"] >= tau else 0
    return solved / n, bud / n, rd / n


def simulate(rows, prob_by_sr, theta, tau):
    n = len(rows); solved = 0; bud = 0.0; rd = 0
    for si, row in enumerate(rows):
        rungs = row["rungs"]; chosen = rungs[-1]; used = len(rungs)
        for ri, rg in enumerate(rungs):
            if prob_by_sr[(si, ri)] >= theta:
                chosen = rg; used = ri + 1; break
        rd += used; bud += chosen["budget"] or chosen["r"]
        solved += 1 if chosen["score"] >= tau else 0
    return solved / n, bud / n, rd / n


def pareto(points):
    front = []
    for p in sorted(points, key=lambda x: x[2]):  # by budget
        if not front or p[1] > front[-1][1] + 1e-9:
            front.append(p)
    return front


def report(run_dir, tau):
    meta, rows = load(run_dir)
    if not rows:
        print(f"  [skip] no per_sample_prog.jsonl under {run_dir}")
        return
    name = meta.get("dataset", os.path.basename(run_dir.rstrip("/")))
    X, y, groups, sr = build_matrix(rows, tau)
    prob, coef = oof_probs(X, y, groups)
    prob_by_sr = {sr[i]: prob[i] for i in range(len(sr))}
    auc = roc_auc_score(y, prob)

    print(f"\n=== {name} | learned stop-policy | tau={tau} | N={len(rows)} ===")
    print(f"ladder: {meta.get('ladder')}")
    print(f"out-of-fold AUC P(solved): {auc:.3f}   (0.5 = signals useless, 1.0 = perfect)")
    print("logreg coefficients (standardized): " +
          ", ".join(f"{k}:{v:+.2f}" for k, v in sorted(coef.items(), key=lambda x: -abs(x[1]))))

    fp = fixed_points(rows, tau)
    print("\nFIXED-r (floor): " + ", ".join(f"r{r}:acc{a:.3f}@b{b:.3f}" for r, b, a in fp))
    oacc, obud, ornd = oracle(rows, tau)
    print(f"ORACLE (ceiling): acc {oacc:.3f}  budget {obud:.3f}  rounds {ornd:.2f}")

    pts = []
    for th in np.linspace(0.05, 0.95, 19):
        acc, bud, rnd = simulate(rows, prob_by_sr, th, tau)
        pts.append((round(float(th), 3), acc, bud, rnd))
    print("\nLEARNED policy theta' sweep:")
    print(f"{'theta':>7} {'acc':>7} {'budget':>8} {'rounds':>7} {'vs_fixed(equal-acc)':>20}")
    for th, acc, bud, rnd in pts:
        bf = budget_at_acc(fp, acc)
        s = "acc>ceiling" if bf is None else (f"{(1-bud/bf)*100:+.1f}%" if bf > 0 else "n/a")
        print(f"{th:>7.2f} {acc:>7.3f} {bud:>8.3f} {rnd:>7.2f} {s:>20}")

    print("\nPareto front (learned):")
    for th, acc, bud, rnd in pareto(pts):
        bf = budget_at_acc(fp, acc)
        s = "acc>fixed-ceiling" if bf is None else (f"{(1-bud/bf)*100:+.1f}% vs fixed b{bf:.3f}" if bf > 0 else "n/a")
        print(f"  theta'={th:.2f}  acc={acc:.3f}  budget={bud:.3f}  rounds={rnd:.2f}   ({s})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    for d in args.run_dirs:
        report(d, args.tau)


if __name__ == "__main__":
    main()
