#!/usr/bin/env python3
"""牌2: single-shot budget predictor (no rounds, no per-task labels at deploy).

Trains a lightweight classifier P(solved | Pass-1 features, r) and turns it into a
single-shot policy: compute features once, pick the smallest budget r whose predicted
success prob >= theta, transmit once. The decisive question is GENERALIZATION:

  - WITHIN (upper bound): per-task GroupKFold by sample (train & test same task).
  - LODO  (the real bar): leave-one-task-out -> predict an UNSEEN task with ZERO
    labels from it. Only a positive LODO result justifies the predictor (KVComm
    needs no per-task labels either).

Inputs (joined by idx):
  features  snapshots/<task>/features/**/per_sample_feat.jsonl
  labels    snapshots/<task>/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl  (score per budget)

Usage:
  python scripts/learn_budget_predictor.py --tasks hotpotqa musique twowikimqa --tau 0.5
"""
import argparse
import glob
import json
import os
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

R_RE = re.compile(r"probe_recv_w16_r([0-9.]+)")


def load_scores(task):
    """-> {r: {idx: score}} from probe runs."""
    out = {}
    for path in glob.glob(f"snapshots/{task}/mtc_receiver/probe_recv_w16_r*/per_sample.jsonl"):
        m = R_RE.search(os.path.dirname(path))
        if not m:
            continue
        r = float(m.group(1))
        d = {}
        for line in open(path):
            row = json.loads(line)
            if "_meta" in row:
                continue
            d[row["idx"]] = row["score"]
        out[r] = d
    return out


def load_feats(task):
    """-> ({idx: [feat vector]}, feature_names)."""
    paths = glob.glob(f"snapshots/{task}/features/**/per_sample_feat.jsonl", recursive=True)
    feats, names = {}, None
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            if "_meta" in row:
                names = row["_meta"].get("features")
                continue
            if names is None:
                names = [k for k in row if k not in ("idx", "id")]
            feats[row["idx"]] = [row[k] for k in names]
        if feats:
            break
    return feats, names


def build(tasks, tau):
    """Pooled rows over (task, idx, r). Returns arrays + bookkeeping."""
    X, y, task_of, idx_of, r_of = [], [], [], [], []
    scores_by_task, feat_names = {}, None
    for t in tasks:
        scores = load_scores(t)
        feats, names = load_feats(t)
        if not scores or not feats:
            print(f"  [warn] {t}: scores={len(scores)} feats={len(feats)} -> skipped")
            continue
        feat_names = feat_names or names
        scores_by_task[t] = scores
        ladder = sorted(scores.keys())
        common = set(feats) & set.intersection(*(set(scores[r]) for r in ladder))
        for idx in sorted(common):
            for r in ladder:
                X.append(feats[idx] + [r])
                y.append(1 if scores[r][idx] >= tau else 0)
                task_of.append(t); idx_of.append(idx); r_of.append(r)
    return (np.array(X, float), np.array(y, int), np.array(task_of),
            np.array(idx_of), np.array(r_of), scores_by_task, (feat_names or []) + ["r"])


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def within_preds(X, y, task_of, idx_of):
    """Per-task GroupKFold-by-sample OOF probabilities (upper bound)."""
    from sklearn.model_selection import GroupKFold
    prob = np.zeros(len(y))
    for t in np.unique(task_of):
        m = task_of == t
        Xi, yi, gi = X[m], y[m], idx_of[m]
        n_groups = len(np.unique(gi))
        k = min(5, n_groups)
        if k < 2 or len(np.unique(yi)) < 2:
            prob[m] = yi.mean()
            continue
        pi = np.zeros(len(yi))
        for tr, te in GroupKFold(n_splits=k).split(Xi, yi, gi):
            c = clf(); c.fit(Xi[tr], yi[tr]); pi[te] = c.predict_proba(Xi[te])[:, 1]
        prob[m] = pi
    return prob


def lodo_preds(X, y, task_of):
    """Leave-one-task-out probabilities (the generalization bar)."""
    prob = np.zeros(len(y))
    for t in np.unique(task_of):
        tr, te = task_of != t, task_of == t
        if len(np.unique(y[tr])) < 2:
            prob[te] = y[tr].mean()
            continue
        c = clf(); c.fit(X[tr], y[tr]); prob[te] = c.predict_proba(X[te])[:, 1]
    return prob


def fixed_curve(scores, common, tau):
    out = []
    for r in sorted(scores.keys()):
        acc = sum(1 for i in common if scores[r][i] >= tau) / len(common)
        out.append((r, r, acc))  # (r, budget=r, acc)
    return out


def budget_at_acc(fp, acc):
    pts = sorted(fp, key=lambda x: x[2])
    if acc <= pts[0][2]:
        return pts[0][1]
    if acc > pts[-1][2] + 1e-9:
        return None
    for (r0, b0, a0), (r1, b1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0:
            return b0 + (acc - a0) / (a1 - a0) * (b1 - b0)
    return pts[-1][1]


def oracle(scores, common, tau):
    ladder = sorted(scores.keys()); bud = 0.0; solved = 0
    for i in common:
        hit = next((r for r in ladder if scores[r][i] >= tau), None)
        bud += hit if hit is not None else ladder[-1]
        solved += 1 if hit is not None else 0
    return solved / len(common), bud / len(common)


def simulate(prob_map, scores, common, ladder, theta, tau):
    bud = 0.0; solved = 0
    for i in common:
        chosen = ladder[-1]
        for r in ladder:
            if prob_map[(i, r)] >= theta:
                chosen = r; break
        bud += chosen
        solved += 1 if scores[chosen][i] >= tau else 0
    return solved / len(common), bud / len(common)


def eval_task(t, scores, prob_map, common, tau, label):
    ladder = sorted(scores.keys())
    fp = fixed_curve(scores, common, tau)
    pts = []
    for th in np.linspace(0.05, 0.95, 19):
        acc, bud = simulate(prob_map, scores, common, ladder, th, tau)
        pts.append((round(float(th), 3), acc, bud))
    # pareto by budget
    front = []
    for p in sorted(pts, key=lambda x: x[2]):
        if not front or p[1] > front[-1][1] + 1e-9:
            front.append(p)
    print(f"\n  -- {label} policy on {t} --")
    best = None
    for th, acc, bud in front:
        bf = budget_at_acc(fp, acc)
        if bf is None:
            s = "acc>ceiling"
        elif bf > 0:
            sv = (1 - bud / bf) * 100
            s = f"{sv:+.1f}% vs fixed b{bf:.3f}"
            if acc >= 0.6 * max(a for _, _, a in fp) and (best is None or sv > best):
                best = sv
        else:
            s = "n/a"
        print(f"    theta={th:.2f}  acc={acc:.3f}  budget={bud:.3f}   ({s})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    tau = args.tau

    X, y, task_of, idx_of, r_of, scores_by_task, feat_names = build(args.tasks, args.tau)
    if len(X) == 0:
        print("no data (need both features and probe scores). run run_features.sh first.")
        return
    print(f"rows={len(X)}  tasks={list(scores_by_task)}  base-rate solved={y.mean():.3f}")

    p_within = within_preds(X, y, task_of, idx_of)
    p_lodo = lodo_preds(X, y, task_of)
    print(f"\nAUC P(solved):  WITHIN={roc_auc_score(y, p_within):.3f}   "
          f"LODO={roc_auc_score(y, p_lodo):.3f}   (0.5=useless)")

    # global feature importance (fit on all)
    c = clf(); c.fit(X, y)
    coef = c.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coef))
    print("top features (|coef|): " +
          ", ".join(f"{feat_names[i]}:{coef[i]:+.2f}" for i in order[:8]))

    for t in args.tasks:
        if t not in scores_by_task:
            continue
        scores = scores_by_task[t]
        ladder = sorted(scores.keys())
        common = sorted(set.intersection(*(set(scores[r]) for r in ladder)) &
                        set(idx_of[task_of == t].tolist()))
        oacc, obud = oracle(scores, common, tau)
        fp = fixed_curve(scores, common, tau)
        bestfix = max(a for _, _, a in fp)
        print(f"\n=== {t} | N={len(common)} | tau={tau} ===")
        print("  fixed-r: " + ", ".join(f"r{r}:acc{a:.3f}" for r, _, a in fp) +
              f"   | ORACLE acc{oacc:.3f}@b{obud:.3f}")
        mask = task_of == t
        pw = {(idx_of[i], r_of[i]): p_within[i] for i in np.where(mask)[0]}
        pl = {(idx_of[i], r_of[i]): p_lodo[i] for i in np.where(mask)[0]}
        eval_task(t, scores, pw, common, tau, "WITHIN (upper bound)")
        eval_task(t, scores, pl, common, tau, "LODO (generalization)")


if __name__ == "__main__":
    main()
