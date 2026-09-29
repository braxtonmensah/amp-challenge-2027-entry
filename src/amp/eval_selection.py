"""Run the two pre-registered gates from PREREG_SELECTION.md.

Gate 1 (usability)  : grouped-CV Spearman must beat its shuffled-label null by >= 0.10.
Gate 2 (decisive)   : out-of-fold, the trained model's top-100 must beat the shipped biophysical
                      score_one's top-100 by >= 0.05 absolute measured panel success rate.

If Gate 2 fails, H1 is refuted and the biophysical selection ships unchanged. A tie is a failure.

Clusters (single-linkage, Levenshtein ratio >= 0.6) are held out whole, so near-duplicate AMP families
cannot straddle train and test. That leak is the standard reason published AMP predictors do not
reproduce, and it is why this script exists at all.
"""
from __future__ import annotations

import argparse
import csv
import math
import random

import numpy as np

from amp.generate import score_one
from amp.predictor import features

SEED = 20260930


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def cluster_by_identity(seqs, thresh=0.6):
    import Levenshtein
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), seqs[i]))
    reps, assign = [], [-1] * len(seqs)
    for i in order:
        for ci, r in enumerate(reps):
            if Levenshtein.ratio(seqs[i], seqs[r]) >= thresh:
                assign[i] = ci
                break
        else:
            reps.append(i)
            assign[i] = len(reps) - 1
    return assign, len(reps)


def ridge_fit(X, y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)


def ridge_pred(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def out_of_fold(X, y, assign, ncl, folds, lam, seed, shuffle_labels=False):
    yy = np.array(y, dtype=float)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        yy = rng.permutation(yy)
    rng = random.Random(seed)
    cl = list(range(ncl))
    rng.shuffle(cl)
    fold_of = {c: i % folds for i, c in enumerate(cl)}
    preds = np.full(len(y), np.nan)
    for f in range(folds):
        te = [i for i in range(len(y)) if fold_of[assign[i]] == f]
        tr = [i for i in range(len(y)) if fold_of[assign[i]] != f]
        if not te or not tr:
            continue
        preds[te] = ridge_pred(ridge_fit(X[tr], yy[tr], lam), X[te])
    return preds, yy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labelled/panel_labels.csv")
    ap.add_argument("--label", default="success_rate")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--topk", type=int, default=100)
    a = ap.parse_args()

    # Direction matters and getting it wrong manufactures a fake win: for success_rate higher is
    # better, for log_mic50 LOWER is better (lower MIC = more potent). Sorting descending on
    # log_mic50 selects the WEAKEST peptides and would score as a pass under a higher-is-better gate.
    lower_is_better = a.label in ("log_mic50", "log_mic", "mic", "log10_mic")
    sign = -1.0 if lower_is_better else 1.0
    print("target direction: %s is better" % ("LOWER" if lower_is_better else "HIGHER"))

    seqs, y = [], []
    with open(a.data, newline="") as fh:
        for row in csv.DictReader(fh):
            seqs.append(row["sequence"])
            y.append(float(row[a.label]))
    print("labelled sequences: %d   target: %s" % (len(seqs), a.label))
    if len(seqs) < 500:
        raise SystemExit("ABORT (pre-registered): fewer than 500 usable labelled sequences.")

    X = np.vstack([features(s) for s in seqs])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    assign, ncl = cluster_by_identity(seqs)
    print("similarity clusters held out whole: %d (mean %.1f seqs/cluster)"
          % (ncl, len(seqs) / float(ncl)))

    preds, yy = out_of_fold(X, y, assign, ncl, a.folds, a.lam, SEED)
    null, ynull = out_of_fold(X, y, assign, ncl, a.folds, a.lam, SEED, shuffle_labels=True)

    ok = ~np.isnan(preds)
    rho = spearman(preds[ok], yy[ok])
    okn = ~np.isnan(null)
    rho_null = spearman(null[okn], ynull[okn])

    print()
    print("=" * 72)
    print("GATE 1 (usability): grouped-CV Spearman vs shuffled-label null")
    print("=" * 72)
    print("  trained model        : %+.3f" % rho)
    print("  shuffled-label null  : %+.3f" % rho_null)
    print("  margin               : %+.3f   (pre-registered requirement: >= 0.10)" % (abs(rho) - abs(rho_null)))
    gate1 = (abs(rho) - abs(rho_null)) >= 0.10
    print("  GATE 1: %s" % ("PASS" if gate1 else "FAIL"))

    # Gate 2: out-of-fold selection head-to-head on MEASURED activity.
    bio = np.array([score_one(s) for s in seqs], dtype=float)
    ymeas = np.array(y, dtype=float)
    idx = np.where(ok)[0]
    k = min(a.topk, len(idx))

    # sign flips the ordering when lower label values are better
    top_model = idx[np.argsort(-sign * preds[idx])[:k]]
    top_bio = idx[np.argsort(-bio[idx])[:k]]   # score_one is always higher-is-better
    rng = np.random.default_rng(SEED)
    rand_means = [float(ymeas[rng.choice(idx, size=k, replace=False)].mean()) for _ in range(200)]

    m_model = float(ymeas[top_model].mean())
    m_bio = float(ymeas[top_bio].mean())
    m_rand = float(np.mean(rand_means))

    print()
    print("=" * 72)
    print("GATE 2 (decisive): measured panel success rate of the selected top-%d" % k)
    print("=" * 72)
    print("  random %d from the labelled pool : %.3f  (spread %.3f-%.3f over 200 draws)"
          % (k, m_rand, min(rand_means), max(rand_means)))
    print("  shipped biophysical score_one    : %.3f" % m_bio)
    print("  trained model (out-of-fold)      : %.3f" % m_model)
    improvement = sign * (m_model - m_bio)
    print("  improvement (direction-corrected): %+.3f   (pre-registered requirement: >= 0.05)"
          % improvement)
    gate2 = improvement >= 0.05
    print("  GATE 2: %s" % ("PASS" if gate2 else "FAIL"))

    print()
    print("  measured median %s of each selected set:" % a.label)
    print("    biophysical   %.3f" % float(np.median(ymeas[top_bio])))
    print("    trained model %.3f" % float(np.median(ymeas[top_model])))

    print()
    print("=" * 72)
    if gate1 and gate2:
        print("VERDICT: H1 SUPPORTED. Use the trained model to select the 100 from the library.")
    else:
        print("VERDICT: H1 REFUTED by the pre-registered gates.")
        print("Ship the biophysical selection unchanged. Do not substitute the model.")
        if not gate1:
            print("  - Gate 1 failed: the model is not separable from its own shuffled-label null.")
        if not gate2:
            print("  - Gate 2 failed: it does not select measurably better peptides than the")
            print("    biophysical score, which is the only comparison that matters.")
    print("=" * 72)


if __name__ == "__main__":
    main()
