"""Activity / haemolysis predictor harness for reranking the AMP library.

WHY A PREDICTOR AND NOT A BIGGER GENERATOR. Compliance alone very likely earns the assay slot
(22 teams entered, 20 wet-lab slots), and the Phase-1 metric is a tiebreak. The award categories are
decided in Phase 2 by measured MIC across a 20-strain panel and HC50 haemolysis, on 25 peptides drawn
from our ranked top list. So the quantity that decides the outcome is the RANKING, not the generator:
the generator only has to supply plausible diverse candidates, which an order-2 Markov chain fitted to
39,448 real antibacterials already does. Replacing an asserted biophysical score with a model trained
on measured MIC acts directly on which 25 peptides get synthesised.

This module is deliberately dependency-light: descriptor features plus a linear/logistic model with
proper grouped cross-validation, no deep learning. On peptide activity tasks with a few thousand
labelled examples, descriptor models are competitive and they cannot silently overfit the way an
under-regularised network can at this sample size. If the data turns out to be large enough to justify
something heavier, swap the estimator, not the evaluation.

THE EVALUATION IS THE POINT. AMP datasets are full of near-duplicate sequences (families, single-point
variants, the same peptide from several papers). Random k-fold on that leaks and flatters a model by a
wide margin, which is exactly how published AMP predictors end up unreproducible. So:

  * sequences are clustered by similarity and whole clusters are held out, never individual sequences;
  * a shuffled-label null is run at the same settings and reported beside the real number;
  * the metric reported is the one that matters for our use, ranking quality (Spearman / AUC), not
    accuracy at some threshold we never use.

Usage (once a labelled file exists):
    py -3.11 -m amp.predictor --data mic.csv --seq-col sequence --label-col log_mic
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"

EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48, "H": -0.40,
    "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85,
    "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
HELIX = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13, "G": 0.57, "H": 1.00,
    "I": 1.08, "K": 1.16, "L": 1.21, "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11,
    "R": 0.98, "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}
POS = {"K": 1.0, "R": 1.0, "H": 0.1}
NEG = {"D": -1.0, "E": -1.0}


def moment(s, deg=100.0):
    ang = np.deg2rad(deg) * np.arange(len(s))
    h = np.array([EISENBERG[c] for c in s])
    return float(np.hypot((h * np.sin(ang)).sum(), (h * np.cos(ang)).sum()) / len(s))


def features(s):
    """Descriptor vector: composition, global physicochemistry, and amphipathicity."""
    n = float(len(s))
    comp = [s.count(a) / n for a in AA]
    q = sum(POS.get(c, 0.0) for c in s) + sum(NEG.get(c, 0.0) for c in s)
    h = sum(EISENBERG[c] for c in s) / n
    hel = sum(HELIX[c] for c in s) / n
    arom = sum(s.count(a) for a in "FWY") / n
    tiny = sum(s.count(a) for a in "AGS") / n
    return np.array(comp + [
        n, math.log(n), q, q / n, h, moment(s), moment(s, 180.0), hel, arom, tiny,
        sum(s.count(a) for a in "KR") / n,
        sum(s.count(a) for a in "DE") / n,
        sum(s.count(a) for a in "ILVFM") / n,
        s.count("C") / n, s.count("P") / n, s.count("G") / n,
    ], dtype=float)


def cluster_by_identity(seqs, thresh=0.6):
    """Greedy single-linkage clusters by Levenshtein ratio. Whole clusters are held out together.

    This is the guard against the standard AMP-dataset leak: near-duplicate sequences split across
    train and test make any model look good.
    """
    import Levenshtein
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), seqs[i]))
    reps, assign = [], [-1] * len(seqs)
    for i in order:
        placed = False
        for ci, r in enumerate(reps):
            if Levenshtein.ratio(seqs[i], seqs[r]) >= thresh:
                assign[i] = ci
                placed = True
                break
        if not placed:
            reps.append(i)
            assign[i] = len(reps) - 1
    return assign, len(reps)


def ridge_fit(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def grouped_cv(seqs, y, folds=5, lam=1.0, seed=25, shuffle_labels=False):
    X = np.vstack([features(s) for s in seqs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    yy = np.array(y, dtype=float)
    if shuffle_labels:
        rng = random.Random(seed)
        yy = yy.copy()
        rng.shuffle(yy)
    assign, ncl = cluster_by_identity(seqs)
    rng = random.Random(seed)
    cl = list(range(ncl))
    rng.shuffle(cl)
    fold_of = {c: i % folds for i, c in enumerate(cl)}
    preds = np.zeros(len(seqs))
    for f in range(folds):
        te = [i for i in range(len(seqs)) if fold_of[assign[i]] == f]
        tr = [i for i in range(len(seqs)) if fold_of[assign[i]] != f]
        if not te or not tr:
            continue
        w = ridge_fit(X[tr], yy[tr], lam)
        preds[te] = ridge_pred(w, X[te])
    return spearman(preds, yy), ncl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()

    seqs, ys = [], []
    with open(a.data, newline="") as fh:
        for row in csv.DictReader(fh):
            s = (row.get(a.seq_col) or "").strip().upper()
            v = row.get(a.label_col)
            if not s or v in (None, ""):
                continue
            if not all(c in AA for c in s) or not (5 <= len(s) <= 60):
                continue
            try:
                ys.append(float(v))
            except ValueError:
                continue
            seqs.append(s)
    print("usable labelled sequences: %d" % len(seqs))
    if len(seqs) < 200:
        raise SystemExit("ABORT: too few labelled sequences to trust any model here.")

    rho, ncl = grouped_cv(seqs, ys, folds=a.folds, lam=a.lam)
    null, _ = grouped_cv(seqs, ys, folds=a.folds, lam=a.lam, shuffle_labels=True)
    print("similarity clusters (held out whole): %d" % ncl)
    print("grouped CV Spearman        : %+.3f" % rho)
    print("shuffled-label null        : %+.3f" % null)
    print()
    if abs(rho) - abs(null) < 0.10:
        print("VERDICT: the model does NOT clear its own null by a useful margin.")
        print("Do not rerank on it. Keep the biophysical score.")
    else:
        print("VERDICT: usable signal. Reranking on this beats an asserted score.")


if __name__ == "__main__":
    main()
