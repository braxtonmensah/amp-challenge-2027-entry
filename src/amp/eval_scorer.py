"""Run the gates of PREREG_SELECTION_2.md: is the revised scorer better than the shipped one?

Weights are fitted on cluster-disjoint half A and every comparison is made on half B, which is never
used for fitting. Otherwise the revised scorer would be graded on the data that chose it.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import random
import re

import numpy as np

from amp.generate import mean_hydrophobicity, net_charge, score_one

AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20260930


def load_hc50(path):
    """The cleaned haemolysis file has CR-only line endings; parse tolerantly, sanity-range the values."""
    raw = io.open(path, encoding="utf-8", errors="replace", newline="").read()
    cells = [c.strip() for c in ",".join(re.split(r"[\r\n]+", raw)).split(",")]
    hc = {}
    for i, c in enumerate(cells):
        s = c.upper()
        if 5 <= len(s) <= 60 and all(ch in AA for ch in s) and i + 3 < len(cells):
            try:
                v = float(cells[i + 3])
            except ValueError:
                continue
            if 0.0 < v < 4.0:
                hc.setdefault(s, []).append(v)
    return {s: float(np.median(v)) for s, v in hc.items()}


def cluster_split(seqs, thresh=0.6, seed=SEED):
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
    rng = random.Random(seed)
    cl = list(range(len(reps)))
    rng.shuffle(cl)
    half = set(cl[:len(cl) // 2])
    A = [i for i in range(len(seqs)) if assign[i] in half]
    B = [i for i in range(len(seqs)) if assign[i] not in half]
    return A, B, len(reps)


def descriptors(seqs):
    return np.column_stack([
        np.array([net_charge(s) for s in seqs], dtype=float),
        np.array([mean_hydrophobicity(s) for s in seqs], dtype=float),
        np.array([float(len(s)) for s in seqs], dtype=float),
    ])


def zrank(v):
    r = np.argsort(np.argsort(v)).astype(float)
    return (r - r.mean()) / (r.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labelled/panel_labels.csv")
    ap.add_argument("--hemolysis", default="data/labelled/hemolysis_clean.csv")
    ap.add_argument("--topk", type=int, default=100)
    a = ap.parse_args()

    hc = load_hc50(a.hemolysis)
    rows = {}
    with open(a.labels, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["sequence"]] = (float(r["success_rate"]), float(r["log_mic50"]))

    paired = sorted(set(hc) & set(rows))
    seqs = paired
    sr = np.array([rows[s][0] for s in seqs])
    lmic = np.array([rows[s][1] for s in seqs])
    lsw = np.array([hc[s] - rows[s][1] for s in seqs])
    print("paired HC50 + panel MIC sequences: %d" % len(seqs))

    A, B, ncl = cluster_split(seqs)
    print("similarity clusters: %d   fit half A: %d   held-out half B: %d" % (ncl, len(A), len(B)))
    if len(B) < 150:
        raise SystemExit("ABORT (pre-registered): half B has fewer than 150 paired sequences; underpowered.")

    X = descriptors(seqs)
    Xz = np.column_stack([zrank(X[:, j]) for j in range(X.shape[1])])

    # fit on A only
    def fit(idx, target):
        M = np.hstack([Xz[idx], np.ones((len(idx), 1))])
        return np.linalg.solve(M.T @ M + 1.0 * np.eye(M.shape[1]), M.T @ target[idx])

    w_sw = fit(A, zrank(lsw))
    print("\nweights fitted on A, target log SW (standardised):")
    print("  charge %+.3f   hydrophobicity %+.3f   length %+.3f" % (w_sw[0], w_sw[1], w_sw[2]))

    def apply(w, idx):
        return np.hstack([Xz[idx], np.ones((len(idx), 1))]) @ w

    Bi = np.array(B)
    k = min(a.topk, len(Bi))
    scorers = {
        "shipped score_one": np.array([score_one(seqs[i]) for i in Bi]),
        "revised (fitted on A)": apply(w_sw, Bi),
        "net charge alone": np.array([net_charge(seqs[i]) for i in Bi]),
        "charge minus hydrophob.": zrank(np.array([net_charge(seqs[i]) for i in Bi]))
                                   - zrank(np.array([mean_hydrophobicity(seqs[i]) for i in Bi])),
    }

    rng = np.random.default_rng(SEED)
    rand_sw = [float(np.median(lsw[rng.choice(Bi, size=k, replace=False)])) for _ in range(200)]
    rand_sr = [float(np.mean(sr[rng.choice(Bi, size=k, replace=False)])) for _ in range(200)]

    print()
    print("=" * 78)
    print("EVALUATED ON HELD-OUT HALF B ONLY  (top-%d selected by each scorer)" % k)
    print("=" * 78)
    print("%-26s %14s %14s" % ("scorer", "median logSW", "mean succ.rate"))
    print("-" * 58)
    print("%-26s %14.3f %14.3f" % ("random 100", float(np.mean(rand_sw)), float(np.mean(rand_sr))))
    res = {}
    for nm, sc in scorers.items():
        top = np.argsort(-sc)[:k]
        gi = Bi[top]
        res[nm] = (float(np.median(lsw[gi])), float(np.mean(sr[gi])))
        print("%-26s %14.3f %14.3f" % (nm, res[nm][0], res[nm][1]))

    base_sw, base_sr = res["shipped score_one"]
    rev_sw, rev_sr = res["revised (fitted on A)"]
    chg_sw, chg_sr = res["net charge alone"]

    print()
    print("=" * 78)
    gA = (rev_sw - base_sw) >= 0.15
    gB = (rev_sr - base_sr) >= -0.03
    gC = abs(chg_sw - rev_sw) <= 0.05
    print("GATE A  safety window, revised - shipped = %+.3f log10   (need >= +0.15): %s"
          % (rev_sw - base_sw, "PASS" if gA else "FAIL"))
    print("GATE B  success rate,  revised - shipped = %+.3f          (need >= -0.03): %s"
          % (rev_sr - base_sr, "PASS" if gB else "FAIL"))
    print("GATE C  charge-alone within 0.05 of fitted (%+.3f): %s"
          % (chg_sw - rev_sw, "YES, ship the simpler scorer" if gC else "no, keep the fitted weights"))
    print()
    if gA and gB:
        print("VERDICT: H2 SUPPORTED. Replace score_one.")
        if gC:
            print("         H3 also holds: ship the SIMPLE charge-based scorer, discard fitted weights.")
    else:
        print("VERDICT: H2 REFUTED by the pre-registered gates. Ship score_one unchanged and report")
        print("         section 2 of PREREG_SELECTION_2.md as a negative result about our own design.")
    print("=" * 78)


if __name__ == "__main__":
    main()
