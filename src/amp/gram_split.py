"""Is the selection rule tuned for the panel we are actually scored on?

The official panel is 15 Gram-negative strains and 5 Gram-positive. Gram-negative killing has to cross an
LPS outer membrane that Gram-positives do not have, so there is a real mechanistic reason the descriptor
dependence could differ. The shipped scorer was tuned on a panel-WEIGHTED label, which is correct on
average but hides a difference in direction if one exists.

If charge, hydrophobicity or length behave differently for Gram-negatives, the envelope and the scorer are
mis-tuned for three quarters of the scored panel, and that is worth knowing before the deadline.

Every correlation here is a partial controlling net charge where the target is not charge itself, because
charge and hydrophobicity correlate -0.729 and raw numbers are confounded.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

import numpy as np

from amp.generate import mean_hydrophobicity, net_charge

AA = "ACDEFGHIKLMNPQRSTVWY"
POTENCY_LOG = math.log10(16.0)

GRAM_NEG = {"a. baumannii", "e. cloacae", "e. coli", "k. pneumoniae", "p. aeruginosa",
            "s. enterica", "s. typhimurium", "salmonella enterica", "salmonella typhimurium"}
GRAM_POS = {"b. subtilis", "s. aureus", "e. faecalis", "e. faecium"}


def zr(v):
    v = np.asarray(v, dtype=float)
    r = np.argsort(np.argsort(v)).astype(float)
    return (r - r.mean()) / (r.std() + 1e-12)


def partial(x, y, controls):
    C = np.column_stack(list(controls) + [np.ones(len(x))])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def auc(x, y, thresh=0.5):
    pos, neg = x[y > thresh], x[y <= thresh]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))).astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grampa", default="data/labelled/grampa.csv")
    ap.add_argument("--min-species", type=int, default=2)
    a = ap.parse_args()

    per = defaultdict(lambda: {"neg": defaultdict(list), "pos": defaultdict(list)})
    with open(a.grampa, newline="") as fh:
        for row in csv.DictReader(fh):
            s = (row.get("sequence") or "").strip().upper()
            if not s or row.get("is_modified") == "True":
                continue
            if not all(c in AA for c in s) or not (8 <= len(s) <= 50):
                continue
            if (row.get("unit") or "").strip() != "uM":
                continue
            b = (row.get("bacterium") or "").strip().lower()
            g = "neg" if b in GRAM_NEG else ("pos" if b in GRAM_POS else None)
            if g is None:
                continue
            try:
                per[s][g][b].append(float(row["value"]))
            except (KeyError, TypeError, ValueError):
                continue

    rows = []
    for s, d in sorted(per.items()):
        rec = {"sequence": s}
        for g in ("neg", "pos"):
            sp = d[g]
            if len(sp) >= a.min_species:
                meds = [statistics.median(v) for v in sp.values()]
                succ = [sum(1 for x in v if x <= POTENCY_LOG) / float(len(v)) for v in sp.values()]
                rec[g + "_mic"] = sum(meds) / len(meds)
                rec[g + "_succ"] = sum(succ) / len(succ)
        if "neg_mic" in rec or "pos_mic" in rec:
            rows.append(rec)

    have_neg = [r for r in rows if "neg_mic" in r]
    have_pos = [r for r in rows if "pos_mic" in r]
    both = [r for r in rows if "neg_mic" in r and "pos_mic" in r]
    print("sequences with >=%d Gram-negative species: %d" % (a.min_species, len(have_neg)))
    print("sequences with >=%d Gram-positive species: %d" % (a.min_species, len(have_pos)))
    print("with both: %d" % len(both))

    print()
    print("=" * 78)
    print("DESCRIPTOR DEPENDENCE BY GRAM CLASS")
    print("=" * 78)
    print("%-22s %10s %10s %10s %10s" % ("", "GN succ", "GN AUC", "GP succ", "GP AUC"))
    print("-" * 66)

    def col(sub, key):
        return np.array([r[key] for r in sub])

    out = {}
    for nm, fn in (("net charge", net_charge),
                   ("mean hydrophobicity", mean_hydrophobicity),
                   ("length", lambda s: float(len(s)))):
        xn = np.array([fn(r["sequence"]) for r in have_neg])
        xp = np.array([fn(r["sequence"]) for r in have_pos])
        yn, yp = col(have_neg, "neg_succ"), col(have_pos, "pos_succ")
        rn = float((zr(xn) * zr(yn)).sum() / len(xn))
        rp = float((zr(xp) * zr(yp)).sum() / len(xp))
        out[nm] = (rn, rp)
        print("%-22s %+10.3f %10.3f %+10.3f %10.3f" % (nm, rn, auc(xn, yn), rp, auc(xp, yp)))

    print()
    print("partials controlling net charge, on log MIC (negative = more potent):")
    for nm, fn in (("mean hydrophobicity", mean_hydrophobicity), ("length", lambda s: float(len(s)))):
        xn = zr([fn(r["sequence"]) for r in have_neg]); qn = zr([net_charge(r["sequence"]) for r in have_neg])
        xp = zr([fn(r["sequence"]) for r in have_pos]); qp = zr([net_charge(r["sequence"]) for r in have_pos])
        print("  %-20s Gram-neg %+.3f      Gram-pos %+.3f"
              % (nm, partial(xn, zr(col(have_neg, "neg_mic")), [qn]),
                 partial(xp, zr(col(have_pos, "pos_mic")), [qp])))

    print()
    print("=" * 78)
    print("IS GRAM-NEGATIVE ACTIVITY THE SAME PROBLEM AS GRAM-POSITIVE?")
    print("=" * 78)
    if len(both) >= 100:
        mn, mp = col(both, "neg_mic"), col(both, "pos_mic")
        print("  correlation of the two log MIC targets: %+.3f (n=%d)"
              % (float((zr(mn) * zr(mp)).sum() / len(mn)), len(both)))
        print("  median log MIC   Gram-neg %.3f (%.1f uM)   Gram-pos %.3f (%.1f uM)"
              % (float(np.median(mn)), 10 ** float(np.median(mn)),
                 float(np.median(mp)), 10 ** float(np.median(mp))))
        sn, sp = col(both, "neg_succ"), col(both, "pos_succ")
        print("  mean success rate at MIC<=16uM:  Gram-neg %.3f   Gram-pos %.3f"
              % (float(sn.mean()), float(sp.mean())))
        print("  sequences active on GP but NOT GN: %d    GN but not GP: %d"
              % (int(((sp > 0.5) & (sn <= 0.5)).sum()), int(((sn > 0.5) & (sp <= 0.5)).sum())))

    print()
    print("=" * 78)
    print("OPTIMAL CHARGE FOR GRAM-NEGATIVES (the envelope cap is currently +5.0)")
    print("=" * 78)
    qn = np.array([net_charge(r["sequence"]) for r in have_neg])
    yn = col(have_neg, "neg_succ")
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 30)):
        m = (qn >= lo) & (qn < hi)
        if m.sum() >= 25:
            print("  charge [%2d,%2d)  n=%-5d  mean GN success %.3f   median log MIC %.3f"
                  % (lo, hi, int(m.sum()), float(yn[m].mean()),
                     float(np.median(col(have_neg, "neg_mic")[m]))))


if __name__ == "__main__":
    main()
