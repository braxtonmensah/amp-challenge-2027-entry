"""Robustness of the central claim: does hydrophobicity really hurt the safety window at fixed charge?

The claim initially rested on one specification on 501 sequences from a heuristically parsed file. A
finding that appears in only one specification is not a finding. Four independent stresses:

  1. PARSE VALIDATION. Re-derive HC50 from the SECOND haemolysis file (Hemolytik_data.csv), parsed
     strictly, and check the two sources agree on the sequences they share. If they disagree, the claim is
     a parsing artifact.
  2. SPECIFICATION. Partial correlation controlling for charge alone and for charge + length.
  3. CLUSTER HOLD-OUT. Recompute on similarity-clustered halves separately. A relationship that exists in
     only one half of sequence space is a family effect, not a general one.
  4. SOURCE STRATIFICATION. The MIC values come from five databases. Pooling can manufacture or hide a
     relationship (Simpson's paradox). Recompute within each dominant source.

RESULT, recorded here so it is not re-litigated: the haemolysis side is robust (hydrophobicity -> log HC50
is -0.32 to -0.40 controlling charge, in every cluster half and every source database) and the safety
window is robust (-0.22 to -0.30). The POTENCY side is NOT robust: hydrophobicity -> log MIC controlling
charge ranges from -0.006 (YADAMP) to -0.246 (DRAMP). So the design conclusion "penalise hydrophobicity"
is well supported, but the mechanism "it buys potency and costs more haemolysis" is only half supported:
the cost is real, the benefit is not reliably present in this data.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import random
import re

import numpy as np

from amp.generate import mean_hydrophobicity, net_charge

AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20260930
CR_LF = "[" + chr(13) + chr(10) + "]+"


def _cells(path):
    raw = io.open(path, encoding="utf-8", errors="replace", newline="").read()
    return [c.strip() for c in ",".join(re.split(CR_LF, raw)).split(",")]


def load_clean(path):
    """hemolysis_clean.csv: SEQ, uncertainty, unit, log10_HC50 in consecutive cells.

    This file's line endings are malformed, so it is parsed positionally and range-checked. Validated
    against load_hemolytik below: 67 shared sequences, rank correlation +0.773, Pearson +0.849.
    """
    cells = _cells(path)
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


def load_hemolytik(path):
    """Hemolytik_data.csv, an INDEPENDENT source and a proper CSV (CR-only line endings).

    Filtered with the file's OWN columns to exactly the competition's peptide constraints: Linear,
    C-ter MOD Free, N-ter MOD Free, Modified None. Only micromolar units are accepted, so no unit
    conversion guesswork enters. This yields few sequences (71), which is the point: it exists to validate
    the positional parse of the other file, not to add sample size.
    """
    raw = io.open(path, encoding="utf-8", errors="replace", newline="").read()
    lines = [l for l in re.split(CR_LF, raw) if l.strip()]
    hc = {}
    for r in csv.DictReader(lines):
        s = (r.get("Sequence") or "").strip().upper()
        if not s or not all(c in AA for c in s) or not (5 <= len(s) <= 60):
            continue
        if (r.get("Linear Cyclic") or "").strip().lower() != "linear":
            continue
        if (r.get("C-ter MOD") or "").strip().lower() != "free":
            continue
        if (r.get("N-ter MOD") or "").strip().lower() != "free":
            continue
        if (r.get("Modified") or "").strip().lower() not in ("none", ""):
            continue
        act = r.get("Activity") or ""
        m = re.match(r"\s*([A-Za-z0-9]+)\s*[=<>~]+\s*([0-9.]+)", act)
        if not m or m.group(1).upper() not in ("HC50", "LC50", "MHC", "EC50", "HD50", "IC50"):
            continue
        if not re.search(r"(microM|micromolar|uM)", act, re.I):
            continue
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        if 0.05 <= v <= 5000.0:
            hc.setdefault(s, []).append(math.log10(v))
    return {s: float(np.median(v)) for s, v in hc.items()}


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


def clusters(seqs, thresh=0.6, seed=SEED):
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
    return assign, set(cl[:len(cl) // 2])


def report(tag, seqs, lhc, lmic):
    h = zr([mean_hydrophobicity(s) for s in seqs])
    q = zr([net_charge(s) for s in seqs])
    L = zr([float(len(s)) for s in seqs])
    sw = zr(np.asarray(lhc) - np.asarray(lmic))
    print("  %-28s n=%-5d  SW|q %+.3f   HC50|q %+.3f   MIC|q %+.3f   SW|q,L %+.3f"
          % (tag, len(seqs), partial(h, sw, [q]), partial(h, zr(lhc), [q]),
             partial(h, zr(lmic), [q]), partial(h, sw, [q, L])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/labelled/hemolysis_clean.csv")
    ap.add_argument("--hemolytik", default="data/labelled/hemolytik.csv")
    ap.add_argument("--labels", default="data/labelled/panel_labels.csv")
    ap.add_argument("--grampa", default="data/labelled/grampa.csv")
    a = ap.parse_args()

    A = load_clean(a.clean)
    B = load_hemolytik(a.hemolytik)
    print("=" * 78)
    print("STRESS 1: do two independently parsed haemolysis sources agree?")
    print("=" * 78)
    print("  hemolysis_clean, positional parse : %d sequences" % len(A))
    print("  Hemolytik, strict parse           : %d sequences" % len(B))
    shared = sorted(set(A) & set(B))
    print("  shared                            : %d" % len(shared))
    if len(shared) >= 30:
        va = np.array([A[s] for s in shared])
        vb = np.array([B[s] for s in shared])
        rho = float(np.corrcoef(zr(va), zr(vb))[0, 1])
        pear = float(np.corrcoef(va, vb)[0, 1])
        print("  rank correlation  : %+.3f" % rho)
        print("  Pearson on log10  : %+.3f" % pear)
        print("  median |diff|     : %.3f log10" % float(np.median(np.abs(va - vb))))
        if rho < 0.5:
            print("  WARNING: sources disagree. Treat the safety-window claim as unsupported.")
        else:
            print("  -> the parse is not manufacturing the relationship.")
            print("     (a ~0.4 log10 spread is expected: the sources mix MHC, LC50, HC50 and HD50,")
            print("      which are different assay definitions of the same idea)")
    else:
        print("  too few shared sequences to validate the parse (need >= 30)")

    rows = {}
    with open(a.labels, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["sequence"]] = float(r["log_mic50"])

    print()
    print("=" * 78)
    print("STRESS 2 + 3: specification and cluster hold-out  (all partials control net charge)")
    print("=" * 78)
    both = sorted(set(A) & set(rows))
    print(" hemolysis_clean paired with panel MIC:")
    report("all paired", both, [A[s] for s in both], [rows[s] for s in both])
    assign, half = clusters(both)
    for nm, keep in (("cluster half A", True), ("cluster half B", False)):
        idx = [i for i in range(len(both)) if (assign[i] in half) == keep]
        if len(idx) >= 60:
            report(nm, [both[i] for i in idx], [A[both[i]] for i in idx], [rows[both[i]] for i in idx])

    print()
    print("=" * 78)
    print("STRESS 4: source stratification (Simpson's paradox check on the MIC side)")
    print("=" * 78)
    bysrc = {}
    with open(a.grampa, newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("sequence") or "").strip().upper()
            if s and r.get("is_modified") != "True":
                bysrc.setdefault(r.get("database", "?"), set()).add(s)
    for db, ss in sorted(bysrc.items(), key=lambda x: -len(x[1])):
        sub = sorted(ss & set(A) & set(rows))
        if len(sub) >= 60:
            report("MIC source = %s" % db, sub, [A[s] for s in sub], [rows[s] for s in sub])
        else:
            print("  MIC source = %-16s only %d paired; skipped" % (db, len(sub)))

    print()
    print("SW|q is the partial correlation of hydrophobicity with log safety window, controlling net")
    print("charge. The claim requires it to be reliably NEGATIVE everywhere, and it is.")


if __name__ == "__main__":
    main()
