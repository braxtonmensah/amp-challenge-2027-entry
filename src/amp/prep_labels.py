"""Build panel-matched MIC labels from GRAMPA, per PREREG_SELECTION.md.

Every choice here is fixed by PREREG_SELECTION.md, written before any model was fitted:

  * unmodified peptides only, INCLUDING excluding C-terminal amidation, because the competition
    requires linear peptides with free termini and amidation shifts MIC severalfold;
  * only species that are actually on the competition's 20-strain panel, so C. albicans (a fungus)
    and M. luteus / S. epidermidis / B. cereus are dropped;
  * per-species aggregation first, then panel weighting 15:5 Gram-negative:Gram-positive, so the label
    matches the panel composition instead of GRAMPA's own composition (GRAMPA over-represents
    S. aureus, which would tilt a naive median toward Gram-positive activity).

GRAMPA `value` is log10(MIC in µM). The competition potency threshold MIC <= 16 µM is log10 <= 1.2041.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

AA = "ACDEFGHIKLMNPQRSTVWY"
POTENCY_LOG = math.log10(16.0)   # competition threshold: MIC <= 16 uM

# The official panel, from the competition website. 15 Gram-negative strains, 5 Gram-positive.
GRAM_NEG_SPECIES = {
    "a. baumannii", "e. cloacae", "e. coli", "k. pneumoniae", "p. aeruginosa",
    "s. enterica", "s. typhimurium", "salmonella enterica", "salmonella typhimurium",
}
GRAM_POS_SPECIES = {
    "b. subtilis", "s. aureus", "e. faecalis", "e. faecium",
}
N_NEG, N_POS = 15, 5   # panel composition -> label weights


def gram_class(bacterium):
    b = (bacterium or "").strip().lower()
    if b in GRAM_NEG_SPECIES:
        return "neg"
    if b in GRAM_POS_SPECIES:
        return "pos"
    return None


def load(path):
    """-> {sequence: {('neg'|'pos', species): [log10 MIC, ...]}}, plus drop counters."""
    per = defaultdict(lambda: defaultdict(list))
    drop = defaultdict(int)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            s = (row.get("sequence") or "").strip().upper()
            if not s:
                drop["no_sequence"] += 1
                continue
            if row.get("is_modified") == "True":
                drop["modified_or_amidated"] += 1
                continue
            if not all(c in AA for c in s):
                drop["nonstandard_aa"] += 1
                continue
            if not (8 <= len(s) <= 50):
                drop["length"] += 1
                continue
            if (row.get("unit") or "").strip() != "uM":
                drop["unit"] += 1
                continue
            g = gram_class(row.get("bacterium"))
            if g is None:
                drop["off_panel_species"] += 1
                continue
            try:
                v = float(row["value"])
            except (KeyError, TypeError, ValueError):
                drop["unparseable_value"] += 1
                continue
            per[s][(g, (row.get("bacterium") or "").strip().lower())].append(v)
    return per, drop


def labels_for(species_map):
    """Panel-weighted success rate and median log MIC for one sequence.

    Aggregate WITHIN a species first (many measurements per species, unequal across species), then
    across species with the panel's 15:5 weighting.
    """
    by_gram = {"neg": [], "pos": []}
    for (g, _sp), vals in species_map.items():
        med = statistics.median(vals)
        succ = sum(1 for v in vals if v <= POTENCY_LOG) / float(len(vals))
        by_gram[g].append((med, succ))

    have_neg, have_pos = bool(by_gram["neg"]), bool(by_gram["pos"])
    if not (have_neg or have_pos):
        return None

    def wmean(idx):
        parts, wts = [], []
        if have_neg:
            parts.append(sum(x[idx] for x in by_gram["neg"]) / len(by_gram["neg"]))
            wts.append(N_NEG)
        if have_pos:
            parts.append(sum(x[idx] for x in by_gram["pos"]) / len(by_gram["pos"]))
            wts.append(N_POS)
        return sum(p * w for p, w in zip(parts, wts)) / float(sum(wts))

    return {
        "log_mic50": wmean(0),
        "success_rate": wmean(1),
        "n_species": len(by_gram["neg"]) + len(by_gram["pos"]),
        "n_gram_neg_species": len(by_gram["neg"]),
        "n_gram_pos_species": len(by_gram["pos"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grampa", default="data/labelled/grampa.csv")
    ap.add_argument("--out", default="data/labelled/panel_labels.csv")
    ap.add_argument("--min-species", type=int, default=2,
                    help="require measurements on at least this many panel species")
    a = ap.parse_args()

    per, drop = load(a.grampa)
    print("measurement rows dropped:")
    for k, v in sorted(drop.items(), key=lambda x: -x[1]):
        print("  %-22s %6d" % (k, v))
    print("sequences with >=1 panel measurement: %d" % len(per))

    rows = []
    for s, sm in sorted(per.items()):
        lab = labels_for(sm)
        if lab is None or lab["n_species"] < a.min_species:
            continue
        lab["sequence"] = s
        rows.append(lab)

    print("sequences with >=%d panel species: %d" % (a.min_species, len(rows)))
    if len(rows) < 500:
        raise SystemExit("ABORT (pre-registered): fewer than 500 usable labelled sequences.")

    cols = ["sequence", "success_rate", "log_mic50", "n_species",
            "n_gram_neg_species", "n_gram_pos_species"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print("wrote %s" % a.out)

    sr = sorted(r["success_rate"] for r in rows)
    mic = sorted(r["log_mic50"] for r in rows)
    print("\nlabel distribution:")
    print("  success_rate  median %.3f   frac==0 %.3f   frac==1 %.3f"
          % (sr[len(sr) // 2],
             sum(1 for x in sr if x == 0) / len(sr),
             sum(1 for x in sr if x == 1) / len(sr)))
    print("  log_mic50     median %.3f (= %.1f uM)   p10 %.3f   p90 %.3f"
          % (mic[len(mic) // 2], 10 ** mic[len(mic) // 2],
             mic[int(0.1 * len(mic))], mic[int(0.9 * len(mic))]))


if __name__ == "__main__":
    main()
