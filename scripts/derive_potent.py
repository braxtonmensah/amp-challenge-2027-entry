"""Derive data/potent_amps.fasta from the public GRAMPA MIC dataset.

Source: GRAMPA (Witten & Witten 2019), the aggregated AMP MIC dataset.
  https://github.com/zswitten/Antimicrobial-Peptides  (grampa.csv)

This script is committed so the training corpus is auditable and re-derivable. It is NOT run by the
generation entry point: the derived FASTA is committed directly, so `uv run generate` needs no network
access and no extra dependency.

Selection rule, applied verbatim to reproduce data/potent_amps.fasta:
  * drop modified peptides (`is_modified`) and C-terminally amidated ones
    (`has_cterminal_amidation`), because the competition requires linear peptides with FREE TERMINI;
  * keep 8-50 residues over the 20 standard proteinogenic amino acids;
  * take the MEDIAN of all reported log10(MIC in uM) values per unique sequence;
  * keep sequences whose median MIC <= 16 uM, the competition's own Potency Threshold.

Yields 2,389 sequences from 51,345 rows. Ordering is the insertion order of first appearance, then the
file is written as-is; the generator sorts its own inputs, so downstream determinism does not depend on
this ordering.
"""
import argparse
import csv
import math
import statistics as st
from collections import defaultdict

STD = set("ACDEFGHIKLMNPQRSTVWY")
THRESHOLD_UM = 16.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grampa", required=True, help="path to grampa.csv")
    ap.add_argument("--out", default="data/potent_amps.fasta")
    args = ap.parse_args()

    mic = defaultdict(list)
    rows = 0
    with open(args.grampa, newline="") as fh:
        for r in csv.DictReader(fh):
            rows += 1
            if r["is_modified"].strip().lower() == "true":
                continue
            if r["has_cterminal_amidation"].strip().lower() == "true":
                continue
            s = r["sequence"].strip().upper()
            if not (8 <= len(s) <= 50) or (set(s) - STD):
                continue
            try:
                mic[s].append(float(r["value"]))
            except ValueError:
                continue

    thr = math.log10(THRESHOLD_UM)
    potent = [s for s, v in mic.items() if st.median(v) <= thr]
    with open(args.out, "w", newline="\n") as out:
        for i, s in enumerate(potent):
            out.write(f">potent_amps_{i}\n{s}\n")
    print(f"rows={rows}  unique eligible={len(mic)}  median MIC<={THRESHOLD_UM}uM -> {len(potent)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
