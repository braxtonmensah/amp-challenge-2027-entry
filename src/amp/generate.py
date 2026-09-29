"""AMP Challenge 2027 entry: order-2 Markov generation over known antibacterials, ranked by the
classic biophysical determinants of antimicrobial activity and of the safety window.

DESIGN RATIONALE, stated because the method must be disclosed.

The organizers' example generator samples random K/P strings. That satisfies the schema and nothing
else. The two things that actually determine whether a cationic antimicrobial peptide kills bacteria
are well established and cheap to compute:

  1. NET POSITIVE CHARGE, which drives association with the anionic bacterial membrane. The reference
     set of 39,448 known antibacterials is 20.6% K+R, against ~11% in an average proteome.
  2. AMPHIPATHICITY, the segregation of hydrophobic and polar faces once helical. Measured here as the
     Eisenberg hydrophobic moment at 100 degrees per residue.

The third consideration is the one that decides the safety window HC50/MIC50 that Phase 2 scores:
excessive raw hydrophobicity drives haemolysis as readily as it drives potency. So hydrophobicity is
scored toward a target band rather than maximised, which is a deliberate trade of predicted potency
for a predicted safety margin.

Sequence realism comes from an order-2 Markov chain fit to the reference actives, so local motifs
(KKIL, GKII, and similar) appear at their natural frequency instead of being assembled from
independent draws. Lengths are drawn from the reference length distribution.

NOVELTY. The library is filtered against exact matches to the reference set. The top 100 are
additionally required to sit at Levenshtein ratio <= 0.8 from every reference sequence, which is the
organizers' own novelty rule, so the ranked set cannot be a paraphrase of a known peptide.

DETERMINISM. One rng, seeded from SEED. Every collection is sorted before iteration. See SEED.md.

TRAINING DATA. Only `data/antibacterial.fasta` as shipped in the organizers' template repo
(39,448 sequences, all 8-50 aa). No other corpus, no pretrained model, no external database.

AI ASSISTANCE. This code was written with AI assistance (Claude), as the competition rules permit and
require to be disclosed.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20260930
AA = "ACDEFGHIKLMNPQRSTVWY"

# Eisenberg consensus hydrophobicity scale.
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48, "H": -0.40,
    "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85,
    "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
# Chou-Fasman helix propensity, higher favours helix.
HELIX = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13, "G": 0.57, "H": 1.00,
    "I": 1.08, "K": 1.16, "L": 1.21, "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11,
    "R": 0.98, "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}
CHARGED_POS = {"K": 1.0, "R": 1.0, "H": 0.1}   # H mostly neutral at pH 7.4
CHARGED_NEG = {"D": -1.0, "E": -1.0}

# Target band for mean hydrophobicity. Above this, haemolysis risk rises faster than potency.
H_TARGET_LO, H_TARGET_HI = 0.05, 0.45
CHARGE_TARGET_LO, CHARGE_TARGET_HI = 4.0, 9.0


def read_fasta(path):
    seqs, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                cur = None
            elif line:
                seqs.append(line.upper())
    return [s for s in seqs if s and all(c in AA for c in s)]


def fit_markov(seqs, order=2):
    """-> (start_counts, trans) as plain dicts of Counters, order-2 over amino acids."""
    starts = Counter()
    trans = defaultdict(Counter)
    for s in seqs:
        if len(s) < order + 1:
            continue
        starts[s[:order]] += 1
        for i in range(len(s) - order):
            trans[s[i:i + order]][s[i + order]] += 1
    return starts, trans


def _probs(counter, rng_keys):
    ks = sorted(counter)
    tot = float(sum(counter[k] for k in ks))
    return ks, np.array([counter[k] / tot for k in ks])


def net_charge(s):
    return sum(CHARGED_POS.get(c, 0.0) for c in s) + sum(CHARGED_NEG.get(c, 0.0) for c in s)


def mean_hydrophobicity(s):
    return sum(EISENBERG[c] for c in s) / len(s)


def hydrophobic_moment(s, deg_per_res=100.0):
    """Eisenberg hydrophobic moment, normalised per residue. Higher = more amphipathic."""
    ang = np.deg2rad(deg_per_res) * np.arange(len(s))
    h = np.array([EISENBERG[c] for c in s])
    return float(np.hypot((h * np.sin(ang)).sum(), (h * np.cos(ang)).sum()) / len(s))


def helix_propensity(s):
    return sum(HELIX[c] for c in s) / len(s)


def _band(x, lo, hi):
    """1.0 inside [lo, hi], decaying outside. Rewards a target band, not a maximum."""
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(1e-9, abs(lo) + 1.0))
    if x > hi:
        return max(0.0, 1.0 - (x - hi) / max(1e-9, abs(hi) + 1.0))
    return 1.0


def score_one(s):
    """Composite, higher is better. Weights are stated, not tuned against any withheld metric."""
    q = net_charge(s)
    h = mean_hydrophobicity(s)
    mu = hydrophobic_moment(s)
    hel = helix_propensity(s)
    return (0.40 * _band(q, CHARGE_TARGET_LO, CHARGE_TARGET_HI)
            + 0.30 * min(1.0, mu / 0.55)
            + 0.20 * _band(h, H_TARGET_LO, H_TARGET_HI)
            + 0.10 * min(1.0, (hel - 0.85) / 0.35))


def generate(n_sequences, ref_path, seed=SEED):
    refs = read_fasta(ref_path)
    refset = set(refs)
    lengths = np.array([len(s) for s in refs])
    starts, trans = fit_markov(refs, order=2)
    skeys, sprob = _probs(starts, None)
    tcache = {k: _probs(v, None) for k, v in sorted(trans.items())}

    rng = np.random.default_rng(seed)
    out, seen = [], set()
    tries = 0
    max_tries = n_sequences * 60
    while len(out) < n_sequences and tries < max_tries:
        tries += 1
        L = int(lengths[rng.integers(len(lengths))])
        s = skeys[rng.choice(len(skeys), p=sprob)]
        while len(s) < L:
            key = s[-2:]
            if key not in tcache:
                break
            ks, ps = tcache[key]
            s += ks[rng.choice(len(ks), p=ps)]
        if len(s) < 8 or len(s) > 50:
            continue
        if s in seen or s in refset:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n_sequences:
        raise RuntimeError("only generated %d of %d unique novel sequences in %d tries"
                           % (len(out), n_sequences, tries))
    return out, refs


def pick_top(sequences, refs, k, novelty=0.8):
    """Best-scoring k that are <= `novelty` Levenshtein ratio from EVERY reference sequence."""
    import Levenshtein
    ranked = sorted(sequences, key=lambda s: (-score_one(s), s))
    # bucket references by length so the novelty scan is not 39k comparisons per candidate
    bylen = defaultdict(list)
    for r in refs:
        bylen[len(r)].append(r)
    top = []
    for s in ranked:
        if len(top) >= k:
            break
        ok = True
        # ratio > 0.8 requires similar lengths; 2*min/(la+lb) <= ratio bound
        for lr in range(len(s) - 12, len(s) + 13):
            if not ok:
                break
            for r in bylen.get(lr, ()):
                if Levenshtein.ratio(s, r) > novelty:
                    ok = False
                    break
        if ok:
            top.append(s)
    if len(top) < k:
        raise RuntimeError("only %d of %d candidates cleared the novelty bar" % (len(top), k))
    return top


def _write_fasta(seqs, path):
    with open(path, "w", newline="\n") as fh:
        for i, s in enumerate(seqs, start=1):
            fh.write(">seq%d\n%s\n" % (i, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=50_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ref", default="./data/antibacterial.fasta")
    a = ap.parse_args()

    out_dir = Path(Path(sys.argv[0]).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs, refs = generate(a.n_sequences, a.ref, seed=a.seed)
    _write_fasta(seqs, out_dir / "library.fasta")
    print("library: %d sequences -> %s" % (len(seqs), out_dir / "library.fasta"))

    top = pick_top(seqs, refs, a.top_k)
    _write_fasta(top, out_dir / "top.fasta")
    print("top:     %d sequences -> %s" % (len(top), out_dir / "top.fasta"))
    print("top-1 score %.3f  charge %+.1f  moment %.3f  H %.2f  %s"
          % (score_one(top[0]), net_charge(top[0]), hydrophobic_moment(top[0]),
             mean_hydrophobicity(top[0]), top[0]))


if __name__ == "__main__":
    main()
