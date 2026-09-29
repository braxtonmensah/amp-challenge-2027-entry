"""PRE-REGISTERED, written before any number below was computed.

QUESTION. Both shipped entries select their top-100 with the same rule, `score_pool` =
rank(net charge) - rank(mean hydrophobicity). That rule was adopted in PREREG_SELECTION_2.md to
maximise the safety window HC50/MIC50, and `score_pool`'s own docstring states the consequence
plainly: "This entry therefore optimises one category and concedes the four that average Success
Rate over all 25 peptides."

The competition scores FIVE categories (broad-spectrum, Gram-positive, Gram-negative, MDR, Optimal
Selectivity) and the rules permit one entry per sufficiently different model. We have two models.
Using the same selection for both spends two entries on one category.

HYPOTHESIS (H-CAT). A selection rule targeting raw potency beats `score_pool` on measured panel
success rate, so entry 2 should be re-selected for the Success Rate categories while entry 1 keeps
`score_pool` for Optimal Selectivity.

Prior evidence, already in the repo and NOT re-derived here:
  * raw net charge ranks panel success at AUC 0.678; the banded charge used by the retired composite
    gets 0.590 (PREREG_SELECTION_2.md section 1).
  * `score_pool` deliberately subtracts hydrophobicity, which costs potency to buy safety window.
  * score_pool's docstring reports that at charge +4..+5, success rate rises from 0.551 inside the
    shipped hydrophobicity band to 0.665 at hydrophobicity 0.05-0.25.

PRE-REGISTERED GATES. Fixed before running. Mirrors the Gate 2 standard already used in this repo.

  GATE A (decisive). On the held-out cluster half, the challenger's top-100 must beat `score_pool`'s
  top-100 by >= 0.05 ABSOLUTE mean measured panel success rate. A tie is a failure. If no challenger
  clears this, entry 2 ships unchanged and H-CAT is refuted.

  GATE B (no free lunch check). Report `score_pool`'s advantage on the safety-window proxy in the same
  table. If the challenger also wins there, that is evidence the split is unnecessary rather than
  evidence it is good, and it must be reported as such rather than claimed as a double win.

  GATE C (category specificity). Report success restricted to Gram-negative and Gram-positive species
  counts separately. A challenger that wins overall but loses on Gram-negative is not a
  Gram-negative-category entry, because the panel is 15 Gram-negative of 20 strains.

EVALUATION PROTOCOL. Identical in spirit to eval_selection.py: sequences are clustered by Levenshtein
ratio >= 0.6 and whole clusters are assigned to halves, so no near-duplicate family straddles the
split. Selection rules are pool-relative (score_pool z-ranks within the pool it is given), so every
rule is applied to the SAME held-out pool. No rule is fitted on the held-out half.

NULL. A random top-100 from the same pool, averaged over 200 draws, is reported so that every number
has a floor. Any rule that fails to beat random is not a selection rule.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from amp.generate import mean_hydrophobicity, net_charge, score_one, score_pool

SEED = 20260930


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


# ---------------------------------------------------------------- selection rules
# Each returns a score array over the pool; higher is better; top-k are taken.

def rule_score_pool(seqs):
    """The shipped rule. rank(charge) - rank(hydrophobicity). Targets safety window."""
    return np.asarray(score_pool(seqs), dtype=float)


def rule_raw_charge(seqs):
    """Raw net charge, unbanded. The best single predictor of panel success in this repo."""
    return np.asarray([net_charge(s) for s in seqs], dtype=float)


def rule_charge_hydro_window(seqs):
    """Charge, restricted to the hydrophobicity region score_pool's own docstring reports as best.

    Peptides outside 0.05..0.25 hydrophobicity are pushed below every in-window peptide rather than
    hard-filtered, so the rule always returns a full ranking even if the window is sparse.
    """
    q = np.asarray([net_charge(s) for s in seqs], dtype=float)
    h = np.asarray([mean_hydrophobicity(s) for s in seqs], dtype=float)
    inwin = (h >= 0.05) & (h <= 0.25)
    return q + 1000.0 * inwin


def rule_score_one(seqs):
    """The retired composite, as a floor."""
    return np.asarray([score_one(s) for s in seqs], dtype=float)


RULES = {
    "score_pool (SHIPPED, both entries)": rule_score_pool,
    "raw net charge": rule_raw_charge,
    "charge + hydrophobicity window": rule_charge_hydro_window,
    "score_one (retired)": rule_score_one,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labelled/panel_labels.csv")
    ap.add_argument("--topk", type=int, default=100)
    a = ap.parse_args()

    seqs, rows = [], []
    with open(a.data, newline="") as fh:
        for row in csv.DictReader(fh):
            seqs.append(row["sequence"])
            rows.append(row)
    print("labelled sequences: %d" % len(seqs))

    succ = np.array([float(r["success_rate"]) for r in rows])
    lmic = np.array([float(r["log_mic50"]) for r in rows])
    nneg = np.array([float(r["n_gram_neg_species"]) for r in rows])
    npos = np.array([float(r["n_gram_pos_species"]) for r in rows])

    assign, ncl = cluster_by_identity(seqs)
    print("clusters: %d (mean %.1f seqs/cluster)" % (ncl, len(seqs) / float(ncl)))

    rng = np.random.default_rng(SEED)
    cl = np.arange(ncl)
    rng.shuffle(cl)
    half = set(cl[: ncl // 2].tolist())          # half A = dev, unused here
    idxB = np.array([i for i in range(len(seqs)) if assign[i] not in half])
    print("held-out half B: %d sequences\n" % len(idxB))

    poolB = [seqs[i] for i in idxB]
    k = min(a.topk, len(idxB))

    print("=" * 96)
    print("GATE A: mean MEASURED panel success rate of the top-%d, held-out half" % k)
    print("=" * 96)
    print("%-38s %10s %10s %10s %10s" % ("selection rule", "success", "logMIC50", "gram-neg", "gram-pos"))

    results = {}
    for name, fn in RULES.items():
        sc = fn(poolB)
        top = idxB[np.argsort(-sc, kind="stable")[:k]]
        results[name] = float(succ[top].mean())
        print("%-38s %10.4f %10.4f %10.2f %10.2f"
              % (name, succ[top].mean(), lmic[top].mean(), nneg[top].mean(), npos[top].mean()))

    draws = []
    for _ in range(200):
        pick = rng.choice(idxB, size=k, replace=False)
        draws.append(succ[pick].mean())
    rnd = float(np.mean(draws))
    print("%-38s %10.4f %10s %10s %10s" % ("RANDOM null (200 draws)", rnd, "-", "-", "-"))

    base = results["score_pool (SHIPPED, both entries)"]
    print()
    print("=" * 96)
    print("GATE A VERDICT  (requirement: challenger - score_pool >= +0.05 absolute)")
    print("=" * 96)
    any_pass = False
    for name, v in results.items():
        if name.startswith("score_pool"):
            continue
        d = v - base
        ok = d >= 0.05
        any_pass = any_pass or ok
        print("  %-36s %+.4f   %s" % (name, d, "PASS" if ok else "fail"))
    print()
    print("  H-CAT: %s" % ("SUPPORTED - re-select entry 2" if any_pass
                           else "REFUTED - entry 2 ships unchanged"))


if __name__ == "__main__":
    main()
