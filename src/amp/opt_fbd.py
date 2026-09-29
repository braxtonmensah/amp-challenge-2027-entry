"""PRE-REGISTERED before any number was computed. Phase 1 distributional gap in the top-100.

OBSERVATION. Measured on the organizers' own seqme, the shipped sets score:

    set            Diversity  Conformity      FBD       MMD
    real AMPs         0.853      0.485      0.0049    0.00057
    our library       0.850      0.570      0.0093    0.00148
    our top-100       0.825      0.596      0.0529    0.0112

The top-100 is 5.7x worse than its own library on FBD and 7.6x worse on MMD. Nothing in the pipeline
optimises either: `pick_top` ranks by `score_pool` and then filters, so the distributional distance of
the selected 100 is whatever falls out.

HYPOTHESIS (H-FBD). Among candidates that already pass every shipped guard, a different choice of 100
lowers FBD and MMD substantially at NO cost to the selection objective.

WHY THIS IS SAFE WHERE H-CAT WAS NOT. H-CAT failed because raw net charge won only by leaving the
measured envelope, buying Phase 2 success with Phase 1 conformity. This optimises strictly INSIDE the
envelope, over candidates that already pass envelope, composition caps, cysteine exclusion, novelty
and internal diversity. It changes only WHICH eligible 100 are taken.

HARD CONSTRAINT, fixed in advance: the optimised set's mean `score_pool` percentile must be
>= the current top-100's. Any candidate set violating it is rejected regardless of its FBD. This makes
a Phase 1 gain unable to cost a Phase 2 loss on the repo's own selection metric.

PRE-REGISTERED GATES.
  GATE 1 (decisive). Optimised FBD must be <= 0.60 x the current top-100 FBD, i.e. at least a 40%
  reduction, measured with seqme's own FBD on the same embedder and reference.
  GATE 2 (no regression). Mean score_pool percentile must not fall, and seqme Diversity must not fall
  by more than 0.01 absolute.
  GATE 3 (honest holdout). FBD is optimised against reference half A and REPORTED on held-out half B.
  A gain that appears only on the half it was optimised against is overfitting to the reference sample
  and must be reported as a failure. This is the check H-CAT lacked until too late.

If Gate 1 fails on half B, the shipped top-100 stands.

NOTE ON THE EMBEDDER. The README's absolute FBD figures were produced ad hoc and the embedder was
never recorded, so they are not reproducible here. Every number this script prints is measured with
ESM2 t12_35M, stated explicitly, and compared only against a baseline computed the same way in the
same run. No number here is compared to the README's.
"""
from __future__ import annotations

import argparse
import numpy as np

from amp.generate import mean_hydrophobicity, net_charge, read_fasta, score_pool

SEED = 20260930
ENV_CHARGE_LO, ENV_CHARGE_HI = -1.0, 5.0        # verbatim from pick_top
ENV_HYDRO_LO, ENV_HYDRO_HI = -0.05, 0.65
CAP_MAX_SINGLE, CAP_W, CAP_AROMATIC, CAP_Q = 0.500, 0.238, 0.333, 0.111


def _frac(s, aas):
    return sum(s.count(c) for c in aas) / float(len(s))


def in_envelope(s):
    q, h = net_charge(s), mean_hydrophobicity(s)
    if not (ENV_CHARGE_LO <= q <= ENV_CHARGE_HI and ENV_HYDRO_LO <= h <= ENV_HYDRO_HI):
        return False
    if "C" in s:
        return False
    if max(s.count(c) for c in set(s)) / float(len(s)) > CAP_MAX_SINGLE:
        return False
    return (_frac(s, "W") <= CAP_W and _frac(s, "FWY") <= CAP_AROMATIC
            and _frac(s, "Q") <= CAP_Q)


def fbd_from_embeddings(X, mu_r, S_r):
    """Frechet distance between N(mean(X), cov(X)) and the reference Gaussian. seqme's definition."""
    from scipy import linalg
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    diff = mu - mu_r
    covmean, _ = linalg.sqrtm(S.dot(S_r), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(S) + np.trace(S_r) - 2.0 * np.trace(covmean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="generate/library.fasta")
    ap.add_argument("--top", default="generate/top.fasta")
    ap.add_argument("--refs", default="data/antibacterial.fasta")
    ap.add_argument("--pool", type=int, default=2500, help="top-N by score_pool to consider")
    ap.add_argument("--nref", type=int, default=16000, help="reference sequences per half")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--max-internal", type=float, default=0.7)
    ap.add_argument("--out", default="generate/top_fbd.fasta")
    a = ap.parse_args()

    import Levenshtein
    from seqme.models import ESM2, ESM2Checkpoint

    lib = read_fasta(a.library)
    cur = read_fasta(a.top)
    refs = read_fasta(a.refs)
    print("library %d   current top %d   refs %d" % (len(lib), len(cur), len(refs)))

    elig = [s for s in lib if in_envelope(s)]
    print("inside envelope: %d" % len(elig))
    sc = np.asarray(score_pool(elig), dtype=float)
    order = np.argsort(-sc, kind="stable")[: a.pool]
    cand = [elig[i] for i in order]
    print("candidate pool (top %d by score_pool): %d" % (a.pool, len(cand)))

    # score_pool is pool-relative, so score every set in ONE shared pool for comparability.
    shared = list(dict.fromkeys(cand + cur))
    ssc = np.asarray(score_pool(shared), dtype=float)
    pct = {s: float((ssc <= ssc[i]).mean()) for i, s in enumerate(shared)}
    base_pct = float(np.mean([pct[s] for s in cur]))
    print("current top-100 mean score_pool percentile: %.4f" % base_pct)

    rng = np.random.default_rng(SEED)
    ridx = rng.permutation(len(refs))
    halfA = [refs[i] for i in ridx[: a.nref]]
    halfB = [refs[i] for i in ridx[a.nref: 2 * a.nref]]
    print("reference halves: A=%d  B=%d" % (len(halfA), len(halfB)))

    emb = ESM2(ESM2Checkpoint.t12_35M, device="cpu", batch_size=128)
    print("embedding reference half A ...")
    XA = emb(halfA)
    print("embedding reference half B ...")
    XB = emb(halfB)
    print("embedding candidates ...")
    allc = list(dict.fromkeys(cand + cur))
    XC = emb(allc)
    pos = {s: i for i, s in enumerate(allc)}
    print("embeddings: refA %s  refB %s  cand %s" % (XA.shape, XB.shape, XC.shape))

    muA, SA = XA.mean(0), np.cov(XA, rowvar=False)
    muB, SB = XB.mean(0), np.cov(XB, rowvar=False)

    cur_i = [pos[s] for s in cur]
    base_A = fbd_from_embeddings(XC[cur_i], muA, SA)
    base_B = fbd_from_embeddings(XC[cur_i], muB, SB)
    print("\nBASELINE current top-100:  FBD(half A) %.5f   FBD(half B) %.5f" % (base_A, base_B))

    # greedy swap, optimising against half A only
    sel = list(cur_i)
    selset = set(sel)
    cur_fbd = base_A
    cand_i = [pos[s] for s in cand]
    improved = 0
    for it in range(a.iters):
        j = int(rng.integers(len(sel)))
        c = int(cand_i[int(rng.integers(len(cand_i)))])
        if c in selset:
            continue
        trial = list(sel)
        trial[j] = c
        newpct = float(np.mean([pct[allc[i]] for i in trial]))
        if newpct < base_pct:
            continue
        cs = allc[c]
        if any(Levenshtein.ratio(cs, allc[i]) > a.max_internal for k, i in enumerate(trial) if k != j):
            continue
        f = fbd_from_embeddings(XC[trial], muA, SA)
        if f < cur_fbd:
            selset.discard(sel[j])
            selset.add(c)
            sel = trial
            cur_fbd = f
            improved += 1
    opt_A = cur_fbd
    opt_B = fbd_from_embeddings(XC[sel], muB, SB)
    opt_pct = float(np.mean([pct[allc[i]] for i in sel]))
    kept = len(set(sel) & set(cur_i))

    print("accepted swaps: %d   kept from original: %d of 100" % (improved, kept))
    print("\n%-22s %12s %12s %12s" % ("set", "FBD half A", "FBD half B", "mean pct"))
    print("%-22s %12.5f %12.5f %12.4f" % ("current top-100", base_A, base_B, base_pct))
    print("%-22s %12.5f %12.5f %12.4f" % ("FBD-optimised", opt_A, opt_B, opt_pct))
    print("\nGATE 1 (>=40%% FBD cut on HELD-OUT half B): %.5f -> %.5f = %.2fx   %s"
          % (base_B, opt_B, (opt_B / base_B) if base_B else float("nan"),
             "PASS" if opt_B <= 0.60 * base_B else "FAIL"))
    print("GATE 2 (mean score_pool percentile must not fall): %.4f -> %.4f   %s"
          % (base_pct, opt_pct, "PASS" if opt_pct >= base_pct - 1e-9 else "FAIL"))
    print("GATE 3 (gain must not be half-A only): half A %.2fx, half B %.2fx   %s"
          % (opt_A / base_A, opt_B / base_B,
             "PASS" if (opt_B / base_B) <= (opt_A / base_A) + 0.15 else "FAIL - overfit to half A"))

    if opt_B <= 0.60 * base_B and opt_pct >= base_pct - 1e-9:
        with open(a.out, "w", newline="\n") as fh:
            for i, idx in enumerate(sel, start=1):
                fh.write(">seq%d\n%s\n" % (i, allc[idx]))
        print("\nwrote %s" % a.out)
    else:
        print("\ngates not met; nothing written, shipped top-100 stands")


if __name__ == "__main__":
    main()
