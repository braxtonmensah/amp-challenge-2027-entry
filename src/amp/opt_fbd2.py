"""Pareto frontier: how much Phase 1 distributional similarity can be bought, and what it costs.

WHY THIS SUPERSEDES opt_fbd.py. That run reported FBD 4.068 -> 3.866 on the held-out reference half,
a 5% cut, and FAILED its 40% gate. The cause was a specification bug in my own constraint, not a
property of the data: Gate 2 required the optimised set's mean `score_pool` percentile not to fall,
but the shipped top-100 IS the argmax of that quantity (percentile 0.9758), so every possible swap
violated it. 14 of 4000 proposed swaps were legal. The search was forbidden, not infeasible.

The original result stands as reported and is worth keeping: under a STRICT no-regression constraint,
top-100 FBD improves by at most ~5%. FBD and `score_pool` are in real tension.

WHAT THIS DOES INSTEAD. Measures that tension rather than banning it. For a sweep of allowed drops in
mean `score_pool` percentile, report the best FBD reachable. That produces a frontier the reader can
price, instead of one number behind a gate that could not be met.

NO GATE IS RESTATED HERE. opt_fbd.py's Gate 1 failed and is recorded as failed. This script reports a
frontier and makes no pass/fail claim; any decision to ship a point on it is a separate, explicit
judgement with the score_pool cost stated in the same breath.

METHOD NOTES.
  * Embeddings are cached to disk. The previous run recomputed and discarded 34,500 ESM2 embeddings.
  * The inner loop optimises FBD in a 64-dim PCA subspace fitted on reference half A, because the full
    480-dim objective needs a matrix square root per proposal and that dominated the last run's cost.
    Final numbers for every frontier point are reported in FULL 480 dims, on held-out half B.
  * Half A fits the PCA and drives the search; half B is never touched until reporting. A frontier that
    only exists on half A is overfitting to the reference sample and will show as a flat half-B column.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from amp.generate import mean_hydrophobicity, net_charge, read_fasta, score_pool

SEED = 20260930
ENV_CHARGE_LO, ENV_CHARGE_HI = -1.0, 5.0
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
    return (_frac(s, "W") <= CAP_W and _frac(s, "FWY") <= CAP_AROMATIC and _frac(s, "Q") <= CAP_Q)


def fbd(X, mu_r, S_r):
    from scipy import linalg
    mu, S = X.mean(0), np.cov(X, rowvar=False)
    d = mu - mu_r
    cm = linalg.sqrtm(S.dot(S_r))
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(d.dot(d) + np.trace(S) + np.trace(S_r) - 2.0 * np.trace(cm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="generate/library.fasta")
    ap.add_argument("--top", default="generate/top.fasta")
    ap.add_argument("--refs", default="data/antibacterial.fasta")
    ap.add_argument("--pool", type=int, default=8000)
    ap.add_argument("--nref", type=int, default=16000)
    ap.add_argument("--pcadim", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--max-internal", type=float, default=0.7)
    ap.add_argument("--cache", default="generate/.emb_cache.npz")
    a = ap.parse_args()

    import Levenshtein

    lib = read_fasta(a.library)
    cur = read_fasta(a.top)
    refs = read_fasta(a.refs)
    elig = [s for s in lib if in_envelope(s)]
    sc = np.asarray(score_pool(elig), dtype=float)
    cand = [elig[i] for i in np.argsort(-sc, kind="stable")[: a.pool]]
    allc = list(dict.fromkeys(cand + cur))
    print("envelope-passing %d   candidate pool %d   with current top: %d"
          % (len(elig), len(cand), len(allc)))

    rng = np.random.default_rng(SEED)
    ridx = rng.permutation(len(refs))
    halfA = [refs[i] for i in ridx[: a.nref]]
    halfB = [refs[i] for i in ridx[a.nref: 2 * a.nref]]

    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        if len(z["allc"]) == len(allc) and list(z["allc"][:5]) == allc[:5]:
            XA, XB, XC = z["XA"], z["XB"], z["XC"]
            print("loaded cached embeddings %s %s %s" % (XA.shape, XB.shape, XC.shape))
        else:
            os.remove(a.cache)
    if not os.path.exists(a.cache):
        from seqme.models import ESM2, ESM2Checkpoint
        emb = ESM2(ESM2Checkpoint.t12_35M, device="cpu", batch_size=256)
        print("embedding refA ..."); XA = emb(halfA)
        print("embedding refB ..."); XB = emb(halfB)
        print("embedding candidates (%d) ..." % len(allc)); XC = emb(allc)
        np.savez_compressed(a.cache, XA=XA, XB=XB, XC=XC, allc=np.array(allc, dtype=object))
        print("cached -> %s" % a.cache)

    shared = allc
    ssc = np.asarray(score_pool(shared), dtype=float)
    pctv = np.array([(ssc <= v).mean() for v in ssc])
    pos = {s: i for i, s in enumerate(allc)}
    cur_i = np.array([pos[s] for s in cur])
    base_pct = float(pctv[cur_i].mean())

    muA, SA = XA.mean(0), np.cov(XA, rowvar=False)
    muB, SB = XB.mean(0), np.cov(XB, rowvar=False)
    mu0 = XA.mean(0)
    Vt = np.linalg.svd(XA - mu0, full_matrices=False)[2]
    W = Vt[: a.pcadim].T
    ZA = (XA - mu0) @ W
    ZC = (XC - mu0) @ W
    muZ, SZ = ZA.mean(0), np.cov(ZA, rowvar=False)

    base_A, base_B = fbd(XC[cur_i], muA, SA), fbd(XC[cur_i], muB, SB)
    print("\nbaseline shipped top-100: FBD(A) %.4f  FBD(B) %.4f  mean pct %.4f"
          % (base_A, base_B, base_pct))

    # Price the frontier in interpretable units. "mean percentile" is not a currency; predicted panel
    # success rate is. Evaluator ridge is trained on cluster-half B of the labelled panel data, exactly
    # as in rl_finetune.py, and applied identically to every frontier point including the baseline.
    import csv as _csv
    from amp.predictor import features as _feat
    from amp.rl_finetune import cluster_by_identity as _cl, ridge_fit as _rf, ridge_pred as _rp
    _s, _r = [], []
    for _row in _csv.DictReader(open("data/labelled/panel_labels.csv", newline="")):
        _s.append(_row["sequence"]); _r.append(float(_row["success_rate"]))
    _r = np.asarray(_r)
    _as, _nc = _cl(_s)
    _rg = np.random.default_rng(SEED)
    _inA = set(_rg.permutation(_nc)[: _nc // 2].tolist())
    _iB = [i for i in range(len(_s)) if _as[i] not in _inA]
    _XB = np.vstack([_feat(_s[i]) for i in _iB])
    _mu, _sd = _XB.mean(0), _XB.std(0) + 1e-9
    _w = _rf((_XB - _mu) / _sd, _r[_iB])
    def pact(idxs):
        X = (np.vstack([_feat(allc[i]) for i in idxs]) - _mu) / _sd
        return float(_rp(_w, X).mean())
    base_act = pact(cur_i)
    print("activity evaluator: ridge on labelled half B (%d seqs); baseline top-100 %.4f"
          % (len(_iB), base_act))

    # NOVELTY SCREEN, added after the first frontier was found to be non-compliant. The optimiser
    # originally enforced only internal diversity, and the resulting sets passed just 86 of 100 on the
    # three-definition identity screen against 100 of 100 for the shipped top-100. Non-compliant
    # candidates are silently replaced by the organizers, so that "gain" was partly fictitious.
    #
    # Screening all 8,000 candidates upfront costs hours. Instead the screen is LAZY and memoised: a
    # candidate is only screened at the moment a swap would otherwise be accepted, and the verdict is
    # cached. Accepted swaps number in the hundreds, so this is affordable.
    from collections import defaultdict as _dd
    from amp.generate import _identity_ok as _idok
    _bylen = _dd(list)
    for _r in refs:
        _bylen[len(_r)].append(_r)
    _novmemo = {}

    def novel_ok(s):
        v = _novmemo.get(s)
        if v is not None:
            return v
        ok = True
        for lr in range(len(s) - 12, len(s) + 13):
            if not ok:
                break
            for r in _bylen.get(lr, ()):
                if Levenshtein.ratio(s, r) > 0.8:
                    ok = False
                    break
        if ok:
            ok = _idok(s, refs, 0.8)
        _novmemo[s] = ok
        return ok

    cand_i = np.array([pos[s] for s in cand])
    print("\n%-12s %10s %10s %10s %10s %10s %8s"
          % ("max drop", "FBD(A)", "FBD(B)", "vs base B", "mean pct", "pred act", "swaps"))
    rows = []
    for drop in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        sel = list(cur_i)
        selset = set(sel)
        cf = fbd(ZC[sel], muZ, SZ)
        nsw = 0
        r2 = np.random.default_rng(SEED + 1)
        for _ in range(a.iters):
            j = int(r2.integers(len(sel)))
            c = int(cand_i[int(r2.integers(len(cand_i)))])
            if c in selset:
                continue
            trial = list(sel); trial[j] = c
            if float(pctv[trial].mean()) < base_pct - drop:
                continue
            cs = allc[c]
            if any(Levenshtein.ratio(cs, allc[i]) > a.max_internal for k, i in enumerate(trial) if k != j):
                continue
            f = fbd(ZC[trial], muZ, SZ)
            if f < cf and novel_ok(cs):
                selset.discard(sel[j]); selset.add(c); sel = trial; cf = f; nsw += 1
        fa, fb = fbd(XC[sel], muA, SA), fbd(XC[sel], muB, SB)
        mp = float(pctv[sel].mean())
        pa = pact(sel)
        rows.append((drop, fa, fb, fb / base_B, mp, nsw, list(sel), pa))
        print("%-12.3f %10.4f %10.4f %9.2fx %10.4f %10.4f %8d"
              % (drop, fa, fb, fb / base_B, mp, pa, nsw))

    print("\nREADING: 'vs base B' is the Phase 1 gain on the HELD-OUT reference half. 'pred act' is")
    print("predicted panel success from a ridge trained on labelled half B; baseline %.4f." % base_act)
    print("A frontier point is only worth taking if pred act does not fall materially.")
    print("\n%-12s %12s %12s" % ("max drop", "FBD gain", "activity cost"))
    for d, _fa, _fb, ratio, _mp, _ns, _sel, pa in rows:
        print("%-12.3f %11.0f%% %+12.4f" % (d, 100 * (1 - ratio), pa - base_act))
    print("novelty screens performed: %d (cached verdicts)" % len(_novmemo))
    for d, _fa, _fb, ratio, _mp, _ns, sel, pa in rows:
        if d in (0.0, 0.005, 0.01, 0.02):
            out = "generate/top_fbd_drop%.3f.fasta" % d
            with open(out, "w", newline="\n") as fh:
                for i, idx in enumerate(sel, 1):
                    fh.write(">seq%d\n%s\n" % (i, allc[idx]))
            print("wrote %s (FBD %.2fx, activity %+.4f) for inspection only" % (out, ratio, pa - base_act))


if __name__ == "__main__":
    main()
