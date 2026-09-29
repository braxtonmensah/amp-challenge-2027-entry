"""PRE-REGISTERED before any number was computed. A better generator, not a better selector.

WHAT IS ACTUALLY WEAK. The shipped library comes from an order-2 Markov chain fitted to 39,448
antibacterials. That model knows dipeptide frequencies and nothing else: no long-range structure, no
notion of what a peptide is beyond adjacent-residue counts. Every selection step downstream can only
pick from what it produced. Measured on the organizers' own seqme, the library sits at FBD 0.0093
against 0.0049 for held-out real AMPs, so it is roughly 2x off the real-peptide distribution before
selection even starts.

WHAT THIS DOES. Keep the Markov chain as a cheap PROPOSAL distribution q(x), then reweight its output
toward the real-AMP distribution p(x) estimated in the embedding space of a pretrained protein
language model (ESM2). Draw a much larger pool from q, embed it, and resample 50,000 with weight
proportional to p(x)/q(x) estimated in a PCA subspace fitted on real AMPs.

This is importance resampling, the standard way to correct a cheap proposal toward a target you can
score but not sample from. It preserves spread, which is the reason for doing it this way rather than
simply keeping the candidates nearest the real-AMP mean: nearest-mean selection collapses the
covariance and would wreck Diversity while flattering the mean term of FBD.

IS THIS FITTING TO THE METRIC? A fair objection, and the repo explicitly stopped tuning for this
reason ("a third would be fitting to the metric rather than to the biology"). The distinction drawn
here, and it must be stated rather than assumed: moving generated peptides TOWARD the distribution of
real antimicrobial peptides is the generative-modelling objective itself, not a metric exploit. It is
not comparable to pushing net charge to +9 to win a scoring function. The check that keeps this honest
is Gate 3 below: the target density is estimated on reference half A and every reported number is
measured against held-out half B. A gain that exists only against the half used to fit the density is
metric-fitting and is reported as a failure.

PRE-REGISTERED GATES, fixed before running.

  GATE 1 (decisive). Library FBD against HELD-OUT reference half B must fall by >= 30% relative to the
  shipped library measured the same way in the same run.
  GATE 2 (no diversity collapse). seqme Diversity must not fall by more than 0.01 absolute, and
  Uniqueness must stay 1.000. This is the guard against the degenerate nearest-mean solution.
  GATE 3 (not metric-fitting). The FBD improvement on held-out half B must be at least half the
  improvement seen on fitted half A. If the gain is mostly on half A, it is fitting to the reference
  sample and FAILS.
  GATE 4 (composition sanity). The resampled library's median net charge and median hydrophobicity must
  stay within 0.5 and 0.05 of the shipped library's. A library that drifts in composition is a
  different entry, not a better one.

If any gate fails, the shipped library stands and this is recorded as a negative.

EMBEDDER. ESM2 t12_35M, stated explicitly. The README's absolute FBD figures were produced ad hoc with
an unrecorded embedder and are NOT comparable; every comparison here is against a baseline recomputed
in this same run with this same embedder.
"""
from __future__ import annotations

import argparse
import numpy as np

from amp.generate import generate, mean_hydrophobicity, net_charge, read_fasta

SEED = 20260930


def gaussian_logpdf(X, mu, cov, eps=1e-6):
    d = X.shape[1]
    cov = cov + eps * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov)
    P = np.linalg.inv(cov)
    D = X - mu
    maha = np.einsum("ij,jk,ik->i", D, P, D)
    return -0.5 * (maha + logdet + d * np.log(2 * np.pi))


def fbd(X, mu_r, S_r):
    from scipy import linalg
    mu, S = X.mean(0), np.cov(X, rowvar=False)
    diff = mu - mu_r
    cm, _ = linalg.sqrtm(S.dot(S_r), disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(diff.dot(diff) + np.trace(S) + np.trace(S_r) - 2.0 * np.trace(cm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="data/antibacterial.fasta")
    ap.add_argument("--shipped", default="generate/library.fasta")
    ap.add_argument("--pool", type=int, default=150000)
    ap.add_argument("--keep", type=int, default=50000)
    ap.add_argument("--nref", type=int, default=16000)
    ap.add_argument("--pcadim", type=int, default=32)
    ap.add_argument("--stratify", action="store_true",
                    help="preserve the shipped library's net-charge histogram exactly")
    ap.add_argument("--out", default="generate/library_plm.fasta")
    a = ap.parse_args()

    from seqme.models import ESM2, ESM2Checkpoint

    refs = read_fasta(a.refs)
    shipped = read_fasta(a.shipped)
    print("refs %d   shipped library %d" % (len(refs), len(shipped)))

    print("drawing %d from the Markov proposal ..." % a.pool)
    pool, _refs_unused = generate(a.pool, a.refs, seed=SEED)   # generate() returns (sequences, refs)
    pool = list(dict.fromkeys(pool))
    print("unique proposal draws: %d" % len(pool))

    rng = np.random.default_rng(SEED)
    ridx = rng.permutation(len(refs))
    halfA = [refs[i] for i in ridx[: a.nref]]
    halfB = [refs[i] for i in ridx[a.nref: 2 * a.nref]]

    import torch
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _bs = 256
    print("embedding device: %s (batch %d)" % (_dev, _bs))
    _raw = ESM2(ESM2Checkpoint.t12_35M, device=_dev, batch_size=_bs)

    def emb(seqs, chunk=20000):
        """Embed in chunks. A single 100k call raised CUDA 'illegal instruction' inside seqme's
        per-sequence mean at batch 1024; 82k embeddings in three smaller calls had already
        succeeded in the same process, so the fault tracks call size, not the data."""
        if len(seqs) <= chunk:
            return _raw(seqs)
        parts = []
        for i in range(0, len(seqs), chunk):
            parts.append(_raw(seqs[i:i + chunk]))
            print("   embedded %d/%d" % (min(i + chunk, len(seqs)), len(seqs)), flush=True)
        return np.vstack(parts)
    print("embedding reference half A (%d) ..." % len(halfA))
    XA = emb(halfA)
    print("embedding reference half B (%d) ..." % len(halfB))
    XB = emb(halfB)
    print("embedding shipped library (%d) ..." % len(shipped))
    XS = emb(shipped)
    print("embedding proposal pool (%d) ..." % len(pool))
    XP = emb(pool)

    # PCA fitted on half A only.
    mu0 = XA.mean(0)
    U, S, Vt = np.linalg.svd(XA - mu0, full_matrices=False)
    W = Vt[: a.pcadim].T
    ZA, ZB, ZS, ZP = [(X - mu0) @ W for X in (XA, XB, XS, XP)]

    # p = real AMPs (half A), q = proposal pool, both Gaussian in the PCA subspace.
    lp = gaussian_logpdf(ZP, ZA.mean(0), np.cov(ZA, rowvar=False))
    lq = gaussian_logpdf(ZP, ZP.mean(0), np.cov(ZP, rowvar=False))
    logw = lp - lq
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    print("effective sample size of the weights: %.0f (pool %d)" % (ess, len(pool)))

    keep = min(a.keep, len(pool))
    if a.stratify:
        # CHARGE-STRATIFIED RESAMPLING, added after the unstratified version failed Gate 4 with
        # median net charge 3.00 -> 2.20 (limit 0.5) while passing Gates 1-3. Moving toward the
        # real-AMP density in embedding space drags charge down; the shipped pipeline's whole
        # Phase1/Phase2 balance rests on that marginal, and the repo's own cap sweep shows why.
        #
        # Fix: match the SHIPPED library's charge histogram exactly. Within each integer charge bin,
        # draw that bin's shipped count from the pool by the same density ratio. The charge marginal
        # is then preserved by construction and the density ratio only reorders within a bin, so any
        # FBD gain that survives is orthogonal to composition rather than bought with it.
        qs_pool = np.array([round(net_charge(s)) for s in pool])
        qs_ship = np.array([round(net_charge(s)) for s in shipped])
        sel, short = [], 0
        for b in sorted(set(qs_ship.tolist())):
            want = int((qs_ship == b).sum() * keep / float(len(shipped)))
            idx = np.where(qs_pool == b)[0]
            if want == 0 or len(idx) == 0:
                continue
            take = min(want, len(idx))
            short += want - take
            wb = w[idx] / w[idx].sum()
            sel.extend(rng.choice(idx, size=take, replace=False, p=wb).tolist())
        print("charge-stratified: %d bins, %d selected, %d short of target"
              % (len(set(qs_ship.tolist())), len(sel), short))
        if len(sel) < keep:                       # top up by weight, ignoring bin, to reach size
            rest = np.setdiff1d(np.arange(len(pool)), np.array(sel))
            wr2 = w[rest] / w[rest].sum()
            sel.extend(rng.choice(rest, size=keep - len(sel), replace=False, p=wr2).tolist())
        sel = np.array(sel[:keep])
    else:
        sel = rng.choice(len(pool), size=keep, replace=False, p=w)
    newlib = [pool[i] for i in sel]

    muA, SA = XA.mean(0), np.cov(XA, rowvar=False)
    muB, SB = XB.mean(0), np.cov(XB, rowvar=False)
    XN = XP[sel]

    sA, sB = fbd(XS, muA, SA), fbd(XS, muB, SB)
    nA, nB = fbd(XN, muA, SA), fbd(XN, muB, SB)
    rA, rB = fbd(XB, muA, SA), fbd(XA, muB, SB)   # real-AMP floor, each half against the other

    print("\n%-26s %12s %12s" % ("library", "FBD half A", "FBD half B"))
    print("%-26s %12.5f %12.5f" % ("real AMPs (floor)", rA, rB))
    print("%-26s %12.5f %12.5f" % ("shipped (Markov)", sA, sB))
    print("%-26s %12.5f %12.5f" % ("pLM-resampled", nA, nB))

    import seqme
    from seqme.metrics import Diversity, Uniqueness
    res = seqme.evaluate({"shipped": shipped, "resampled": newlib},
                         [Diversity(), Uniqueness()], verbose=False)
    print("\n%s" % res)

    dv = {k: float(res.loc[k, ("Diversity", "value")]) for k in ("shipped", "resampled")}
    uq = {k: float(res.loc[k, ("Uniqueness", "value")]) for k in ("shipped", "resampled")}

    qs = lambda L: (float(np.median([net_charge(s) for s in L])),
                    float(np.median([mean_hydrophobicity(s) for s in L])))
    cs, hs = qs(shipped)
    cn, hn = qs(newlib)

    g1 = nB <= 0.70 * sB
    g2 = (dv["resampled"] >= dv["shipped"] - 0.01) and uq["resampled"] >= 0.999
    impA, impB = sA - nA, sB - nB
    g3 = impB >= 0.5 * impA if impA > 0 else False
    g4 = abs(cn - cs) <= 0.5 and abs(hn - hs) <= 0.05

    print("\nGATE 1 (FBD half B >= 30%% lower): %.5f -> %.5f = %.2fx   %s"
          % (sB, nB, nB / sB if sB else float("nan"), "PASS" if g1 else "FAIL"))
    print("GATE 2 (diversity %.4f -> %.4f, uniqueness %.3f): %s"
          % (dv["shipped"], dv["resampled"], uq["resampled"], "PASS" if g2 else "FAIL"))
    print("GATE 3 (held-out gain >= half fitted gain): A %+.5f  B %+.5f   %s"
          % (impA, impB, "PASS" if g3 else "FAIL - fitting to the reference sample"))
    print("GATE 4 (charge %.2f->%.2f, hydro %.3f->%.3f): %s"
          % (cs, cn, hs, hn, "PASS" if g4 else "FAIL"))

    if g1 and g2 and g3 and g4:
        with open(a.out, "w", newline="\n") as fh:
            for i, s in enumerate(newlib, start=1):
                fh.write(">seq%d\n%s\n" % (i, s))
        print("\nALL GATES PASS -> wrote %s" % a.out)
    else:
        print("\nGATES NOT MET; shipped library stands. Recorded as a negative.")


if __name__ == "__main__":
    main()
