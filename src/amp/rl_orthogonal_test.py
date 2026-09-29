"""Is the RL policy's learned signal ORTHOGONAL to net charge, or redundant with it?

THE SITUATION THIS RESOLVES. Two measured results that are both true and point opposite ways:

  * rl_likelihood_test: the fine-tuned policy assigns higher likelihood to high-activity held-out
    peptides than the base policy does, delta +0.01873, 95% CI [+0.00446, +0.03257], with a
    shuffled-reward null on zero. The policy really did learn activity.
  * rl_finetune Gate 1: the RL library's top-100, selected by the shipped rule, scored 0.5506 against
    0.5629 for the shipped library. Slightly WORSE.

The reconciling hypothesis: what the policy learned is net charge (median charge moved +2.20 -> +4.00),
and `score_pool` already ranks on charge, so the policy's knowledge is redundant with the selector.

THE TEST. Compare RL-generated and shipped-generated sequences at MATCHED net charge. If the RL policy
knows anything about activity beyond charge, its sequences should score higher on the held-out
activity ridge WITHIN a charge bin. If the advantage vanishes once charge is controlled, the learned
signal is charge and nothing else, and this route is closed.

This is the same confound structure that killed H-CAT: an apparent gain that is really a composition
shift. There it was caught by re-running inside the envelope. Here it is caught by stratifying.

PRE-REGISTERED, fixed before running:
  GATE O1 (decisive). Pooled charge-stratified difference in predicted activity (RL minus shipped,
  weighted by bin size, bins with >= 100 sequences on both sides) must be >= +0.02 with a bootstrap
  95% CI excluding 0.
  GATE O2 (direction consistency). The difference must be positive in a MAJORITY of qualifying bins.
  A pooled win driven by one bin is not an orthogonal signal.

If both fail, the honest conclusion is that KL-anchored reward-weighted fine-tuning on this corpus
recovers the charge axis the shipped scorer already uses, and adds nothing on top of it.

The evaluator ridge is trained on cluster-half B; the policy is fine-tuned on half A. Unchanged.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from amp.generate import net_charge, read_fasta
from amp.lm import PeptideLM, encode
from amp.predictor import features
from amp.rl_finetune import cluster_by_identity, ridge_fit, ridge_pred, sample_lib
from amp.rl_likelihood_test import finetune

SEED = 20260930


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint/peptide_lm.pt")
    ap.add_argument("--labels", default="data/labelled/panel_labels.csv")
    ap.add_argument("--refs", default="data/antibacterial.fasta")
    ap.add_argument("--shipped", default="generate_lm/library.fasta")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--minbin", type=int, default=100)
    a = ap.parse_args()

    dev = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    seqs, rew = [], []
    with open(a.labels, newline="") as fh:
        for r in csv.DictReader(fh):
            seqs.append(r["sequence"])
            rew.append(float(r["success_rate"]))
    rew = np.asarray(rew)
    assign, ncl = cluster_by_identity(seqs)
    cl = rng.permutation(ncl)
    inA = set(cl[: ncl // 2].tolist())
    iA = [i for i in range(len(seqs)) if assign[i] in inA]
    iB = [i for i in range(len(seqs)) if assign[i] not in inA]

    XB = np.vstack([features(seqs[i]) for i in iB])
    mu, sd = XB.mean(0), XB.std(0) + 1e-9
    wr = ridge_fit((XB - mu) / sd, rew[iB])
    print("evaluator ridge on half B (%d seqs); policy trains on half A (%d)" % (len(iB), len(iA)))

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    XA = encode([seqs[i] for i in iA]).to(dev)
    rA = torch.tensor(rew[iA], dtype=torch.float32, device=dev)
    w = torch.exp((rA - rA.mean()) / a.temp)
    w = w / w.mean()
    print("fine-tuning policy (%d steps, beta=%.2f) ..." % (a.steps, a.beta))
    model = finetune(ck["cfg"], ck["state"], XA, w, a.steps, a.lr, a.beta, a.batch, dev)

    refset = set(read_fasta(a.refs))
    print("sampling %d from RL policy ..." % a.n)
    rl = sample_lib(model, a.n, refset, seed=SEED)
    ship = read_fasta(a.shipped)[: a.n]
    print("RL %d   shipped %d" % (len(rl), len(ship)))

    def pact(L):
        X = (np.vstack([features(s) for s in L]) - mu) / sd
        return ridge_pred(wr, X)

    p_rl, p_sh = pact(rl), pact(ship)
    q_rl = np.array([round(net_charge(s)) for s in rl])
    q_sh = np.array([round(net_charge(s)) for s in ship])

    print("\nunstratified: RL %.4f   shipped %.4f   diff %+.4f"
          % (p_rl.mean(), p_sh.mean(), p_rl.mean() - p_sh.mean()))
    print("median charge: RL %.2f   shipped %.2f"
          % (np.median([net_charge(s) for s in rl]), np.median([net_charge(s) for s in ship])))

    bins = sorted(set(q_rl.tolist()) & set(q_sh.tolist()))
    print("\n%-8s %8s %8s %10s %10s %10s" % ("charge", "n_rl", "n_ship", "RL", "shipped", "diff"))
    rows, tot, wsum = [], 0.0, 0.0
    for b in bins:
        mr, ms = q_rl == b, q_sh == b
        if mr.sum() < a.minbin or ms.sum() < a.minbin:
            continue
        d = p_rl[mr].mean() - p_sh[ms].mean()
        n = min(mr.sum(), ms.sum())
        rows.append((b, mr, ms, d, n))
        tot += d * n
        wsum += n
        print("%-8d %8d %8d %10.4f %10.4f %+10.4f"
              % (b, mr.sum(), ms.sum(), p_rl[mr].mean(), p_sh[ms].mean(), d))
    if not rows:
        print("\nno charge bin has >= %d on both sides; test inconclusive" % a.minbin)
        return
    pooled = tot / wsum

    boots = []
    for _ in range(4000):
        s = 0.0
        for b, mr, ms, _d, n in rows:
            ar = rng.choice(p_rl[mr], size=mr.sum(), replace=True).mean()
            as_ = rng.choice(p_sh[ms], size=ms.sum(), replace=True).mean()
            s += (ar - as_) * n
        boots.append(s / wsum)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    npos = sum(1 for r in rows if r[3] > 0)

    print("\npooled charge-stratified diff: %+.4f   95%% CI [%+.4f, %+.4f]" % (pooled, lo, hi))
    print("bins positive: %d of %d" % (npos, len(rows)))
    g1 = pooled >= 0.02 and lo > 0
    g2 = npos > len(rows) / 2.0
    print("\nGATE O1 (pooled >= +0.02, CI excludes 0): %s" % ("PASS" if g1 else "FAIL"))
    print("GATE O2 (majority of bins positive): %s" % ("PASS" if g2 else "FAIL"))
    print("VERDICT: %s" % ("orthogonal signal exists - worth building on" if (g1 and g2) else
                           "NO orthogonal signal; the RL recovered the charge axis score_pool already uses"))


if __name__ == "__main__":
    main()
