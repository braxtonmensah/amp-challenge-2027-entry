"""PRE-REGISTERED before any number was computed. Spend the RL signal in SELECTION, not generation.

WHY THIS AND NOT THE GENERATOR. rl_finetune.py showed the RL policy generates a library whose
composition drifts (median charge +2.20 -> +4.00) and whose diversity falls, failing three gates. But
rl_orth_control.py showed the policy carries signal that is genuinely orthogonal to BOTH axes
`score_pool` uses: +0.0149 predicted panel success at matched charge AND hydrophobicity, CI
[+0.0098, +0.0196], with a shuffled-reward null at +0.0012 CI [-0.0038, +0.0061].

So the signal is real and the generator is the wrong place to spend it. Using the policy as a RANKER
over the SHIPPED library keeps the library's composition, diversity and novelty exactly as they are,
and applies the orthogonal signal only at the selection step, where it is free.

THE SCORE. For each envelope-passing candidate, compute mean per-token log-likelihood under the RL
policy and under the frozen base policy, and use the DIFFERENCE:

    lift(x) = logP_rl(x) - logP_base(x)

Differencing is essential. Raw logP_rl would rank on generic peptide-likeness, which is what the base
policy already encodes and what the shipped pipeline's distributional metrics reward anyway. The
difference isolates what the reward moved, which is the only part shown to be orthogonal.

    new_score(x) = zrank(charge) - zrank(hydrophobicity) + W * zrank(lift)

W = 0 recovers `score_pool` exactly, so the shipped selection is nested inside this as a special case
and the sweep below cannot fail to contain it.

PRE-REGISTERED GATES, fixed before running. Reported at every W; the decision uses W chosen ONLY by
these gates, not by eyeballing.

  GATE S1 (decisive). Predicted panel success of the top-100, on a ridge trained on labelled
  cluster-half B, must beat the shipped top-100 by >= +0.02 absolute. The measured orthogonal margin
  is +0.0149 on library average and selection should concentrate it, so +0.02 is the bar for this
  being worth taking. A tie is a failure.
  GATE S2 (composition, veto). Median net charge within 0.5 and median hydrophobicity within 0.05 of
  the shipped top-100. This is the guard that caught H-CAT and the RL generator; any candidate set
  that drifts is rejected regardless of its activity.
  GATE S3 (compliance, veto). 100 of 100 on the three-definition identity screen, max internal
  Levenshtein <= 0.7, zero cysteine. Equal to the shipped top-100, not merely close.
  GATE S4 (Phase 1 must not be paid for this). FBD against a held-out reference half must not rise
  by more than 10% versus the shipped top-100.

The policy is fine-tuned on labelled cluster-half A; the activity evaluator is fitted on half B. The
policy never sees the evaluator's sequences. Unchanged from every other test in this series.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
import torch

from amp.generate import (mean_hydrophobicity, net_charge, read_fasta, _identity_ok, _zrank)
from amp.lm import PeptideLM, encode
from amp.predictor import features
from amp.rl_finetune import cluster_by_identity, in_envelope, ridge_fit, ridge_pred
from amp.rl_likelihood_test import finetune, seq_logprob

SEED = 20260930
STEPS, LR, BETA, TEMP, BATCH = 300, 1e-4, 0.5, 0.25, 64


def main():
    import Levenshtein
    dev = torch.device("cpu")
    rng = np.random.default_rng(SEED)

    seqs, rew = [], []
    for r in csv.DictReader(open("data/labelled/panel_labels.csv", newline="")):
        seqs.append(r["sequence"]); rew.append(float(r["success_rate"]))
    rew = np.asarray(rew)
    assign, ncl = cluster_by_identity(seqs)
    inA = set(rng.permutation(ncl)[: ncl // 2].tolist())
    iA = [i for i in range(len(seqs)) if assign[i] in inA]
    iB = [i for i in range(len(seqs)) if assign[i] not in inA]
    XB = np.vstack([features(seqs[i]) for i in iB])
    mu, sd = XB.mean(0), XB.std(0) + 1e-9
    wr = ridge_fit((XB - mu) / sd, rew[iB])
    print("policy trains on half A (%d); activity ridge on half B (%d)" % (len(iA), len(iB)))

    ck = torch.load("checkpoint/peptide_lm.pt", map_location="cpu", weights_only=False)
    base = PeptideLM(**ck["cfg"]).to(dev); base.load_state_dict(ck["state"]); base.eval()
    XA = encode([seqs[i] for i in iA]).to(dev)
    rA = torch.tensor(rew[iA], dtype=torch.float32, device=dev)
    w = torch.exp((rA - rA.mean()) / TEMP); w = w / w.mean()
    print("fine-tuning policy ...")
    pol = finetune(ck["cfg"], ck["state"], XA, w, STEPS, LR, BETA, BATCH, dev)

    lib = read_fasta("generate/library.fasta")
    cur = read_fasta("generate/top.fasta")
    refs = read_fasta("data/antibacterial.fasta")
    elig = [s for s in lib if in_envelope(s)]
    print("envelope-passing candidates: %d" % len(elig))

    XE = encode(elig).to(dev)
    print("scoring likelihood under policy and base ...")
    lift = seq_logprob(pol, XE) - seq_logprob(base, XE)
    q = np.array([net_charge(s) for s in elig]); h = np.array([mean_hydrophobicity(s) for s in elig])
    zq, zh, zl = _zrank(q), _zrank(h), _zrank(lift)
    print("lift: mean %+.4f  sd %.4f" % (lift.mean(), lift.std()))

    bylen = defaultdict(list)
    for r in refs:
        bylen[len(r)].append(r)
    memo = {}

    def novel_ok(s):
        v = memo.get(s)
        if v is not None:
            return v
        ok = True
        for lr in range(len(s) - 12, len(s) + 13):
            if not ok:
                break
            for r in bylen.get(lr, ()):
                if Levenshtein.ratio(s, r) > 0.8:
                    ok = False; break
        if ok:
            ok = _identity_ok(s, refs, 0.8)
        memo[s] = ok
        return ok

    def pick(score, k=100, max_internal=0.7):
        top = []
        for i in np.argsort(-score, kind="stable"):
            if len(top) >= k:
                break
            s = elig[i]
            if any(Levenshtein.ratio(s, t) > max_internal for t in top):
                continue
            if not novel_ok(s):
                continue
            top.append(s)
        return top

    def pact(L):
        return float(ridge_pred(wr, (np.vstack([features(s) for s in L]) - mu) / sd).mean())

    base_top = pick(zq - zh)
    base_act = pact(base_top)
    shipped_act = pact(cur)
    print("\nshipped top.fasta predicted activity   %.4f" % shipped_act)
    print("W=0 reproduction (sanity, should match) %.4f  overlap with shipped %d/100"
          % (base_act, len(set(base_top) & set(cur))))

    print("\n%-6s %10s %10s %8s %8s %8s %6s" % ("W", "pred act", "vs shipped", "medQ", "medH", "maxInt", "cys"))
    results = []
    for W in (0.0, 0.25, 0.5, 1.0, 2.0):
        top = pick(zq - zh + W * zl)
        pa = pact(top)
        mq = float(np.median([net_charge(s) for s in top]))
        mh = float(np.median([mean_hydrophobicity(s) for s in top]))
        mi = max(Levenshtein.ratio(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top)))
        cy = sum(1 for s in top if "C" in s)
        results.append((W, top, pa, mq, mh, mi, cy))
        print("%-6.2f %10.4f %+10.4f %8.2f %+8.3f %8.3f %6d" % (W, pa, pa - shipped_act, mq, mh, mi, cy))

    sq = float(np.median([net_charge(s) for s in cur])); sh = float(np.median([mean_hydrophobicity(s) for s in cur]))
    print("\nshipped reference: medQ %.2f  medH %+.3f" % (sq, sh))
    print("\n%-6s %-10s %-12s %-12s" % ("W", "S1 act", "S2 comp", "S3 compliance"))
    for W, top, pa, mq, mh, mi, cy in results:
        s1 = (pa - shipped_act) >= 0.02
        s2 = abs(mq - sq) <= 0.5 and abs(mh - sh) <= 0.05
        s3 = (len(top) == 100) and mi <= 0.7 and cy == 0 and all(novel_ok(s) for s in top)
        print("%-6.2f %-10s %-12s %-12s" % (W, "PASS" if s1 else "fail",
                                            "PASS" if s2 else "fail", "PASS" if s3 else "fail"))
    ok = [r for r in results if (r[2] - shipped_act) >= 0.02 and abs(r[3] - sq) <= 0.5
          and abs(r[4] - sh) <= 0.05 and r[5] <= 0.7 and r[6] == 0]
    if ok:
        best = max(ok, key=lambda r: r[2])
        out = "generate/top_rlsel_W%.2f.fasta" % best[0]
        with open(out, "w", newline="\n") as fh:
            for i, s in enumerate(best[1], 1):
                fh.write(">seq%d\n%s\n" % (i, s))
        print("\nGATES S1-S3 PASS at W=%.2f (%+.4f activity) -> %s  [S4 FBD check still required]"
              % (best[0], best[2] - shipped_act, out))
    else:
        print("\nNo W clears the pre-registered gates. Shipped selection stands.")


if __name__ == "__main__":
    main()
