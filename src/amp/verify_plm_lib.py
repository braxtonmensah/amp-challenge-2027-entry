"""Is the pLM-resampled library actually shippable? Gates 1-4 tested the LIBRARY, not the entry.

The entry is a library PLUS a top-100 drawn from it under every shipped guard. A library that scores
better distributionally is worthless if it cannot fill a compliant, active top-100. Checks, all
against the shipped library measured identically in the same run:

  A. envelope-passing candidates (need >> 100)
  B. can a top-100 be filled under novelty + internal diversity + cysteine + composition caps
  C. predicted panel success of that top-100, ridge trained on labelled cluster-half B
  D. composition of the top-100 itself, not just of the library
"""
import csv, json
from collections import defaultdict
import numpy as np, Levenshtein
from amp.generate import (read_fasta, net_charge, mean_hydrophobicity, score_pool, _identity_ok)
from amp.predictor import features
from amp.rl_finetune import cluster_by_identity, in_envelope, ridge_fit, ridge_pred

SEED = 20260930
refs = read_fasta("data/antibacterial.fasta")
bylen = defaultdict(list)
for r in refs: bylen[len(r)].append(r)
memo = {}
def novel_ok(s):
    v = memo.get(s)
    if v is not None: return v
    ok = True
    for lr in range(len(s)-12, len(s)+13):
        if not ok: break
        for r in bylen.get(lr, ()):
            if Levenshtein.ratio(s, r) > 0.8: ok = False; break
    if ok: ok = _identity_ok(s, refs, 0.8)
    memo[s] = ok
    return ok

seqs, rew = [], []
for r in csv.DictReader(open("data/labelled/panel_labels.csv", newline="")):
    seqs.append(r["sequence"]); rew.append(float(r["success_rate"]))
rew = np.asarray(rew)
assign, ncl = cluster_by_identity(seqs)
rng = np.random.default_rng(SEED)
inA = set(rng.permutation(ncl)[:ncl//2].tolist())
iB = [i for i in range(len(seqs)) if assign[i] not in inA]
XB = np.vstack([features(seqs[i]) for i in iB]); mu, sd = XB.mean(0), XB.std(0)+1e-9
wr = ridge_fit((XB-mu)/sd, rew[iB])
def pact(L): return float(ridge_pred(wr, (np.vstack([features(s) for s in L])-mu)/sd).mean())

def build_top(lib, k=100):
    elig = [s for s in lib if in_envelope(s)]
    sc = np.asarray(score_pool(elig), dtype=float)
    top = []
    for i in np.argsort(-sc, kind="stable"):
        if len(top) >= k: break
        s = elig[i]
        if any(Levenshtein.ratio(s, t) > 0.7 for t in top): continue
        if not novel_ok(s): continue
        top.append(s)
    return elig, top

for name, path in (("shipped", "generate/library.fasta"), ("pLM-resampled", "generate/library_plm.fasta")):
    lib = read_fasta(path)
    elig, top = build_top(lib)
    mi = max(Levenshtein.ratio(top[i], top[j]) for i in range(len(top)) for j in range(i+1, len(top))) if len(top) > 1 else 0
    print("%-14s lib %6d  envelope %6d  top-100 filled %3d  predAct %.4f  medQ %.2f  medH %+.3f  maxInt %.3f  cys %d"
          % (name, len(lib), len(elig), len(top), pact(top),
             np.median([net_charge(s) for s in top]), np.median([mean_hydrophobicity(s) for s in top]),
             mi, sum(1 for s in top if "C" in s)))
    if name != "shipped":
        with open("generate/top_plm.fasta", "w", newline="\n") as fh:
            for i, s in enumerate(top, 1): fh.write(">seq%d\n%s\n" % (i, s))
        print("wrote generate/top_plm.fasta")
