"""The two controls the orthogonality test lacked. PRE-REGISTERED before running.

rl_orthogonal_test found RL-generated sequences beat base-LM sequences by +0.0305 on the held-out
activity ridge at MATCHED net charge, positive in 12 of 12 bins, CI [+0.0253, +0.0358].
Two confounds could produce that without the policy having learned activity:

  C1. HYDROPHOBICITY. score_pool ranks on charge AND hydrophobicity. Controlling only for charge cannot
      show the signal is orthogonal to the selector. Here both axes are stratified jointly.
  C2. "LOOKS LIKE A REAL AMP". The evaluator ridge is fitted on real peptides, so ANY fine-tune that
      drifts the policy toward real AMPs scores higher whether or not it learned activity. The control
      is a policy fine-tuned identically but on SHUFFLED rewards: it drifts the same way and learns
      nothing. rl_likelihood_test used this null and it sat on zero; the orthogonality test did not.

GATES, fixed in advance:
  G-C1. Jointly stratified (charge x hydrophobicity) pooled diff, true reward, >= +0.02 with bootstrap
        95% CI excluding 0.
  G-C2. The SAME statistic for the shuffled-reward policy must be indistinguishable from 0 (CI contains
        0) AND materially smaller than the true-reward effect.

If G-C2 fails, the +0.0305 is fine-tuning drift toward real peptides, not learned activity, and the RL
route is closed regardless of how good the headline number looked.
"""
import csv, numpy as np, torch
from amp.generate import net_charge, mean_hydrophobicity, read_fasta
from amp.lm import encode
from amp.predictor import features
from amp.rl_finetune import cluster_by_identity, ridge_fit, ridge_pred, sample_lib
from amp.rl_likelihood_test import finetune

SEED = 20260930
N, STEPS, LR, BETA, TEMP, BATCH, MINBIN = 8000, 300, 1e-4, 0.5, 0.25, 64, 60
dev = torch.device("cpu"); rng = np.random.default_rng(SEED)
seqs, rew = [], []
for r in csv.DictReader(open("data/labelled/panel_labels.csv", newline="")):
    seqs.append(r["sequence"]); rew.append(float(r["success_rate"]))
rew = np.asarray(rew)
assign, ncl = cluster_by_identity(seqs)
inA = set(rng.permutation(ncl)[: ncl // 2].tolist())
iA = [i for i in range(len(seqs)) if assign[i] in inA]
iB = [i for i in range(len(seqs)) if assign[i] not in inA]
XB = np.vstack([features(seqs[i]) for i in iB]); mu, sd = XB.mean(0), XB.std(0) + 1e-9
wr = ridge_fit((XB - mu) / sd, rew[iB])
ck = torch.load("checkpoint/peptide_lm.pt", map_location="cpu", weights_only=False)
XA = encode([seqs[i] for i in iA]).to(dev)
refset = set(read_fasta("data/antibacterial.fasta"))
ship = read_fasta("generate_lm/library.fasta")[:N]

def pact(L):
    return ridge_pred(wr, (np.vstack([features(s) for s in L]) - mu) / sd)

def hbin(s):
    return int(np.floor(mean_hydrophobicity(s) / 0.15))

def arm(rvec, tag):
    w = torch.exp((rvec - rvec.mean()) / TEMP); w = w / w.mean()
    m = finetune(ck["cfg"], ck["state"], XA, w, STEPS, LR, BETA, BATCH, dev)
    lib = sample_lib(m, N, refset, seed=SEED)
    p_r, p_s = pact(lib), pact(ship)
    k_r = [(round(net_charge(s)), hbin(s)) for s in lib]
    k_s = [(round(net_charge(s)), hbin(s)) for s in ship]
    keys = sorted(set(k_r) & set(k_s))
    rows, tot, ws = [], 0.0, 0.0
    for k in keys:
        mr = np.array([x == k for x in k_r]); ms = np.array([x == k for x in k_s])
        if mr.sum() < MINBIN or ms.sum() < MINBIN: continue
        d = p_r[mr].mean() - p_s[ms].mean(); n = min(mr.sum(), ms.sum())
        rows.append((k, mr, ms, d, n)); tot += d * n; ws += n
    if not rows:
        print("%s: no joint bin reaches %d on both sides" % (tag, MINBIN)); return None
    pooled = tot / ws
    boots = []
    for _ in range(3000):
        s = 0.0
        for k, mr, ms, _d, n in rows:
            s += (rng.choice(p_r[mr], mr.sum(), True).mean() - rng.choice(p_s[ms], ms.sum(), True).mean()) * n
        boots.append(s / ws)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    npos = sum(1 for r in rows if r[3] > 0)
    print("%-22s pooled %+.4f  CI [%+.4f, %+.4f]  bins %d  positive %d"
          % (tag, pooled, lo, hi, len(rows), npos))
    return pooled, lo, hi

print("jointly stratified by net charge x hydrophobicity (0.15 bins), min %d per cell\n" % MINBIN)
t = arm(torch.tensor(rew[iA], dtype=torch.float32, device=dev), "true reward")
n = arm(torch.tensor(rng.permutation(rew[iA]), dtype=torch.float32, device=dev), "shuffled reward")
if t and n:
    g1 = t[0] >= 0.02 and t[1] > 0
    g2 = (n[1] <= 0 <= n[2]) and abs(n[0]) < t[0] / 2.0
    print("\nG-C1 (true effect >= +0.02, CI excludes 0): %s" % ("PASS" if g1 else "FAIL"))
    print("G-C2 (shuffled null contains 0 and is < half): %s" % ("PASS" if g2 else "FAIL"))
    print("VERDICT: %s" % ("orthogonal to BOTH selector axes and not drift - REAL"
                           if (g1 and g2) else "not established"))
