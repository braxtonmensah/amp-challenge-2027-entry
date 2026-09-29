"""Did the RL fine-tune actually learn the reward? A likelihood test with no proxy and no OOD step.

WHY THIS EXISTS. rl_finetune.py judges the policy by generating sequences and scoring them with a
ridge trained on real AMPs. That ridge is fitted on real peptides and then applied to GENERATED ones,
which are outside its training distribution, so a difference there is not clean evidence that the
policy learned anything about activity.

This test avoids both problems. It asks a question about the policy directly, on real held-out
sequences it was never trained on:

    Does the fine-tuned policy move probability mass toward HIGH-activity peptides,
    relative to what the base policy already did, more than it does for LOW-activity ones?

    delta = [ logP_rl(high) - logP_base(high) ] - [ logP_rl(low) - logP_base(low) ]

Differencing against the base policy removes any overall likelihood drift from fine-tuning, and
differencing high against low removes anything that is just "these sequences are easier to model".
If reward-weighted training worked, delta > 0. If the policy merely drifted, delta ~ 0.

PRE-REGISTERED, fixed before running:
  GATE L1. delta > 0 with a paired bootstrap 95% CI excluding 0, on the cluster-disjoint held-out
  half B. Sequences are split at the median measured success rate of half B.
  GATE L2 (specificity). The same statistic computed with SHUFFLED rewards must be indistinguishable
  from 0. This is the null that catches a delta produced by anything other than the reward.

A failure here means the RL fine-tune did not learn activity, whatever the generated-library numbers
say, and it should be reported as the stronger evidence because it is the cleaner measurement.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F

from amp.lm import PAD, VOCAB, PeptideLM, encode
from amp.rl_finetune import cluster_by_identity

SEED = 20260930


@torch.no_grad()
def seq_logprob(model, X, batch=256):
    """Mean per-token log-probability of each sequence under `model`."""
    out = []
    for i in range(0, len(X), batch):
        x = X[i:i + batch]
        logits = model(x[:, :-1])
        tgt = x[:, 1:]
        lp = -F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                              ignore_index=PAD, reduction="none").view(tgt.shape)
        m = (tgt != PAD).float()
        out.append(((lp * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def finetune(base_cfg, state, XA, w, steps, lr, beta, batch, dev, seed=SEED):
    torch.manual_seed(seed)
    ref = PeptideLM(**base_cfg).to(dev)
    ref.load_state_dict(state)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model = PeptideLM(**base_cfg).to(dev)
    model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        j = torch.randint(0, len(XA), (min(batch, len(XA)),), device=dev)
        x, ww = XA[j], w[j]
        logits = model(x[:, :-1])
        tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             ignore_index=PAD, reduction="none").view(tgt.shape)
        m = (tgt != PAD).float()
        per_seq = (ce * m).sum(1) / m.sum(1).clamp(min=1)
        rwr = (ww * per_seq).mean()
        with torch.no_grad():
            rlog = F.log_softmax(ref(x[:, :-1]), dim=-1)
        kl = F.kl_div(rlog, F.log_softmax(logits, dim=-1), log_target=True, reduction="none").sum(-1)
        kl = (kl * m).sum() / m.sum().clamp(min=1)
        loss = rwr + beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


def paired_boot(hi, lo, n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        a = rng.choice(hi, size=len(hi), replace=True)
        b = rng.choice(lo, size=len(lo), replace=True)
        d.append(a.mean() - b.mean())
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint/peptide_lm.pt")
    ap.add_argument("--labels", default="data/labelled/panel_labels.csv")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
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
    print("policy-train half A: %d   held-out half B: %d" % (len(iA), len(iB)))

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    base = PeptideLM(**ck["cfg"]).to(dev)
    base.load_state_dict(ck["state"])
    base.eval()

    XA = encode([seqs[i] for i in iA]).to(dev)
    XB = encode([seqs[i] for i in iB]).to(dev)
    rA = torch.tensor(rew[iA], dtype=torch.float32, device=dev)
    rB = rew[iB]

    def run(rvec, tag):
        w = torch.exp((rvec - rvec.mean()) / a.temp)
        w = w / w.mean()
        m = finetune(ck["cfg"], ck["state"], XA, w, a.steps, a.lr, a.beta, a.batch, dev)
        lp_rl = seq_logprob(m, XB)
        lp_bs = seq_logprob(base, XB)
        d = lp_rl - lp_bs
        med = np.median(rB)
        hi, lo = d[rB > med], d[rB <= med]
        mean, lo_ci, hi_ci = paired_boot(hi, lo)
        print("%-28s delta %+.5f   95%% CI [%+.5f, %+.5f]   n_hi %d n_lo %d"
              % (tag, mean, lo_ci, hi_ci, len(hi), len(lo)))
        return mean, lo_ci, hi_ci

    print("\nGATE L1: true rewards")
    m1, l1, h1 = run(rA, "true reward")
    print("\nGATE L2: shuffled-reward null")
    perm = torch.tensor(rng.permutation(rew[iA]), dtype=torch.float32, device=dev)
    m0, l0, h0 = run(perm, "shuffled reward")

    print("\n" + "=" * 78)
    g1 = l1 > 0
    g2 = (l0 <= 0 <= h0)
    print("GATE L1 (delta > 0, CI excludes 0): %s" % ("PASS" if g1 else "FAIL"))
    print("GATE L2 (shuffled null contains 0): %s" % ("PASS" if g2 else "FAIL"))
    print("VERDICT: %s" % ("the RL fine-tune learned activity on held-out sequences"
                           if (g1 and g2) else
                           "NOT demonstrated - the policy did not measurably learn the reward"))


if __name__ == "__main__":
    main()
