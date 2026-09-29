# KL-anchored RL fine-tuning: the policy learns real, orthogonal signal, below the useful threshold

Run 2026-09-29. `src/amp/rl_finetune.py`, `rl_likelihood_test.py`, `rl_orthogonal_test.py`,
`rl_orth_control.py`. **Nothing shipped. The shipped generator stands.**

## What was run, precisely

PPO on a 6.4B protein language model needs ~13GB VRAM and days; this machine has four CPU threads.
What was run is the same objective without the large model and without PPO's clipped surrogate:

    maximise  E_{x ~ pi}[ r(x) ]  -  beta * KL( pi || pi_ref )

with `pi_ref` the shipped 0.81M transformer, `r` the measured panel success rate of 2,904 peptides,
and the first term's policy gradient estimated offline by reward-weighted MLE (reward-weighted
regression) since a sequence is generated then scored once. Calling this RL is accurate; calling it
PPO would not be. The KL anchor is load-bearing: `lm.py` warns that memorising this corpus is worse
than useless because memorised real AMPs fail the <=80% identity rule. Zero memorised duplicates were
produced, so the anchor worked.

Throughout, the policy is fine-tuned on cluster-half A of the labelled data and every evaluator is
fitted on half B, with whole similarity families held out. The policy never sees the evaluator's data.

## 1. The generated library FAILED, and informatively

| gate | result |
|---|---|
| 1, predicted success >= +0.05 | **-0.0123 FAIL** |
| 2, zero memorised duplicates | 0 PASS |
| 3, diversity within 0.01 | 0.8572 -> 0.8395 **FAIL** |
| 4, composition within 0.5 charge | **2.20 -> 4.00 FAIL** |

Gate 4 is the finding. The policy learned the reward and expressed it as **net charge**, the axis that
raises panel success and collapses Phase 1 conformity. That is the third independent arrival at the
conflict `pick_top` already documents, after H-CAT and the repo's own cap sweep.

## 2. The policy DID learn activity, cleanly

Likelihood test on held-out real sequences, no proxy and no out-of-distribution step. Does the
fine-tuned policy raise likelihood on high-activity held-out peptides more than on low-activity ones,
relative to the base policy?

    true reward      delta +0.01873   95% CI [+0.00446, +0.03257]   PASS
    shuffled reward  delta +0.00300   95% CI [-0.01078, +0.01676]   PASS (null on zero)

This is the strongest evidence in the whole exercise, because it is measured on real peptides the
policy never saw and the shuffled-reward arm rules out fine-tuning drift.

## 3. Orthogonality: real, and smaller than required

Stratified against the selector's own axes, versus the base-LM library, on the held-out activity ridge.

| control | pooled diff | 95% CI | bins positive |
|---|---|---|---|
| charge only | +0.0305 | [+0.0253, +0.0358] | 12 of 12 |
| **charge x hydrophobicity** | **+0.0149** | **[+0.0098, +0.0196]** | 28 of 36 |
| charge x hydrophobicity, **shuffled reward** | +0.0012 | [-0.0038, +0.0061] | 21 of 38 |

Pre-registered G-C1 required **>= +0.02 with the CI excluding 0**. Measured **+0.0149**. **FAIL.**
G-C2 passed: the shuffled null sits on zero, so the effect is the reward and not drift.

Read precisely, because the binary verdict hides the shape of it:

* The effect is **statistically real**. The CI excludes zero and the shuffled arm does not.
* It is **genuinely orthogonal** to both axes `score_pool` uses, which is more than expected: the
  working hypothesis was that RL had merely rediscovered net charge, and that is refuted.
* Controlling for hydrophobicity halved it, from +0.0305 to +0.0149. About half of the charge-only
  estimate was the hydrophobicity axis the selector already exploits.
* **+0.0149 is below the +0.02 threshold registered as decision-relevant, so this does not justify
  restructuring the entry.** The bar was not moved after seeing the number.

## Conclusion

KL-anchored reward-weighted fine-tuning on 2,904 measured peptides produces a policy that really has
learned something about activity beyond charge and hydrophobicity. The margin is about +0.015 in
predicted panel success, on a ridge proxy, which is too small to buy back the diversity and
composition costs its generated library incurs.

**The shipped generator stands.** The obvious next step, if this is ever revisited with more compute:
use the RL policy's likelihood as a third selection term inside `pick_top` rather than as a generator,
which keeps the shipped library's composition and diversity and spends the orthogonal signal only
where it is free. That was not run here and is untested.

---

## 4. Spending the signal in SELECTION instead: also fails, monotonically

The conclusion above named the obvious next step: use the policy's likelihood as a third term in
`pick_top` rather than as a generator, keeping the shipped library's composition and diversity and
spending the orthogonal signal only where it is free. `src/amp/rl_as_selector.py` ran it.

Score: `zrank(charge) - zrank(hydrophobicity) + W * zrank(logP_rl - logP_base)`. The difference against
the base policy is used rather than raw policy likelihood, so the term carries only what the reward
moved and not generic peptide-likeness. **W = 0 recovers `score_pool` exactly**, so the shipped
selection is nested in the sweep and the comparison cannot be rigged.

Sanity check passed first: at W = 0 the harness reproduced the shipped top-100 at **100/100 overlap**
and identical predicted activity 0.5705.

| W | predicted activity | vs shipped | median charge | S1 activity | S2 composition |
|---|---|---|---|---|---|
| 0.00 | 0.5705 | +0.0000 | 5.00 | fail | PASS |
| 0.25 | 0.5517 | **-0.0188** | 4.20 | fail | **fail** |
| 0.50 | 0.5406 | **-0.0299** | 4.15 | fail | **fail** |
| 1.00 | 0.5336 | **-0.0368** | 4.10 | fail | **fail** |
| 2.00 | 0.5059 | **-0.0646** | 4.00 | fail | **fail** |

Every non-zero weight makes the selection **worse**, monotonically in W, and drags median net charge
from 5.00 toward 4.00. The mechanism is visible: the policy's likelihood lift favours sequences
resembling the half-A real AMPs it was fine-tuned on, and those sit at lower charge than the top of
`score_pool`'s ranking, so the term fights the charge axis rather than adding to it.

## Final verdict on the RL route

Closed from both directions.

* **As a generator**: 3 of 4 gates failed, driven by composition drift (charge +2.20 -> +4.00).
* **As a selector**: all gates failed at every weight tested, monotonically worse than W = 0.
* **The learning itself is real**: the likelihood test passed against a shuffled-reward null, and the
  orthogonal margin of +0.0149 at matched charge and hydrophobicity has a CI excluding zero.

A real signal that does not convert into a usable artifact is still a negative. The shipped generator
and the shipped selection rule both stand, and the only surviving improvement from this session is the
FBD-optimised top-100 in `RESULT_FBD_FRONTIER.md`, which touches neither.
