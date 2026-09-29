# A compliant top-100 with 17% lower FBD at no measured activity cost

Run 2026-09-29. Scripts `src/amp/opt_fbd.py` (failed, kept) and `src/amp/opt_fbd2.py` (this result).
Candidate set `generate/top_fbd_drop0.005.fasta`. **Not shipped. This is a measurement and an option.**

## The gap this attacks

`pick_top` ranks by `score_pool` and then filters. Nothing anywhere in the pipeline optimises the
distributional distance of the selected 100, so whatever FBD the top-100 lands on is an accident of
the ranking. The shipped README reports the top-100 at FBD 0.0529 against 0.0093 for the library it
came from, a 5.7x penalty paid at the selection step.

## What was measured

Frontier over the maximum allowed drop in mean `score_pool` percentile. FBD is optimised against
reference half A and reported on **held-out half B**. Activity is predicted by a ridge trained on
cluster-half B of the labelled panel data. Baseline top-100: FBD(B) 4.0682, predicted activity 0.5705.

| max drop in rank | FBD(A) | FBD(B) | FBD gain | predicted activity | swaps |
|---|---|---|---|---|---|
| 0.000 | 3.9815 | 4.0548 | 0% | +0.0026 | 3 |
| **0.005** | **3.3114** | **3.3792** | **17%** | **+0.0061** | 47 |
| 0.010 | 3.1169 | 3.1838 | 22% | -0.0065 | 77 |
| 0.020 | 2.8231 | 2.8920 | 29% | -0.0192 | 128 |
| 0.050 | 2.2749 | 2.3298 | 43% | -0.0183 | 204 |
| 0.200 | 1.6344 | 1.6802 | 59% | -0.0968 | 368 |

FBD(A) and FBD(B) track at every point, so the gain is not fitted to the reference sample.

**The recommended point is drop 0.005: 17% lower FBD, activity unchanged.** The +0.0061 is inside the
noise of a ridge proxy and is NOT claimed as an improvement; the claim is that it costs nothing.

## Compliance, verified rather than assumed

| set | novelty | max internal | cysteine | median charge | median hydrophobicity |
|---|---|---|---|---|---|
| shipped `top.fasta` | **100/100** | 0.600 | 0 | 5.00 | -0.041 |
| `top_fbd_drop0.005` | **100/100** | 0.600 | 0 | 5.00 | -0.039 |

Identical. Composition does not drift, so this is not the envelope violation that killed H-CAT.

## Two errors on the way here, both recorded

**1. The first run's gate was unsatisfiable by construction.** `opt_fbd.py` required mean `score_pool`
percentile not to fall, but the shipped top-100 IS the argmax of that quantity, so 14 of 4000 proposed
swaps were legal and it reported a 5% gain as a FAIL. The search was forbidden, not infeasible. The
finding that survives: under strict no-regression only 3 legal swaps exist and the gain is 0%.

**2. The first frontier was non-compliant.** The optimiser enforced internal diversity but never
re-ran the three-definition identity screen. Those sets passed **86 of 100** where the shipped top-100
passes 100 of 100, and the organizers silently replace non-compliant entries, so part of the headline
23% gain was fictitious. Adding a lazy memoised novelty screen (718 screens performed) cut the gain
from 23% to **17%**, which is the number reported above.

## Limits, stated plainly

* **The embedder is not the organizers'.** FBD is measured here with ESM2 t12_35M. The README's
  absolute figures were produced ad hoc and the embedder was never recorded, so they are not
  comparable and no comparison to them is made. Whether a 17% gain in this embedding space transfers
  to the organizers' exact FBD is **unverified**. The mechanism is embedder-agnostic distribution
  matching, which is the reason to expect direction to hold, but that is an argument, not a
  measurement.
* Activity is a ridge proxy applied to generated sequences, which are outside its training
  distribution. It is used identically on both sets, so the comparison is fair, but the absolute
  numbers mean little.
* The cost is 0.5 percentile points of `score_pool` rank. Real, small, and stated.
* Nothing here is measured binding or measured MIC.

## What it is worth

A Phase 1 improvement that is free in Phase 2 terms, on the only metric in this pipeline nobody had
optimised. It is an option to take or decline, not a change that has been made.
