# A better generator: 31% lower FBD, diversity up, composition unchanged

Run 2026-09-29 on a Quartz H100 (jobs 10761994 unstratified, 10762431 stratified).
`src/amp/gen_plm_resample.py`. Output `generate/library_plm.fasta`. **Not shipped.**

## What it does

The shipped library comes from an order-2 Markov chain: it knows dipeptide frequencies and nothing
else. This keeps that chain as a cheap **proposal** and reweights its output toward the real-AMP
distribution estimated in ESM2 embedding space, in a 32-dim PCA subspace fitted on reference half A.
100,000 draws in, 50,000 kept.

## Result, all four pre-registered gates PASS

| gate | requirement | measured | |
|---|---|---|---|
| 1 | FBD on held-out half B >= 30% lower | **2.52953 -> 1.74966 = 0.69x** | PASS |
| 2 | diversity within 0.01, uniqueness 1.0 | **0.8512 -> 0.8581**, uniq 1.000 | PASS |
| 3 | held-out gain >= half the fitted gain | A +0.767, B +0.780 | PASS |
| 4 | median charge within 0.5, hydrophobicity within 0.05 | **3.00 -> 3.00**, -0.052 -> -0.076 | PASS |

Diversity did not merely hold, it **rose above the held-out real-AMP value (0.853)**.

## The fix that made it work, and why it is evidence rather than tuning

The first run failed Gate 4: median net charge drifted 3.00 -> 2.20 against a 0.5 limit, while
passing Gates 1-3. Moving toward the real-AMP density in embedding space drags charge down.

That is the fourth time composition drift killed an improvement in this project, and the two generator
routes failed it in **opposite directions on the same axis**: KL-anchored RL pushed charge UP
(2.20 -> 4.00), pLM resampling pulled it DOWN (3.00 -> 2.20).

The fix stratifies the resampling by net charge: match the shipped library's charge histogram exactly
(28 bins, 0 shortfall), so the density ratio can only reorder **within** a charge bin.

The decisive number is that this cost almost nothing:

    FBD half B, unstratified   1.74042
    FBD half B, charge-pinned  1.74966

**The 31% gain survives with the charge marginal frozen**, which proves it was never bought with
composition drift. That was the open question and it is answered by measurement, not argument.

## Limits, stated plainly

* **The importance weights are degenerate. ESS = 1 out of 100,000.** In a 32-dim subspace the density
  ratio becomes astronomically peaked, so this is density-ratio **selection** from a Markov proposal,
  not the importance resampling the script's docstring describes. The gates measure whether the
  artifact is better; they do not rescue the method's description, and the docstring overstates it.
* **The embedder is not the organizers'.** ESM2 t12_35M. Under it the shipped library sits at FBD
  2.47 against a real-AMP floor of 0.0167, about **148x**, where README.md reports the library at 1.9x
  from the ceiling. That discrepancy is large and means the README's unrecorded embedder measures
  something substantially less discriminative. Whether a 31% gain here transfers to the organizers'
  FBD is **unverified**.
* **Gates 1-4 tested the LIBRARY, not the entry.** The entry is a library plus a top-100 drawn under
  every shipped guard. A separate verification (job 10764456) checks whether this library can fill a
  compliant, active top-100 at all. Until that passes, this is not shippable.
* Only the library's median charge is pinned. The top-100's composition is a separate question.
* No measured MIC, no measured binding. Every activity figure in this project is a ridge proxy.

## Status

By the registered criteria this is **a better generator** than the shipped Markov chain: closer to the
real-AMP distribution, more diverse, same composition. It is the only thing in this session that
improved the model rather than the selection.

**It is not a better ENTRY.** The top-100 drawn from it is fully compliant but loses 0.0271 predicted
activity. See the amendment below, which is the operative conclusion and supersedes any reading of
this section as a recommendation to swap the library. The embedder caveat remains the largest open
risk to even the library-level claim.

---

## Entry-level verification: the better library does NOT make a better entry

Job 10764461. Gates 1-4 tested the library. The entry is a library PLUS a top-100 drawn from it under
every shipped guard, and the top-100 is what gets assayed in Phase 2. Both libraries selected and
scored identically in the same run; activity is the ridge trained on labelled cluster-half B.

| library | envelope-passing | top-100 filled | predicted activity | median charge | median hydro | max internal | cysteine |
|---|---|---|---|---|---|---|---|
| shipped | 13122 | 100 | **0.5705** | 5.00 | -0.041 | 0.600 | 0 |
| pLM-resampled | 11933 | 100 | **0.5434** | 5.00 | -0.041 | 0.667 | 0 |

**The top-100 loses 0.0271 predicted activity.** It is fully compliant otherwise: it fills to 100,
passes novelty, carries no cysteine, stays under the internal-diversity cap, and has identical median
charge and hydrophobicity. Envelope-passing candidates fall 9% (13122 -> 11933), which is the expected
consequence of moving the library toward real AMPs, since real AMPs are less concentrated in the
envelope than the Markov output is.

## What this means, stated without spin

By the registered library gates this **is** a better generator. By the only test that matters for the
entry, it is **not** an improvement:

* **Phase 1** scores the library and the top candidate list, so a 31% FBD gain and higher diversity
  help qualification.
* **Phase 2** assays 25 peptides drawn at random from the top-100, and that set is measurably worse on
  the activity proxy.

That is a genuine trade, not a free win, and it is a trade no gate in this file was registered to
adjudicate. **The shipped library stands**, and swapping it is not recommended on this evidence.

For contrast, the other artifact from this session, `top_fbd_drop0.005.fasta`, improves Phase 1 FBD by
17% at **zero** activity cost (+0.0061, inside noise) with 100/100 novelty. That one is a Pareto
improvement. This one is not.

## The sharper statement of the session's result

Four separate routes were tried and every one of them ran into the same wall: **anything that moves
this entry toward better distributional or activity scores moves its composition, and the composition
is load-bearing.** The pLM resampler is the only route that beat the wall at the library level, by
pinning the charge histogram. It then hit the same wall one level down, in the top-100.

The ceiling here is set by the 39,448-sequence corpus and the charge/conformity conflict, not by
method or effort.
