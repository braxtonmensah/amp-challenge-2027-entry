# Pre-registration 2: revising the scoring function itself

`PREREG_SELECTION.md` stays in place and is not amended. That one asked whether a trained MIC model
should *replace* the biophysical score for selection; its primary gate failed and we recorded that. This
one asks a different and, as it turns out, more important question: **are the four terms of the
biophysical score itself carrying signal, and is the band trade real?**

Written before any revised scorer was fitted or tested. Author: Braxton Mensah (`bsmensah@iu.edu`).
Method developed with AI assistance (Anthropic Claude), disclosed per competition rules.

---

## 1. What prompted this, measured on 2,904 panel-matched unmodified sequences

Per-term signal against measured panel success rate (MIC <= 16 uM, 15:5 Gram-neg:Gram-pos weighting):

| term | weight in shipped `score_one` | Spearman vs success rate | AUC |
|---|---|---|---|
| net charge, raw | - | +0.287 | **0.678** |
| net charge, **banded +4..+9** | 0.40 | +0.244 | 0.590 |
| hydrophobic moment | 0.30 | +0.022 | **0.514** |
| mean hydrophobicity, banded | 0.20 | -0.047 | 0.505 |
| Chou-Fasman helix propensity | 0.10 | -0.006 | **0.498** |
| length (not used) | - | +0.202 | 0.612 |
| **composite `score_one`** | | +0.158 | **0.595** |

Two facts follow. The 0.30-weighted hydrophobic-moment term and the 0.10-weighted helix term are
**indistinguishable from noise**, and banding the charge **destroys** signal the raw value has
(0.678 -> 0.590). The composite therefore ranks *worse than net charge alone*.

## 2. The band trade, measured for the first time on 501 sequences with paired HC50 and panel MIC

The shipped README calls the hydrophobicity band "the single most consequential choice in this entry",
justified by "raw hydrophobicity drives haemolysis about as readily as it drives killing". That claim
was asserted from the literature and never measured. Measured:

| | raw rho | partial rho, controlling net charge |
|---|---|---|
| hydrophobicity -> log HC50 | -0.309 | **-0.366** |
| hydrophobicity -> log MIC50 | +0.256 | **-0.176 (sign flips)** |
| hydrophobicity -> log SW | -0.454 | **-0.239** |
| net charge -> log SW | +0.419 | +0.144 (controlling hydrophobicity) |

charge and hydrophobicity correlate **-0.729**, so the raw numbers are confounded and the partials are
the honest ones. At fixed charge, hydrophobicity *does* buy potency, as the literature says. But it
costs HC50 roughly **twice** what it buys in MIC. So the trade is real and simply not worth taking.

**The band is the wrong instrument.** A band rewards the interior of 0.05-0.45 and penalises both tails
equally, but the relationship is monotone: lower hydrophobicity is better for the safety window. Our
band's lower bound of 0.05 actively excludes the best region. In-band peptides measure a median
log SW of 1.089 against 1.593 out of band (Mann-Whitney p = 2.8e-10).

## 3. Hypotheses

H2. A scorer that (a) uses net charge **unbanded**, (b) **penalises** hydrophobicity instead of banding
it, and (c) drops the moment and helix terms, selects 100 peptides with a better measured safety window
than the shipped `score_one`, without a material loss of measured success rate.

H3 (the simplicity control). **Net charge alone** does as well as any fitted combination. If H3 holds,
ship net charge with a hydrophobicity penalty and nothing else, because unnecessary terms are how a
scoring function acquires the noise documented in section 1.

## 4. Protocol, fixed now

* Sequences are clustered by Levenshtein ratio >= 0.6, single linkage, and the **clusters** are split
  into disjoint halves A and B. No sequence family appears in both.
* Weights for the revised scorer are fitted on **A only**, by ridge on rank-standardised descriptors
  (net charge, mean hydrophobicity, length) against measured log SW and against success rate.
* All scorers are then evaluated on **B only**, which is never used for fitting.
* Compared on B: shipped `score_one`; revised fitted scorer; net charge alone; and a random-100 draw
  for scale.
* Reported: measured mean success rate and measured median log SW of each selected 100.

## 5. Gates, fixed now

**Gate A (safety window).** The revised scorer must beat `score_one` on measured median log SW of the
selected 100 by **>= 0.15 log10** (a ~1.4x window improvement) on held-out half B.

**Gate B (do no harm).** It must not lose more than **0.03** absolute measured success rate against
`score_one`. The Optimal Selectivity category excludes peptides inactive on every strain, so buying
window by sacrificing potency can score zero.

**Gate C (simplicity).** If net charge alone is within **0.05 log10** of the fitted scorer on Gate A,
the fitted weights are discarded and the simpler scorer ships.

**If Gate A fails, `score_one` ships unchanged** and section 2 is reported as a negative result about
our own design rather than acted on.

## 6. Abort and honesty conditions

* Fewer than 150 paired HC50 + MIC sequences in half B -> report the split as underpowered, do not
  choose a scorer on it.
* The haemolysis source file (`hemolysis_clean.csv`) has malformed line endings and was parsed
  heuristically; 1,258 sequences recovered, 501 joined to panel MIC. Any result here is conditional on
  that parse being right, and the parse is checked by range-sanity on log10 HC50 (0 to 4).
* All of section 1 and 2 is measured on **known, published AMPs**. Our candidates must sit below 80%
  identity to any known AMP, so every relationship here is applied out of distribution by construction.
  This is the main reason the gates are set on held-out measured outcomes rather than on correlations.
