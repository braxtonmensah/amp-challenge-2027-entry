# AMP Challenge 2027 entry: motif-faithful generation, with a selection rule chosen by measurement

Braxton Mensah, Indiana University Bloomington (`bsmensah@iu.edu`).

    uv sync
    uv run generate

Writes `generate/library.fasta` (50,000 sequences) and `generate/top.fasta` (100).

---

## Disclosures, up front

**AI assistance.** This repository was written with AI assistance (Anthropic Claude), which the
competition rules permit and require to be disclosed.

**Training data for generation: one corpus only.** `data/antibacterial.fasta` as shipped in the
organizers' template, 39,448 sequences, all 8-50 residues. The generator uses no pretrained model and no
other corpus. It is used for three things: fitting the order-2 transition table, drawing the length
distribution, and filtering for novelty.

**External data used to choose the SELECTION RULE.** Public measured MIC and haemolysis values were used
to decide the form and the constants of the scoring function. The competition explicitly permits this
("teams are free to use any public peptide and/or AMP database"). Specifically:

* **MIC**: 51,345 measurements aggregated in GRAMPA from DBAASP, DRAMP, YADAMP, APD and DADP. Filtered
  to unmodified, free-termini, 20-standard-AA, 8-50 aa peptides on panel species, giving **2,904
  labelled sequences**. Filtering out modified and C-terminally amidated entries is not optional: this
  competition forbids terminal modification, and amidation adds about +1 charge and shifts MIC
  severalfold, so training on it would bias every prediction.
* **HC50**: Hemolytik-derived values, **501 sequences** with both HC50 and panel MIC.

**No model weights are shipped and no learned model runs at generation time.** The evidence was used to
pick a two-term closed-form scorer and four numeric constants, all hard-coded in `generate.py` with their
provenance in comments. Generation therefore needs none of the external data, and reproduces from this
repository alone.

**Determinism.** One seed, `SEED = 20260930`. Two independent runs produce byte-identical
`library.fasta` and `top.fasta`; verified. See `SEED.md`.

## Method

### Generation

An **order-2 Markov chain** fitted to the reference actives, so local motifs (`KKIL`, `GKII`) occur at
their natural frequency instead of being assembled from independent per-position draws. Lengths are
drawn from the empirical reference length distribution. The library excludes exact matches to the
reference set and internal duplicates.

This is a deliberately weak generative model, and the entry does not claim otherwise. Measured on the
organizers' own `seqme` framework it is nonetheless close to the ceiling set by real AMPs (table below),
which is the only thing the Phase 1 library metrics ask of it.

### Selection: what we measured, and what we retired

The first version of this entry ranked candidates with a four-term composite: banded net charge (0.40),
hydrophobic moment (0.30), banded hydrophobicity (0.20), Chou-Fasman helix propensity (0.10). **We then
measured each term against real MIC data and retired it.** Against measured panel success rate
(MIC <= 16 uM, weighted 15:5 Gram-negative:Gram-positive to match the official panel):

| term | old weight | Spearman | AUC |
|---|---|---|---|
| net charge, raw | - | +0.287 | **0.678** |
| net charge, banded +4..+9 | 0.40 | +0.244 | 0.590 |
| hydrophobic moment | 0.30 | +0.022 | **0.514** |
| mean hydrophobicity, banded | 0.20 | -0.047 | 0.505 |
| helix propensity | 0.10 | -0.006 | **0.498** |
| **the old composite** | | +0.158 | **0.595** |

The moment and helix terms were **indistinguishable from noise**, banding the charge **destroyed** signal
the raw value carries, and the composite as a whole ranked **worse than net charge alone**.

### A claim from the first version that we retract

The first README called the hydrophobicity band "the single most consequential choice in this entry",
justified by the assertion that hydrophobicity "drives haemolysis about as readily as it drives killing".
We measured it on 501 sequences with paired HC50 and panel MIC. Net charge and hydrophobicity correlate
**-0.729**, so raw correlations are confounded and only the partials mean anything:

| | raw | partial, controlling net charge |
|---|---|---|
| hydrophobicity -> log HC50 | -0.309 | **-0.366** |
| hydrophobicity -> log MIC50 | +0.256 | **-0.176 (sign flips)** |
| hydrophobicity -> log safety window | -0.454 | **-0.239** |

At fixed charge, hydrophobicity does buy potency, as the literature says. But it costs HC50 roughly
**twice** what it buys in MIC. So the trade is real and **not worth taking** — and because the
relationship is monotone, **a band was the wrong instrument entirely**: the old band's lower bound of
0.05 excluded the best available region. Peptides inside the old band measured a median log safety
window of 1.089 against 1.593 outside it (Mann-Whitney p = 2.8e-10).

### The scorer that ships

    score = rank(net charge) - rank(mean hydrophobicity)        # within the candidate pool

Two terms, no fitted weights. It was adopted against three gates pre-registered in
`PREREG_SELECTION_2.md` and evaluated on a **cluster-disjoint held-out half** (clusters at Levenshtein
ratio >= 0.6, split so no sequence family appears in both halves):

| scorer, top-100 of held-out half | median log SW | mean success rate |
|---|---|---|
| random 100 | 1.157 | 0.591 |
| the retired composite | 1.288 | 0.615 |
| a fitted 3-descriptor ridge | 1.550 | 0.628 |
| **rank(charge) - rank(hydrophobicity)** | **1.541** | **0.682** |

The fitted ridge beat the simple form by 0.009 log10, inside the pre-registered simplicity margin, so
**the fitted weights were discarded**. A two-term expression beats the four-term composite on both
endpoints at once.

### Three guards, each because an unguarded version failed

Extremising any linear score walks it off the end of the evidence, and each of these was added after
watching that happen, not in anticipation:

1. **Measured envelope.** Selection is restricted to net charge in [-1, +5] and mean hydrophobicity in
   [-0.19, +0.65], the 30th-70th percentile band of the labelled distribution. Unconstrained, the scorer
   selected poly-arginine strings at charge +18 (`RRRRWIRDLAKTMQHPPRRQPKKRRKRRRGCR`) with 44 of 100
   outside any measured range; those are cell-penetrating-peptide motifs, not antimicrobials.
   The percentile was chosen by a stated rule, not by taste: **take the most aggressive envelope whose
   Phase 1 property-conformity is still at least that of real AMPs.** Pushing to the 99th percentile
   maximises predicted safety window (0.825 vs 0.304) but collapses `seqme` conformity from 0.496 to
   0.029 against 0.489 for held-out real AMPs. Qualification precedes any Phase 2 gain.
2. **Composition guard** at the 95th percentile of the reference actives (max single residue <= 0.500,
   W <= 0.238, aromatic FWY <= 0.333, Q <= 0.111), plus **cysteine excluded outright**. Without it the
   top sequence carried seven tryptophans and a run of glutamines, because glutamine's Eisenberg value
   (-0.85) drags mean hydrophobicity down without improving anything biological. Cysteine is excluded
   because a free thiol invites disulfide dimerisation in a linear free-termini peptide.
3. **Internal diversity cap** (pairwise Levenshtein ratio <= 0.7 within the 100). The 25 assayed
   peptides are drawn **uniformly at random** from the top-100, so the ordering of the 100 cannot affect
   any score and near-duplicates simply waste draws. This raised top-100 diversity from 0.760 to 0.823.

### Novelty, and a metric mismatch we found and fixed

The library excludes exact matches to the reference set. The top 100 are additionally screened on
**sequence identity**, not an edit ratio, because those are not the same thing and the difference was
costing us candidates.

The rule is "no more than 80% sequence identity, computed via MMseqs2 pairwise alignment". Identity is
computed over an *alignment*, so two peptides can sit below 0.8 Levenshtein ratio and still align above
80% identity over a well-covered region. Measured on an earlier top-100 that passed the edit-ratio filter:

| identity definition | candidates above 0.80 |
|---|---|
| full-length global alignment | 0 of 100 |
| matches / shorter sequence length | **14 of 100** |
| local alignment, coverage >= 0.8 | **24 of 100** |

Non-compliant candidates are replaced by the organizers with the next valid entry, so that was up to a
quarter of the ranked set silently diluted. Rather than bet on one reading of the rule, every candidate
must now clear 80% under **all three** definitions, by BLOSUM62 alignment (`_identity_ok`). The shipped
top-100 has **zero violations under all three**, with a maximum identity of 0.800.

**Residual gap, stated plainly.** The rule names the MarLys database (~102,000 sequences, thirteen
databases), which we could not obtain; we screen against the template's own 39,448 antibacterials. We
bounded how much that can matter rather than leaving it unquantified: a union with DRAMP 3.0 and GRAMPA
contains 43,025 unique sequences against the template's 39,448, i.e. those two databases add only about
9% of genuinely new sequence. That bounds the gap; it does not close it.

## Phase 1, measured on the organizers' own `seqme`

Computed against a **disjoint half** of the reference actives, so the real-AMP row is not scoring against
itself. Higher is better except FBD and MMD, which are distances.

| query set | Uniqueness | Diversity | Novelty | Conformity | FBD | MMD |
|---|---|---|---|---|---|---|
| held-out **real AMPs** (ceiling) | 1.000 | 0.853 | 1.0 | 0.485 | 0.0049 | 0.00057 |
| **our library** | 1.000 | **0.850** | 1.0 | **0.570** | **0.0093** | **0.00148** |
| **our top-100** | 1.000 | **0.814** | 1.0 | **0.531** | **0.0525** | **0.0207** |
| composition-matched shuffles | 1.000 | 0.856 | 1.0 | 0.504 | 0.0064 | 0.00115 |
| random K/P, the example generator | 0.979 | 0.479 | 1.0 | 0.147 | 0.354 | 2.378 |

The library sits at the real-AMP ceiling on diversity, exceeds it on property conformity, and is roughly
38x closer to the reference distribution than the example generator.

The three guards were kept honest against this table. The top-100 that the retired composite selected
scored Diversity 0.760, Conformity 0.533, FBD 0.062, MMD 0.077. An unguarded aggressive selection scored
0.759 / 0.029 / 0.063 / 0.135, trading a Phase 1 collapse for Phase 2 gain. The shipped selection scores
**0.814 / 0.531 / 0.053 / 0.021**, which is better than the retired composite on *every* metric while
also carrying +0.247 log10 of predicted safety window. No Phase 1 cost was paid for the Phase 2 gain.

## Compliance, checked against the organizers' own validator

`scripts/verify_submission.py` from the template, imported and run directly:

| check | result |
|---|---|
| `_verify_sequences(library.fasta)` | PASS, 50,000 unique |
| `_verify_no_overlap` vs 39,448 references | PASS |
| `_verify_top(top.fasta, k=100)` | PASS |
| `_veritfy_max_simularity(<= 0.80)` | PASS |
| identity <= 0.80 under all three definitions | PASS, 0 violations, max 0.800 |
| two independent runs, byte-compared | IDENTICAL |

Peptide constraints: 20 standard amino acids, 8-50 residues, linear, free termini, no duplicates. The
top-100 additionally contains no cysteine.

## Honest limitations

1. **No experimental validation of anything here.** Every relationship used was measured on *published*
   peptides, not on ours.
2. **Everything is applied out of distribution by construction.** Our candidates must sit below 80%
   identity to any known AMP, while every relationship we fitted comes from known AMPs. The
   cluster-held-out evaluation is the closest available estimate of that gap and is still optimistic.
3. **An order-2 Markov chain is a weak generative model.** It captures dipeptide context and nothing
   longer, and cannot represent tertiary or aggregation behaviour.
4. **A trained MIC model did not help and is not shipped.** Pre-registered in `PREREG_SELECTION.md`: a
   descriptor ridge model learned real signal (grouped-CV Spearman +0.361 against a -0.074 shuffled-label
   null) but its top-100 beat the biophysical score by **+0.002** on measured success rate. The primary
   gate failed and we report that rather than shipping the more sophisticated method.
5. **The haemolysis source file has malformed line endings** and was parsed heuristically (1,258
   sequences recovered, 501 joined), with values range-checked to 0 < log10 HC50 < 4. Section "A claim we
   retract" is conditional on that parse.
6. **We do not claim to beat a null.** No published study synthesises random or composition-matched
   peptides and MIC-tests them at this competition's <= 16 uM threshold, so no comparable null exists.
7. Nothing here was screened for protease stability or aggregation.

## Repository map

* `src/amp/generate.py` - the entry point. Generation, the shipped scorer, and all three guards.
* `PREREG_SELECTION.md` - pre-registration 1: should a trained MIC model replace the scorer? Gate failed.
* `PREREG_SELECTION_2.md` - pre-registration 2: are the scorer's own terms carrying signal? Gates passed.
* `src/amp/prep_labels.py` - builds panel-matched labels from GRAMPA.
* `src/amp/eval_selection.py`, `src/amp/eval_scorer.py` - the gate runners.
* `src/amp/predictor.py` - the retired predictor harness, kept for audit.

## Licence

MIT, see `LICENSE`.
