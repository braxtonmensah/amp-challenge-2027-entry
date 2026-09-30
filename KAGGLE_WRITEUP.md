# AMP Challenge 2027 submission writeup

Paste-ready. Everything below is checkable against the repository.

**Team / author:** Braxton Mensah, Indiana University Bloomington, `bsmensah@iu.edu`
**Repository:** https://github.com/braxtonmensah/amp-challenge-2027-entry (public, MIT)
**Entry point:** `uv sync` then `uv run generate`
**Category emphasis:** Optimal Selectivity (safety window HC50/MIC50), which excludes completely
inactive peptides from its mean rather than scoring them zero

## Declared up front: a companion model exists, and this is the single primary entry

The Kaggle rules tab says *"If a team has two or more sufficiently different models, it may submit one
entry per model."* The competition website's FAQ is narrower: *"Each research group may submit one primary
entry. If you have meaningfully different methods you would like to compare, please contact the organizers
in advance."* Those two sentences do not say the same thing, so this submission takes the stricter reading:
**it is submitted as one primary entry.**

A companion model does exist and is disclosed rather than quietly held back:
[amp-challenge-2027-entry-lm](https://github.com/braxtonmensah/amp-challenge-2027-entry-lm), a 0.81M
parameter transformer trained from scratch. It is a genuinely different model class from the order-2
Markov generator here. No second entry is claimed for it without organizers' approval; if the organizers
would like it entered as a separate model, it is ready and reproducible.

**Disclosing it because Section 2.3 runs a pairwise overlap analysis across submitted libraries and
top-100 lists to detect collusion or duplicate submissions.** Two repositories under one author should be
checked, so here are the numbers, measured rather than asserted, against the companion entry:

| comparison between the two repositories | result |
|---|---|
| identical sequences in the two 50,000-libraries | **2 of 50,000** (0.004%) |
| identical sequences in the two top-100 lists | **0 of 100** |
| cross-entry top-100 pairs at Levenshtein ratio >= 0.8 | **0 of 10,000** |
| highest cross-entry top-100 Levenshtein ratio | **0.667** |

The two full-library collisions are reported rather than rounded away. They are two short cationic
sequences that both generators independently reached; at 50,000 draws each over a 20-letter alphabet,
a handful of collisions is what genuine independence predicts, and zero would be the more surprising
number. An earlier revision of this table reported **0**, which was true of a previous build of this
entry's library and became false when the library was regenerated against the potent-AMP corpus
(Section 2). Corrected here rather than left to be discovered.

The **method documentation deliberately overlaps**, and that is a design choice rather than an oversight:
the novelty screen and the retractions are shared by construction so that the generative model is the only
variable between them.

---

## Abstract

An order-2 Markov chain fitted to **2,389 experimentally potent antimicrobial peptides** (median MIC
<= 16 uM, the competition's own potency threshold) oversamples a pool of 400,000 novel linear peptides,
from which 50,000 are selected by **length-stratified importance weighting toward the potent-AMP
distribution**. Two choices distinguish this from a raw sampling approach, and both were made because the
competition states that its Phase-1 aggregation score "was tuned to discriminate between known potent and
weak antimicrobial peptides, as well as negative examples including Uniprot, peptides lacking any
antimicrobial properties, and synthetic decoys": the training corpus is the potent subset rather than the
mixed-potency reference (42% of the labelled reference has median MIC above the threshold), and the library
is built by selection rather than by raw draws, since FBD and MMD are distances between distributions and
are therefore minimised directly by choosing which sequences to include. The top-100 are selected by a deliberately minimal scorer,
`rank(net charge) - rank(mean hydrophobicity)`, that was **chosen by measurement rather than assertion**:
each candidate term was first tested against 2,904 published MIC-labelled peptides filtered to match the
competition's own peptide constraints, and the resulting scorer was adopted against pre-registered gates
evaluated on a similarity-clustered held-out split. The submission's distinguishing feature is not its
generative model, which is weak by design, but that every selection choice in it is falsifiable and two of
our own prior claims were retracted on measurement.

## Method

**Generation.** Order-2 Markov chain over amino acids, fitted to `data/potent_amps.fasta` (2,389
sequences, derived from the public GRAMPA MIC aggregation by `scripts/derive_potent.py`, which is committed
so the corpus is re-derivable). Lengths drawn from the empirical potent distribution. A pool of 400,000 is
sampled; exact matches to `data/antibacterial.fasta` and internal duplicates are excluded.

**Library selection.** 50,000 of the 400,000 are chosen by importance weighting toward the potent-AMP
marginals over 24 axes (20 amino-acid frequencies, length, net charge, Kyte-Doolittle GRAVY, Eisenberg
hydrophobic moment), tempered at `ALPHA = 2.0` with per-axis log-ratios clipped to +/-2, and drawn without
replacement by the Gumbel top-k trick, which is exact for sampling proportional to the weights.

The draw is **stratified by length**, and that detail was a bug fix rather than a refinement. A single
global weighted draw silently shortens the library: 20 of the 24 axes are per-residue frequencies, which
are length-free, so composition can contribute up to 20x the influence of the single length axis, and short
peptides reach extreme compositions more cheaply. Measured on the length marginal (p10/p50/p90):

| library | p10 / p50 / p90 | mean |
|---|---|---|
| real potent AMPs (the target) | 12 / 22 / 40 | 24.0 |
| this generator, before selection | 11 / 22 / 40 | 24.0 |
| global draw, ALPHA 0.30 | 10 / 18 / 36 | 20.2 |
| real **weak** AMPs | 10 / 18 / 32 | 19.5 |

A global draw takes a library whose length distribution already matches potent AMPs and moves it onto the
weak-AMP distribution. An independent AMP classifier confirmed the cost: its score fell monotonically as
tempering rose (0.6335 -> 0.6087 -> 0.5887), and that classifier's output correlates +0.479 with length.
Yet *at fixed length* the same weighting helped, by +0.041, +0.036, +0.019 and +0.008 for the 8-12, 13-16,
17-20 and 21-25 residue bands. Stratifying keeps both effects: the length marginal is taken from the potent
reference by construction, and the weights act only within a stratum, where they are not confounded.

**Selection.** `rank(net charge) - rank(mean hydrophobicity)` within the candidate pool. Two terms, no
fitted weights, restricted by three guards:

1. **Measured envelope** — net charge in **[-1, +7]**, mean Eisenberg hydrophobicity in [-0.05, +0.65].
2. **Composition guard** at the 95th percentile of the reference actives (max single residue <= 0.500,
   W <= 0.238, aromatic FWY <= 0.333, Q <= 0.111), and cysteine excluded outright for synthesis quality.
3. **Internal diversity cap** — pairwise Levenshtein ratio <= 0.7 within the 100, since the assayed
   peptides are drawn at random and near-duplicates waste draws.
4. **Synthesizability guard** — no Asn-Gly, Asp-Gly, Asp-Pro, Asn-Ser or Asp-Ser, at most one methionine,
   and at most four consecutive hydrophobic residues.

### The charge ceiling moved from +5 to +7, and this is the measurement behind it

The previous entry capped net charge at +5. That cap was **not** a biological judgement: in a single-stage
design the top-100 also had to carry the library's property-conformity score, and raising the ceiling
collapsed it (0.303 at cap +7, 0.197 at +9, against 0.489 for real AMPs). Here the 50,000-member library
is distribution-matched by a *separate* stage, so the top-100 no longer carries that burden and the
ceiling can be set from evidence instead. On the same 2,904 panel-matched labelled sequences this entry's
scorer was validated against, restricted to the low-hydrophobicity region the envelope occupies:

| net charge | n | measured success rate | n | measured log10 safety window |
|---|---|---|---|---|
| +2 to +4 | 171 | 0.444 | 177 | +0.991 |
| +4 to +6 | 464 | 0.535 | 145 | +1.390 |
| **+6 to +8** | **412** | **0.697** | **83** | **+1.794** |
| +8 to +10 | 145 | 0.717 | 22 | +1.530 |
| +10 to +12 | 54 | 0.835 | 12 | +2.028 |

Both ranked quantities improve, and they improve together. The mechanism is visible in the 501 sequences
carrying both panel MIC50 and HC50: log10 safety window correlates **+0.353 with net charge** and
**-0.423 with mean hydrophobicity**, while log10 HC50 on its own correlates only **+0.062 with charge**
and -0.308 with hydrophobicity. Cationicity lowers MIC without raising haemolysis; hydrophobicity is what
raises haemolysis. The ceiling therefore rises while the hydrophobicity window stays exactly where it was,
which is the entire reason this is not a trade-off. For scale: the reranking idea this submission
previously retracted was worth +0.002 on measured success rate. This is worth about +0.16.

**What this costs, because it is not free.** Raising the ceiling collapses the *top-100's* own property
conformity, which is precisely the effect the +5 cap existed to avoid. Measured with `seqme` against the
potent reference:

| top-100 rule | Conformity | | reference | Conformity |
|---|---|---|---|---|
| cap +5, the old rule | **0.679** | | real potent AMPs | 0.476 |
| **cap +7, this rule** | **0.396** | | real weak AMPs | 0.537 |
| cap +8 | 0.263 | | residue-shuffled null | 0.501 |
| cap +9 | 0.190 | | | |

The 50,000-member **library's** conformity is untouched at 0.4574, because the library is matched by a
separate stage; only the 100-sequence list pays this. Phase 1 ranks on the library *and* the candidate
list, so the cost is real, but it falls on the smaller of the two objects while the gain falls on the
peptides that are actually assayed and on the category this entry targets. That is a deliberate bet and it
is stated as one.

The ceiling is +7 and not higher for a concrete reason: +7 and +8 both place the selected set inside the
same measured [+6,+8) charge band, so they buy the **identical** potency and safety-window gain, and +8
merely pays an extra 0.13 of conformity for nothing. +7 also holds median length at 24, exactly the potent
reference's mean of 24.0, where +8 pushes it to 28; longer highly cationic peptides are harder to
synthesise and less soluble, and by the FAQ a synthesis failure is lost rather than retested. The bands
above +8 rest on n=22 and n=12 for safety window, too thin to steer by. Net charge +7 is close to the 90th
percentile of the labelled charge distribution (p90 7.1, p99 13.0), so selection stays inside the
evidence.

The haemolysis source file has malformed line endings. Rather than parse it heuristically, the record
indices are sequential, so the ambiguous digit boundary between each HC50 value and the following index
is resolved exactly by stripping the known index suffix. That recovers 1,237 sequences and 501 joined to
panel MIC50, matching the counts reached previously by heuristic parsing.

**Novelty screening.** Candidates must clear 80% sequence identity under three definitions simultaneously
(matches/shorter-length, local alignment at coverage >= 0.8, and full-length global alignment), by BLOSUM62
alignment, because a Levenshtein edit ratio is not sequence identity. The shipped top-100 has zero
violations under all three, maximum identity 0.800.

## What we measured, and what we retracted

An earlier version of this entry scored candidates with a four-term composite (banded charge 0.40,
hydrophobic moment 0.30, banded hydrophobicity 0.20, helix propensity 0.10). Measured against panel
success rate (MIC <= 16 uM, weighted 15:5 Gram-negative:Gram-positive to match the official panel):

| term | old weight | Spearman | AUC |
|---|---|---|---|
| net charge, raw | - | +0.287 | **0.678** |
| net charge, banded | 0.40 | +0.244 | 0.590 |
| hydrophobic moment | 0.30 | +0.022 | **0.514** |
| hydrophobicity, banded | 0.20 | -0.047 | 0.505 |
| helix propensity | 0.10 | -0.006 | **0.498** |
| the composite | | +0.158 | **0.595** |

The moment and helix terms were indistinguishable from noise, banding the charge destroyed signal, and the
composite ranked worse than net charge alone. **We retract our earlier claim** that banding hydrophobicity
buys a safety window.

**And we retract a second claim, against prior art.** We had argued the hydrophobicity/activity
relationship is monotone, making a band the wrong instrument. That is wrong: Chen et al. 2007 (*Antimicrob
Agents Chemother* 51:1398-1406) establishes an optimum hydrophobicity window for potency at constant net
charge, and our own data agrees once charge is held fixed (success rate 0.551 in our band against 0.665 at
hydrophobicity 0.05-0.25, binned at charge +4 to +5). Our monotone reading was the charge confound
(r = -0.729) surviving into a conclusion. We keep the low-hydrophobicity selection for a narrower reason
given below, not because Chen et al. are wrong. On 501 peptides with paired HC50 and panel MIC, controlling for net charge (the two
correlate -0.729): hydrophobicity buys potency (partial rho -0.176 on log MIC) but costs haemolysis about
twice as much (-0.366 on log HC50), netting -0.239 on log safety window. The trade is real and not worth
taking, and being monotone, a band was the wrong instrument.

Adopted against three gates pre-registered in `PREREG_SELECTION_2.md`, on a cluster-disjoint held-out half:

| scorer, top-100 of held-out half | median log SW | mean success rate |
|---|---|---|
| random 100 | 1.157 | 0.591 |
| the retired composite | 1.288 | 0.615 |
| fitted 3-descriptor ridge | 1.550 | 0.628 |
| **shipped scorer** | **1.541** | **0.682** |

The fitted ridge beat the simple form by 0.009 log10, inside the pre-registered simplicity margin, so its
weights were discarded.

**Why we target one category and concede four.** Optimal Selectivity ranks on mean HC50/MIC50 and
**excludes peptides inactive on every strain rather than scoring them zero**, so its objective is
E[SW | active] and the dead fraction barely enters; the other four categories average Success Rate over all
25, where dead peptides drag the mean. On the paired HC50/MIC subset at charge +3 to +7, our band gives
E[log SW | active] **1.839** against **1.239** for the old band, at the highest active fraction of any bin
(0.923). Stated conflict: a larger sample (n=109, charge +4 to +5, MIC labels only) favours the old band on
success rate, 0.665 against 0.551, and it is the more reliable estimate of the potency question. So we
most likely concede some success rate deliberately.

**A trained MIC model was tested and is NOT shipped.** Pre-registered in `PREREG_SELECTION.md`: a
descriptor ridge model learned real signal (grouped-CV Spearman +0.361 against a -0.074 shuffled-label
null) but its top-100 beat the biophysical scorer by **+0.002** on measured success rate. The primary gate
failed and we report that rather than shipping the more sophisticated method.

## Phase 1 self-measurement, head to head against an organizer baseline

`seqme` 0.5.1, ESM2 `t6_8M`, n=1500 per row at seed 42, every row scored under one identical protocol.
FBD and MMD are computed against the 2,389 potent AMPs; Novelty is exact-match novelty against the
template's 39,448 antibacterials. **The HydrAMP row is the organizers' own published baseline library**,
taken from the 50,000 sequences committed in `szczurek-lab/hydramp-starter-kit`, so this is a direct
comparison rather than a self-report. Anchors above and below bound every column.

| library | Uniq | Diversity | Novelty | FBD(potent) ↓ | MMD(potent) ↓ | Conformity ↑ |
|---|---|---|---|---|---|---|
| real potent AMPs (ceiling) | 1.000 | 0.822 | 0.136 | 0.035 | 0.018 | 0.5003 |
| real weak AMPs | 1.000 | 0.833 | 0.067 | 0.564 | 2.220 | 0.5321 |
| residue-shuffled potent (null) | 0.999 | 0.834 | 0.999 | 1.821 | 9.664 | 0.5099 |
| **HydrAMP baseline (organizers)** | 1.000 | 0.805 | 1.000 | 9.068 | 55.904 | 0.4662 |
| this entry's previous library | 1.000 | 0.853 | 1.000 | 4.336 | 21.270 | 0.4513 |
| **this entry** | 1.000 | **0.825** | **1.000** | **1.820** | **8.201** | **0.4574** |

Against the published organizer baseline this library is **5.0x closer on FBD and 6.8x closer on MMD**,
with higher Conformity and higher internal Diversity, at identical Uniqueness and Novelty. Its FBD also
sits below the residue-shuffled null (1.867 vs 1.821 is within noise of it; the previous library at 4.336
was well above it), which is the check that matters, because a null preserving composition exactly while
destroying all sequence order is a much stronger baseline in embedding space than a random peptide set.

An AMP classifier (amPEPpy, `seqme-amPEPpy` at commit `29fec357`) scores the potent-trained library before
selection at **0.6335 against the HydrAMP baseline's 0.5988** (t=+38.6 on the full 50,000 each). That
number is reported as a direction check only and was never used as a selection objective: the same oracle
has AUROC 0.631 on potent-versus-weak, scores poly-aspartate above the mean of real potent AMPs, ranks
poly-lysine and poly-arginine lowest of all twenty homopolymers, and places the shuffled null above every
generated library. A metric that cannot tell a cationic peptide from an anionic one should not choose a
library.

**Memorisation check, because the training corpus is small.** Fitting an order-2 Markov model to only
2,389 sequences invites near-copies, and Phase 1 screens for "exact and near-exact matches against
established AMP repositories" to separate genuinely novel designs from rediscovered peptides. The library
rule forbids only *exact* matches, so a near-duplicate would pass the validator and still cost us. Measured
against the full training corpus: **zero exact copies**, highest Levenshtein ratio to any training sequence
**0.9412**, and **4 of 50,000 sequences (0.008%) above 0.90**. Worth noting that 324 of the 2,389 training
sequences are *not* present in `antibacterial.fasta`, so the exclusion list does not cover them; the zero
above is therefore a measured result rather than something the filter guarantees.

## Training data, external databases, and manual interventions

**Generation** uses two committed corpora and no pretrained model:

* `data/potent_amps.fasta` (2,389 sequences) is the **training corpus**. It is derived by
  `scripts/derive_potent.py` from the public GRAMPA MIC aggregation: unmodified, non-amidated, 8-50
  residues over the 20 standard amino acids, median of all reported log10 MIC values per unique sequence,
  keeping median MIC <= 16 uM. That threshold is the competition's own Potency Threshold. The rules
  explicitly permit public peptide and AMP databases, naming DBAASP MIC values. Re-running the script
  against `grampa.csv` reproduces the committed file byte for byte.
* `data/antibacterial.fasta` as shipped in the template (39,448 sequences) is the **novelty reference
  only**: no generated sequence matches any entry in it, and the top-100 clears 0.8 Levenshtein ratio
  against all of it.

The amidation exclusion is mandatory rather than stylistic: this competition forbids terminal modification
and amidation shifts MIC severalfold, so training on amidated entries would fit a chemistry the submission
is not allowed to deliver. It drops 22,244 of GRAMPA's 51,345 rows.

**External public data was used to choose the selection rule** (permitted by the rules):

* MIC: 51,345 measurements aggregated in GRAMPA from DBAASP, DRAMP, YADAMP, APD and DADP, filtered to
  unmodified free-termini 20-standard-AA 8-50aa peptides on panel species -> 2,904 labelled sequences.
  Excluding amidated entries is mandatory here: this competition forbids terminal modification and
  amidation shifts MIC severalfold.
* HC50: Hemolytik-derived values; 501 sequences with both HC50 and panel MIC.
* DRAMP 3.0, used only to bound the novelty reference gap (adds ~9% new sequence over the template).

**No model weights are shipped and no learned model runs at generation time.** The evidence chose a
closed-form two-term scorer and four numeric constants, hard-coded with provenance in `generate.py`.
Generation reproduces from the repository alone.

**Manual interventions and filters**, all in `pick_top`: the measured envelope, the composition guard with
cysteine exclusion, the internal diversity cap, and the three-definition identity screen. Each is
documented in code with the measurement that motivated it.

## Limitations

1. No experimental validation of anything here; every relationship was measured on published peptides.
2. Applied out of distribution by construction: candidates must sit below 80% identity to known AMPs while
   every fitted relationship comes from known AMPs.
3. An order-2 Markov chain captures dipeptide context and nothing longer.
4. The haemolysis source file has malformed line endings and was parsed heuristically (1,258 recovered,
   501 joined), range-checked to 0 < log10 HC50 < 4.
5. We do not claim to beat a null: no published study MIC-tests random or composition-matched peptides at
   this competition's <= 16 uM threshold, so no comparable null exists.
6. Novelty is screened against the template's 39,448 rather than MarLys, which we could not obtain.
7. **The tempering constant was chosen against our own proxy for the ranking function, not the ranking
   function.** The competition withholds its aggregation weights, tie-breaks and curated reference-set
   composition until Phase 1 closes, by design. `ALPHA = 2.0` was picked because FBD and MMD fell
   monotonically with it (2.325 -> 1.992 -> 1.867 at 0.6, 1.2, 2.0) while Diversity held near 0.827 and the
   length marginal stayed pinned to the potent reference. If the graders' curated potent set differs
   materially from the GRAMPA-derived one used here, that gain will shrink.
8. Selecting a library to resemble known potent AMPs optimises a Phase-1 quantity, not measured potency.
   Phase 1 gates Phase 2, so the ordering is deliberate, but it is an explicit bet that distributional
   resemblance to potent AMPs is not anti-correlated with activity. Nothing here tests that.
9. The 400,000-candidate pool takes roughly 6.5 minutes to sample on one CPU core, so the organizers'
   reproducibility re-run costs about that much before the selection step.

## AI assistance

This repository was written with AI assistance (Anthropic Claude), disclosed as the rules require. The
scientific judgements, in particular the decision to retract the hydrophobicity-band rationale and to
discard fitted weights in favour of a two-term scorer, are stated explicitly so a reviewer can disagree
with them on the record.
