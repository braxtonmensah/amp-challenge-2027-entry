# AMP Challenge 2027 submission writeup

Paste-ready. Everything below is checkable against the repository.

**Team / author:** Braxton Mensah, Indiana University Bloomington, `bsmensah@iu.edu`
**Repository:** https://github.com/braxtonmensah/amp-challenge-2027-entry (public, MIT)
**Entry point:** `uv sync` then `uv run generate`
**Category emphasis:** Optimal Selectivity (safety window HC50/MIC50)

---

## Abstract

An order-2 Markov chain fitted to the competition's 39,448 reference antibacterials generates a library
of 50,000 novel linear peptides, preserving natural dipeptide motif frequencies and the empirical length
distribution. The top-100 are selected by a deliberately minimal scorer,
`rank(net charge) - rank(mean hydrophobicity)`, that was **chosen by measurement rather than assertion**:
each candidate term was first tested against 2,904 published MIC-labelled peptides filtered to match the
competition's own peptide constraints, and the resulting scorer was adopted against pre-registered gates
evaluated on a similarity-clustered held-out split. The submission's distinguishing feature is not its
generative model, which is weak by design, but that every selection choice in it is falsifiable and two of
our own prior claims were retracted on measurement.

## Method

**Generation.** Order-2 Markov chain over amino acids, fitted to `data/antibacterial.fasta` only. Lengths
drawn from the empirical reference distribution. Exact reference matches and internal duplicates excluded.
Single seed (`20260930`); two runs are byte-identical.

**Selection.** `rank(net charge) - rank(mean hydrophobicity)` within the candidate pool. Two terms, no
fitted weights, restricted by three guards:

1. **Measured envelope** — net charge in [-1, +5], mean Eisenberg hydrophobicity in [-0.19, +0.65], the
   30th-70th percentile band of the labelled distribution.
2. **Composition guard** at the 95th percentile of the reference actives (max single residue <= 0.500,
   W <= 0.238, aromatic FWY <= 0.333, Q <= 0.111), and cysteine excluded outright for synthesis quality.
3. **Internal diversity cap** — pairwise Levenshtein ratio <= 0.7 within the 100, since the 25 assayed
   peptides are drawn uniformly at random and near-duplicates waste draws.

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

## Phase 1 self-measurement (`seqme`, against a disjoint reference half)

| query set | Uniqueness | Diversity | Novelty | Conformity | FBD | MMD |
|---|---|---|---|---|---|---|
| held-out real AMPs (ceiling) | 1.000 | 0.853 | 1.0 | 0.485 | 0.0049 | 0.00057 |
| **our library** | 1.000 | **0.850** | 1.0 | **0.570** | **0.0093** | **0.00148** |
| **our top-100** | 1.000 | **0.819** | 1.0 | **0.539** | **0.0533** | **0.0163** |
| composition-matched shuffles | 1.000 | 0.856 | 1.0 | 0.504 | 0.0064 | 0.00115 |
| random K/P (example generator) | 0.979 | 0.479 | 1.0 | 0.147 | 0.354 | 2.378 |

## Training data, external databases, and manual interventions

**Generation** uses one corpus: `data/antibacterial.fasta` as shipped in the template (39,448 sequences).
No pretrained model, no other corpus.

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

## AI assistance

This repository was written with AI assistance (Anthropic Claude), disclosed as the rules require. The
scientific judgements, in particular the decision to retract the hydrophobicity-band rationale and to
discard fitted weights in favour of a two-term scorer, are stated explicitly so a reviewer can disagree
with them on the record.
