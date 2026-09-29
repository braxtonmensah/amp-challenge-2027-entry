# AMP Challenge 2027 — Entry 2 of 2: transformer language model

Paste-ready. This is a **second, separate entry** under the competition rule: *"If a team has two or more
sufficiently different models, it may submit one entry per model."* A 4-layer autoregressive transformer
and an order-2 Markov chain are different models. Entry 1 is the Markov submission.

**Author:** Braxton Mensah, Indiana University Bloomington, `bsmensah@iu.edu`
**Repository:** https://github.com/braxtonmensah/amp-challenge-2027-entry (public, MIT)
**Entry point:** `uv sync` then `uv run generate_lm`
**Category emphasis:** Optimal Selectivity (safety window HC50/MIC50)

---

## Abstract

A decoder-only transformer language model trained from scratch on the competition's own 39,448 reference
antibacterials generates a library of 50,000 novel linear peptides. The top-100 are selected by
`rank(net charge) - rank(mean hydrophobicity)`, a two-term rule chosen by measurement against published
MIC and HC50 data and applied under three guards (a measured property envelope, a composition guard at
the reference set's 95th percentile with cysteine excluded, and an internal diversity cap). The model is
deliberately small, 0.81M parameters, because a larger one memorised the corpus that novelty is scored
against. Trained weights ship in the repository.

## Why a language model, and why this one

The companion entry generates with an order-2 Markov chain, which conditions each residue on exactly the
previous two. It cannot represent helical periodicity or long-range charge patterning — the properties
that distinguish an amphipathic helix from a random cationic string. A transformer can.

**Density-model comparison**, same corpus, same held-out split, per-residue perplexity:

| model | held-out perplexity |
|---|---|
| uniform random over 20 residues | 20.00 |
| order-2 Markov chain | 14.79 |
| **this transformer** | **7.72** |

**Model selection was measured, not assumed.** Two sizes were trained. The larger (d=256, 8 heads,
6 layers, 4.76M parameters) reached a *better* best validation perplexity, 6.93 against 7.72 — but its
train loss fell to 1.21 while validation rose to 2.10, an 0.88 gap with validation degrading after its
peak. That is memorisation, and this competition scores novelty against the very corpus it memorised.
Its library measured a **worse MMD (0.00156) than even the Markov chain**. Bigger lost, so the shipped
model is the small one.

Architecture: 4 layers, 4 heads, d_model 128, learned positional embeddings, dropout 0.1, 0.81M
parameters. AdamW with OneCycle, 40 epochs, batch 256. Trained on one NVIDIA H100.

## Phase 1 self-measurement (`seqme`, against a disjoint half of the reference set)

| library | Diversity | Novelty | Conformity | FBD | MMD |
|---|---|---|---|---|---|
| held-out **real AMPs** (ceiling) | 0.856 | 1.0 | 0.489 | 0.00486 | 0.00050 |
| order-2 Markov (entry 1) | 0.852 | 1.0 | 0.583 | 0.00889 | 0.00115 |
| **this transformer** | **0.858** | 1.0 | 0.490 | **0.00568** | **0.00097** |
| transformer large (rejected) | 0.861 | 1.0 | 0.516 | 0.00578 | 0.00156 |
| random K/P (example generator) | 0.479 | 1.0 | 0.147 | 0.354 | 2.378 |

This library is **36% closer to the reference distribution on FBD** than the Markov library, better on
MMD, and its diversity slightly exceeds the held-out real-AMP value.

**Why both entries exist.** The one metric the Markov library wins is Conformity, 0.583 against 0.490.
Note that this transformer's 0.490 is essentially the real-AMP value of 0.489 — it matches real peptides —
while the Markov chain scores *above* what real AMPs score. Since the Phase 1 aggregation weights are
withheld, we could not determine which the grader rewards, so both models are submitted rather than
guessing.

## Selection

Identical machinery to entry 1, so the two entries differ only in their generative model:

`score = rank(net charge) - rank(mean hydrophobicity)`, two terms, no fitted weights, adopted against
three pre-registered gates on a cluster-disjoint held-out half (`PREREG_SELECTION_2.md`). Guards:

1. **Measured envelope** — net charge in [-1, +5], mean hydrophobicity in [-0.05, +0.65].
2. **Composition guard** at the reference set's 95th percentile (max single residue <= 0.500, W <= 0.238,
   aromatic FWY <= 0.333, Q <= 0.111), cysteine excluded outright for synthesis quality.
3. **Internal diversity cap** (pairwise Levenshtein <= 0.7). The 25 assayed peptides are drawn uniformly
   at random from the top-100, so ordering cannot affect any score and near-duplicates waste draws.

**Novelty** is screened on *sequence identity*, not an edit ratio, under three definitions simultaneously
(matches/shorter-length, local alignment at coverage >= 0.8, full-length global), by BLOSUM62 alignment.
**Zero violations, all three.**

## Compliance

| check | result |
|---|---|
| `_verify_sequences(library.fasta)` | PASS, 50,000 unique |
| `_verify_no_overlap` vs 39,448 references | PASS |
| `_verify_top(top.fasta, k=100)` | PASS |
| `_veritfy_max_simularity(<= 0.80)` | PASS |
| identity <= 0.80 under all three definitions | PASS, 0 violations |
| two independent runs, byte-compared | **IDENTICAL** |

20 standard amino acids, 8-50 residues, linear, free termini, unique, no cysteine in the top-100.

**Determinism note:** sampling is forced onto the CPU even when a GPU is present, because CUDA and CPU
draw from different random streams and reproduction is byte-compared. 50,000 sequences takes ~25 minutes
on CPU.

## Training data and disclosures

**Generation training data: one corpus only**, `data/antibacterial.fasta` as shipped in the template
(39,448 sequences). No pretrained weights, no external sequences. Trained weights ship as
`checkpoint/peptide_lm.pt`.

**External public data was used to choose the SELECTION RULE** (explicitly permitted): measured MIC from
GRAMPA (aggregating DBAASP, DRAMP, YADAMP, APD, DADP), filtered to unmodified free-termini peptides on
panel species, 2,904 labelled sequences; and Hemolytik-derived HC50, 501 sequences with both. No learned
model runs at generation time beyond the shipped LM: the selection rule is closed-form with constants
hard-coded and their provenance in comments.

**AI assistance.** This repository was written with AI assistance (Anthropic Claude), as the rules permit
and require to be disclosed.

## Limitations

1. **No experimental validation.** Every relationship used was measured on *published* peptides.
2. **Applied out of distribution by construction.** Candidates must sit below 80% identity to known AMPs
   while every fitted relationship comes from known AMPs.
3. **0.81M parameters is small.** This is not competitive in scale with billion-parameter protein
   language models; it is a well-regularised model on a small corpus, and it was chosen over a larger one
   on measured evidence of memorisation.
4. Perplexity is not the objective. A better density model can be a worse generator for this task; the
   choice between models was made on the Phase 1 metrics above, not on perplexity.
5. **We do not claim to beat a null.** No published study MIC-tests random or composition-matched
   peptides at this competition's <= 16 uM threshold.
6. Novelty is screened against the template's 39,448 rather than MarLys, which we could not obtain. A
   union with DRAMP 3.0 and GRAMPA is 43,025 unique against 39,448, which bounds the gap at about 9%.
7. The Aggregation Score includes oracle-predicted potency over the full library, and this library is
   unfiltered model output. Filtering toward predicted potency was tested and **rejected**: it lost on
   all four measured Phase 1 metrics (conformity 0.583 -> 0.368, MMD 7.7x worse).
