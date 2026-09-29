# Pre-registration: learned selection of the top-100 from the 50,000-sequence library

Written **before** any model was fitted or any cross-validation number was seen. Recorded here so the
selection procedure the competition asks us to document is falsifiable rather than asserted, and so a
reviewer can check that we honoured the abort conditions.

Author: Braxton Mensah, Indiana University Bloomington (`bsmensah@iu.edu`).
Method developed with AI assistance (Anthropic Claude), disclosed per competition rules.

---

## 0. Two corrections to our own earlier reasoning, on reading the rules

**(a) Ranking inside the top-100 is worth nothing.** The rules state that 25 peptides are drawn
**uniformly at random** from each team's top-100, and that team scores are the arithmetic mean of the
peptide metric over those 25. Therefore the order of our 100 has no effect on any score. What matters
is *which 100 sequences are in the set* — i.e. the selection threshold applied to 50,000 candidates.
Our earlier plan to "rerank the top 100" was aimed at a quantity the competition does not measure.
This pre-registration is about **selection**, not ordering.

**(b) External MIC data is explicitly permitted.** Our published README stated "no external database".
That was a self-imposed restriction, not a rule: the competition proposal says teams "are free to use
any public peptide and/or AMP database for training their model" and points specifically at DBAASP's
MIC values. Continuing to refuse public MIC data would forfeit the competition's central lever for no
reason. The README disclosure will be updated to state exactly what is used.

## 1. Hypothesis

H1. A descriptor model trained on measured MIC values selects a set of 100 peptides with better
measured activity than our shipped asserted-biophysics score does.

This is falsifiable and it may well be false: the biophysical score encodes charge and amphipathicity,
which are the dominant known determinants, so a trained model has a real chance of adding nothing.

## 2. Data and label, fixed now

Source: GRAMPA (`grampa.csv`, 51,345 MIC measurements aggregated from DBAASP, DRAMP, YADAMP, APD,
DADP; all values log10 MIC in µM). Haemolysis: Hemolytik-derived `log10_HC50`. Both public.

**Inclusion, driven by the competition's own peptide constraints** (linear, free termini, no
amidation, 20 standard AA, 8-50 aa):

* `is_modified == False` — **including excluding C-terminal amidation.** The competition forbids
  terminal modification, and amidation adds ~+1 charge and typically shifts MIC severalfold. Training
  on amidated measurements and applying the model to free-acid peptides would bias every prediction.
  This drops GRAMPA from 51,345 to ~29,101 measurement rows and is not optional.
* 20 standard amino acids only, length 8-50.

**Label, matched to the scored metric and the actual 20-strain panel** (15 Gram-negative, 5
Gram-positive; panel read from the competition website):

* Panel-matched species only. Gram-negative: *A. baumannii, E. cloacae, E. coli, K. pneumoniae,
  P. aeruginosa, S. enterica* (incl. Typhimurium). Gram-positive: *B. subtilis, S. aureus,
  E. faecalis, E. faecium*. Everything else is dropped, **including *C. albicans*** (a fungus, 2,860
  rows) and *M. luteus*, *S. epidermidis*, *B. cereus*, which are not on the panel.
* Primary label `success_rate`: the fraction of a sequence's panel-matched measurements with
  MIC <= 16 µM (the competition's potency threshold; log10 value <= 1.2041), computed per species then
  **averaged with panel weights 15:5 Gram-negative:Gram-positive** so the target matches the panel
  composition rather than GRAMPA's own composition (GRAMPA over-represents *S. aureus*).
* Secondary label `log_mic50`: the panel-weighted median log10 MIC.
* Haemolysis label `log_hc50`, modelled separately.

## 3. Evaluation

* Features: the existing descriptor vector (composition, length, charge, Eisenberg hydrophobicity,
  hydrophobic moment at 100 deg and 180 deg, helix propensity, aromatic/tiny/cationic/anionic
  fractions). No learned embeddings.
* Estimator: ridge regression. Deliberately not a network: at a few thousand sequences a descriptor
  model cannot silently overfit the way an under-regularised network can.
* **Similarity-clustered grouped 5-fold CV.** Whole clusters are held out together, single-linkage at
  Levenshtein ratio >= 0.6. AMP datasets are dense with near-duplicates, and random k-fold on them
  leaks; that leak is the standard reason published AMP predictors do not reproduce.
* A **shuffled-label null** is run at identical settings and reported next to every real number.

## 4. Gates, fixed now

**Gate 1 (usability).** Grouped-CV Spearman must exceed the shuffled-label null by **>= 0.10** in
absolute value. Below that, the model is not distinguishable from its own null and is not used.

**Gate 2 (the decisive head-to-head).** Within held-out folds only, take the 100 highest-scoring
sequences by (a) the trained model and (b) the shipped biophysical `score_one`. Compare the **measured**
panel-weighted `success_rate` of the two sets. The trained model must beat the biophysical score by
**>= 0.05 absolute success rate** to be adopted.

**If Gate 2 fails, H1 is refuted and we ship the biophysical selection unchanged, and say so.** A tie
is a failure, not a licence to prefer the more sophisticated method.

**Gate 3 (haemolysis, filter only).** The HC50 model is used only to drop candidates in the predicted
worst decile of HC50, and only if it independently clears Gate 1. It never contributes to the
selection score, because the "Optimal Selectivity" category requires activity in at least one strain
before a peptide is counted at all, so suppressing potency to chase HC50 can score zero.

## 5. Abort conditions

* Fewer than 500 usable labelled sequences after filtering -> do not model; ship biophysical.
* More than 50% of label rows unparseable -> treat as a format problem, not a finding.
* Any change to these gates after seeing a result must be recorded as a new pre-registration with the
  old one left in place.

## 6. Known limitation, stated in advance

GRAMPA labels come from known natural and published AMPs. Our candidates are required to sit below 80%
identity to any known AMP, so the model is applied **out of distribution** by construction. The
cluster-held-out CV is the closest available estimate of that gap, and it is still optimistic. This
limitation cannot be removed with the data available and is the main reason Gate 2 is set on a
measured head-to-head rather than on a correlation alone.
