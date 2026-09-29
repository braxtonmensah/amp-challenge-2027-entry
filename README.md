# AMP Challenge 2027 entry: motif-faithful generation with biophysical ranking

Braxton Mensah, Indiana University Bloomington (`bsmensah@iu.edu`).

    uv sync
    uv run generate

Writes `generate/library.fasta` (50,000 sequences) and `generate/top.fasta` (100, ranked by file order).

---

## Disclosures, up front

**AI assistance.** This repository was written with AI assistance (Anthropic Claude), which the
competition rules permit and require to be disclosed. The scientific choices below, in particular the
deliberate decision in the "safety window" section to score hydrophobicity toward a band rather than
maximise it, are stated so a reviewer can disagree with them explicitly.

**Training data.** One corpus only: `data/antibacterial.fasta` as shipped in the organizers' template
repository, 39,448 sequences, all 8-50 residues. **No pretrained model, no external database, no
additional corpus, no proprietary data.** The reference set is used for three things: fitting the
order-2 transition table, drawing the length distribution, and filtering for novelty.

**Determinism.** One seed, `SEED = 20260930`, one `numpy` Generator, sorted iteration throughout. See
`SEED.md`. Two independent runs on this machine produced byte-identical `library.fasta` and
`top.fasta`.

## Method

### Why not maximise the obvious thing

The organizers' example generator samples random `K`/`P` strings. It satisfies the schema and carries no
biology. Two determinants of cationic antimicrobial peptide activity are well established and cheap:

1. **Net positive charge**, which drives association with the anionic bacterial surface. The reference
   actives are **20.6% K+R**, roughly double a typical proteome.
2. **Amphipathicity**, the segregation of a hydrophobic from a polar face on helix formation, measured
   here as the Eisenberg hydrophobic moment at 100 degrees per residue.

### The safety window, and the trade this entry makes deliberately

Phase 2 scores a **safety window, HC50/MIC50**, not raw potency. Raw hydrophobicity drives haemolysis
about as readily as it drives killing, so maximising it buys MIC at the cost of HC50 and can lose on the
metric that is actually reported.

**So hydrophobicity is scored toward a target band (mean Eisenberg 0.05 to 0.45) rather than
maximised, and net charge toward +4 to +9 rather than maximised.** This trades some predicted potency
for a predicted safety margin. It is a judgement, it is reversible in two constants at the top of
`generate.py`, and it is the single most consequential choice in this entry.

### Sequence realism

An **order-2 Markov chain** fitted to the reference actives, so local motifs (`KKIL`, `GKII` and
similar) occur at their natural frequency instead of being assembled from independent per-position
draws. Lengths are drawn from the empirical reference length distribution (median 18).

### Ranking

Composite, weights stated rather than tuned against any withheld metric:

| term | weight | rationale |
|---|---|---|
| net charge in band +4 to +9 | 0.40 | membrane association |
| hydrophobic moment, saturating at 0.55 | 0.30 | amphipathicity |
| mean hydrophobicity in band 0.05-0.45 | 0.20 | potency vs haemolysis trade |
| Chou-Fasman helix propensity | 0.10 | helical AMPs dominate the class |

Phase-1 aggregation weights are withheld by the organizers by design, so nothing here is fitted to
them.

### Novelty

The library excludes exact matches to the reference set. The top 100 additionally sit at
**Levenshtein ratio <= 0.8 from every one of the 39,448 reference sequences**, which is the
organizers' own novelty rule, so no ranked candidate is a paraphrase of a known peptide.

## Compliance, checked against the organizers' own validator

The check functions in the template's `scripts/verify_submission.py` were imported and run directly
against this output:

| check | result |
|---|---|
| `_verify_sequences(library.fasta)` | PASS, 50,000 unique |
| `_verify_no_overlap` vs 39,448 references | PASS, no exact match |
| `_verify_top(top.fasta, k=100)` | PASS, all 100 present in the library |
| `_veritfy_max_simularity(<= 0.80)` | PASS |
| two independent runs, byte-compared | IDENTICAL |

## Honest limitations

1. **No experimental validation of anything here.** The ranking is biophysical, not learned from
   measured MIC values, because no MIC-labelled data was used.
2. **An order-2 Markov chain is a weak generative model.** It captures dipeptide context and nothing
   longer. It cannot represent the tertiary or aggregation behaviour that decides real activity.
3. **The helix assumption is baked in.** The hydrophobic moment is computed at 100 degrees per
   residue, so beta-sheet and cyclic antimicrobials are scored by a model that does not describe them.
4. **The scoring weights are asserted from the literature, not fitted.** They were not validated
   against any held-out activity data, because none was used.
5. Nothing here was screened for cytotoxicity beyond the hydrophobicity band, protease stability, or
   synthesis feasibility.

## Licence

MIT, see `LICENSE`.
