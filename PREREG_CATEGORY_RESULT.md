# H-CAT: refuted. Both entries keep `score_pool`.

Pre-registered in `src/amp/eval_category_split.py` before any number below existed. Run 2026-09-29.

## The hypothesis

Both shipped entries select with the same rule, `score_pool` = rank(net charge) - rank(mean
hydrophobicity), whose own docstring says it "optimises one category and concedes the four that
average Success Rate". The competition scores five categories and the rules allow one entry per
sufficiently different model. So: re-select entry 2 for raw potency, cover the conceded four, and
spend the second entry on a category the first one gives away.

Gate A, fixed in advance: the challenger's top-100 must beat `score_pool`'s by **>= 0.05 absolute**
measured panel success rate on a cluster-disjoint held-out half. A tie is a failure.

## What happened

Unconstrained, it looked decisive. Held-out half, n = 1600:

| rule | success | logMIC50 | gram-neg |
|---|---|---|---|
| `score_pool` (shipped) | 0.6600 | 0.8589 | 2.45 |
| raw net charge | **0.7749** | **0.6036** | **2.69** |
| charge + hydrophobicity window | 0.7761 | 0.7305 | 2.18 |
| random null | 0.5491 | - | - |

**+0.1149, more than twice the gate.** Gate B then showed no safety-window cost either: on the 505
sequences with paired HC50, raw charge scored E[log SW | active] 1.9013 against `score_pool`'s
1.8979, a tie, while winning on success rate. That reads as strict dominance, and it would have
justified changing the shipped selection.

## The confound that killed it

The shipped `pick_top` selects only **inside a measured envelope** (net charge -1..+5, hydrophobicity
-0.05..+0.65, cysteine excluded, composition caps at the 95th percentile of real antibacterials).
Gate A applied every rule with no envelope at all. Repeating it with all rules restricted to that
same envelope, on the same held-out half:

| rule, **inside the envelope** | success | gram-neg | median charge |
|---|---|---|---|
| `score_pool` (shipped) | 0.5827 | **2.05** | 4.00 |
| raw net charge | 0.6092 | 1.79 | 4.30 |
| `score_one` (retired) | 0.5848 | 1.59 | 4.00 |
| random null | 0.4466 | - | - |

**+0.0265. The gate required +0.05. H-CAT is refuted.** Gram-negative coverage is also *worse*
(1.79 vs 2.05), which matters because the panel is 15 Gram-negative of 20 strains.

The entire unconstrained win was raw charge reaching past the envelope for high-charge peptides the
guard forbids. That is not a new finding; it is the one `pick_top` already documents and already
rejected: Gram-negative success does rise with charge (0.606 at 4-6, 0.770 at 6-8, 0.812 at 8-10),
but every variant lifting median charge above +5 collapses the organizers' own Phase 1 conformity
metric (0.303 at cap +7, 0.197 at +9, against 0.489 for real AMPs), and qualification for the assay
precedes any Phase 2 gain.

## What this is worth

A negative. The selection rule was already at a defended local optimum and the obvious challenger
loses to it once compared on equal terms. Recorded because the unconstrained table is exactly the
result that would have been shipped as an improvement by anyone who did not run the envelope check,
and because the repo now contains the comparison `PREREG_SELECTION_2.md` never made: `score_pool`
against raw net charge rather than against the retired composite.

The bar was not moved after seeing the number. +0.0265 is a failure at a gate of +0.05.
