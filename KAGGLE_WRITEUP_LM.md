# DO NOT SUBMIT THIS FILE. The transformer entry is a different repository.

This file previously described the transformer entry and gave **this** repository as its link, with
`uv run generate_lm` as its entry point. Submitting it that way would have failed silently. It is kept
as a redirect rather than deleted so that nobody follows a stale copy of it.

## Why

The organizers' validator, `scripts/verify_submission.py`, hardcodes:

    ENTRY_POINT = "generate"
    library_fasta = dir / ENTRY_POINT / "library.fasta"

It runs `uv run generate` and reads `generate/library.fasta`. It never invokes any other script name, and
the template README states the requirement directly: "Entry point runnable via `uv run generate`".

In this repository `generate` is the **order-2 Markov chain**. So if this repository were submitted a
second time as the transformer entry, the validator would regenerate the Markov library, and entry 2
would be a byte-identical duplicate of entry 1. Nothing would raise an error. A git branch does not fix
it either, because graders clone the default branch.

One repository per model is the only arrangement the validator can distinguish.

## Where the two entries actually live

| entry | model | repository | writeup to paste |
|---|---|---|---|
| 1 | order-2 Markov | https://github.com/braxtonmensah/amp-challenge-2027-entry | `KAGGLE_WRITEUP.md` in **this** repo |
| 2 | transformer LM | https://github.com/braxtonmensah/amp-challenge-2027-entry-lm | `KAGGLE_WRITEUP.md` in **that** repo |

## Two corrections that belong with this redirect

**The transformer seqme figures of diversity 0.858, conformity 0.490, FBD 0.00568 and MMD 0.00097 are
withdrawn.** They were measured on a development library produced by the `lm.py sample` subcommand, which
sizes batches adaptively off a 1024 default and uses CUDA when a GPU is present. The submitted entry
point samples with a fixed batch of 512 pinned to the CPU, so it draws a different random stream and is a
different library. Measured on the files that actually ship, the transformer library scores diversity
0.8563, conformity 0.4850, FBD 0.00555, MMD 0.000732, against a real-AMP ceiling of 0.8530 / 0.4918 /
0.00485 / 0.000502. The qualitative reading is unchanged; the numbers are not.

**`generate_lm/` in this repository is not a submitted artifact.** It was produced by the float32
sampling path. The transformer entry now samples in float64, because float32 did not reproduce across
machines: two machines running the same seed and the same torch version produced libraries differing in
one sequence of 50,000. See the other repository for the measurement and the fix.

`src/amp/lm.py` in this repository also still carries the untyped causal-mask bug, in which the mask
stays float32 whatever dtype the model is in. It is harmless here because nothing in this entry's
generation path uses `lm.py` at all, and it is fixed in the transformer repository.
