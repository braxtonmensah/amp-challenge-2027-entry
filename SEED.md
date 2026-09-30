# Determinism

Every stochastic step is seeded from a single constant, `SEED = 20260930`, defined once in
`src/amp/generate.py` and imported by the entry point `src/amp/generate_matched.py`. No other source of
randomness is used: no `random.seed()` elsewhere, no unseeded `numpy.random`, no time- or PID-derived
entropy, no set/dict iteration order used to influence output (all collections are sorted before use).

There are two stochastic stages, each with its own generator seeded from that constant: Markov sampling of
the 1,200,000-candidate pool, and the Gumbel top-k draw that selects 50,000 of them. The pool is sorted
before the draw, so the selection does not depend on the order in which candidates were generated.

Re-running `uv run generate` on any machine with the pinned dependency versions reproduces the
submitted library byte for byte.

## Verify it

    uv sync
    uv run generate
    # regenerates generate/library.fasta and generate/top.fasta IN PLACE

The run overwrites the working tree, so compare against the committed copies from git rather than
against files that no longer exist:

    git show HEAD:generate/library.fasta | sha256sum
    git show HEAD:generate/top.fasta     | sha256sum
    sha256sum generate/library.fasta generate/top.fasta

Expected, for the commit that ships these files:

    library.fasta  edb12c30a497148598b5996fe6e30f613cdadb51e06f342af7bc543b65374145
    top.fasta      5024802990138238b9fb74beb729a996b7cb6955588b50d56eedb83628466954

These are the hashes of the potent-corpus library produced by `amp.generate_matched`. An earlier
revision of this file listed `53c88abc…` and `6336aa7c…`, which were correct for the previous
`amp.generate` library and became stale when the entry point changed. Recorded here rather than left
for a reviewer to hit.

An earlier version of this file gave the recipe as `uv run generate --out check.csv` followed by
`diff check.csv submission/library.csv`. That was wrong on three counts: there is no `--out` flag, the
output is FASTA rather than CSV, and no `submission/` directory exists. A reviewer following it would
have got an argparse error and no comparison. Corrected here rather than left to be discovered.
