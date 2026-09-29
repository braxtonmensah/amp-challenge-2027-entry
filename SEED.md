# Determinism

Every stochastic step is seeded from a single constant, `SEED = 20260930`, defined once in
`src/amp/generate.py`. No other source of randomness is used: no `random.seed()` elsewhere, no
unseeded `numpy.random`, no time- or PID-derived entropy, no set/dict iteration order used to
influence output (all collections are sorted before use).

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

    library.fasta  53c88abcfa59a55218f82e09bdf014a785cbd7186297a137e052d9e6e10e3034
    top.fasta      6336aa7ca4c6c1f6221d6c5ed27f0a88f2b14b383ef86d40eebdd56c3e39f448

An earlier version of this file gave the recipe as `uv run generate --out check.csv` followed by
`diff check.csv submission/library.csv`. That was wrong on three counts: there is no `--out` flag, the
output is FASTA rather than CSV, and no `submission/` directory exists. A reviewer following it would
have got an argparse error and no comparison. Corrected here rather than left to be discovered.
