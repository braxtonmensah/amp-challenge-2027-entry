# Determinism

Every stochastic step is seeded from a single constant, `SEED = 20260930`, defined once in
`src/amp/generate.py`. No other source of randomness is used: no `random.seed()` elsewhere, no
unseeded `numpy.random`, no time- or PID-derived entropy, no set/dict iteration order used to
influence output (all collections are sorted before use).

Re-running `uv run generate` on any machine with the pinned dependency versions must reproduce the
submitted library byte for byte. A reviewer can verify with:

    uv run generate --out check.csv
    diff check.csv submission/library.csv
