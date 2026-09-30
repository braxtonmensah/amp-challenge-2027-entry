# Determinism

Every stochastic step is seeded from a single constant, `SEED = 20260930`, defined once in
`src/amp/generate.py` and imported by the entry point `src/amp/generate_matched.py`. No other source of
randomness is used: no `random.seed()` elsewhere, no unseeded `numpy.random`, no time- or PID-derived
entropy, no set/dict iteration order used to influence output (all collections are sorted before use).

There are two stochastic stages, each with its own generator seeded from that constant: Markov sampling of
the 1,200,000-candidate pool, and the Gumbel top-k draw that selects 50,000 of them. The pool is sorted
before the draw, so the selection does not depend on the order in which candidates were generated.

## What has actually been tested, and what has not

Re-running the entry point with the pinned dependency versions reproduces the submitted library byte
for byte. Stating the scope of that precisely, because the claim is easy to overstate:

| test | result |
|---|---|
| two independent runs, same machine, Windows 11 / Python 3.11.9 / numpy 2.4.6 | **byte-identical, library and top-100** |
| across CPU architectures | **not tested for this entry** |

The second row is a real gap rather than a formality. The companion LM entry was checked on AMD EPYC
7742 against Intel Xeon Gold 6248 precisely because different matmul kernels pick different reduction
orders; this entry has no such check, so "reproduces on any machine" is **not** claimed. What is claimed
is that the generation is seeded, order-independent by construction (the candidate pool is sorted before
any weighted draw), and reproducible on a matching environment. Anything stronger would be asserted
rather than measured.

## Verify it

    uv sync
    uv run generate
    # regenerates generate/library.fasta and generate/top.fasta IN PLACE

`uv sync` installs five packages and about 64 MB: numpy, levenshtein, rapidfuzz, biopython and this
package. It deliberately does **not** install torch.

Two things about the environment are pinned rather than merely suggested, because both can silently
change the output and neither is the verifier's fault if it does:

* **`requires-python = ">=3.11,<3.12"`.** The lock file resolves numpy 2.4.6 only under Python 3.11.
  An earlier `>=3.10` allowed a 3.10 interpreter, where the same lock resolved numpy **2.2.6**, and
  the submitted library was produced on 2.4.6. A verifier on 3.10 would have regenerated a different
  library from the same repository and correctly concluded the reproducibility claim was false.
* **`numpy==2.4.6`, exactly.** It was previously `>=2.2.6`, a floor rather than a pin, which left three
  numpy versions reachable across the resolution. There is now exactly one.

**torch is not a dependency of this entry.** It is imported only by the companion language-model and
RL research modules, none of which sit on the entry point's import path. It was previously declared as
a top-level dependency, which meant `uv sync` on Linux pulled the full CUDA wheel set (cudnn, cublas,
nccl and the rest, roughly 2.7 GB). Tested, that download timed out and the sync failed, which would
have failed this entry's reproducibility re-run for a reason having nothing to do with the method. To
run the companion generator or the RL modules instead:

    uv sync --extra lm     # adds torch

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
