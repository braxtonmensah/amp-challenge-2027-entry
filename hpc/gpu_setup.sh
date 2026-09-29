#!/bin/bash
# Login-node half: build a CUDA env and PRE-DOWNLOAD ESM2. Compute nodes have no network.
set -eo pipefail
ROOT=/N/scratch/bsmensah/amp-gpu
export UV_CACHE_DIR=/N/scratch/bsmensah/amp-verify/uvcache
export UV_PYTHON_INSTALL_DIR=/N/scratch/bsmensah/amp-verify/pythons
export PATH=/N/scratch/bsmensah/amp-verify/bin:$PATH
export HF_HOME=$ROOT/hf

mkdir -p "$ROOT" "$HF_HOME"
if [ ! -d "$ROOT/repo" ]; then
  git clone --quiet https://github.com/braxtonmensah/amp-challenge-2027-entry.git "$ROOT/repo"
fi
cd "$ROOT/repo"
echo "== repo HEAD $(git rev-parse --short HEAD)"

uv sync --quiet 2>&1 | tail -3 || true
uv pip install --quiet seqme transformers 2>&1 | tail -3
echo "== torch check"
uv run python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"
echo "== pre-downloading ESM2 t12_35M into $HF_HOME"
uv run python -c "
from transformers import AutoTokenizer, AutoModel
m='facebook/esm2_t12_35M_UR50D'
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
print('cached ok')
"
echo "SETUP DONE"
