#!/bin/bash
#SBATCH -J plm-resample
#SBATCH -p h100-single
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /N/scratch/bsmensah/amp-gpu/plm_%j.log
set -eo pipefail
ROOT=/N/scratch/bsmensah/amp-gpu
export UV_CACHE_DIR=/N/scratch/bsmensah/amp-verify/uvcache
export UV_PYTHON_INSTALL_DIR=/N/scratch/bsmensah/amp-verify/pythons
export PATH=/N/scratch/bsmensah/amp-verify/bin:$PATH
export HF_HOME=$ROOT/hf
export HF_HUB_OFFLINE=1          # compute nodes have no network; model is pre-cached
cd "$ROOT/repo"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -u -m amp.gen_plm_resample --pool 100000 --stratify
echo "JOB DONE"
