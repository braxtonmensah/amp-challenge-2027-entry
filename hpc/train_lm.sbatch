#!/bin/bash
#SBATCH -J amplm
#SBATCH -A students
#SBATCH -p h100-single
#SBATCH --gpus-per-node=1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /N/u/bsmensah/Quartz/amp-lm/out/lm_%j.out
#SBATCH -e /N/u/bsmensah/Quartz/amp-lm/out/lm_%j.err

# Train two sizes of the peptide language model and sample from each, so the choice between them is
# made on measured validation loss and on the organizers' own Phase 1 metrics rather than on the
# assumption that bigger is better. With 39,448 short sequences a larger model can simply memorise the
# corpus, which is actively harmful here because the competition scores novelty against that corpus.

set -u
cd /N/u/bsmensah/Quartz/amp-lm
PY=/N/u/bsmensah/Quartz/af3screen/venv/bin/python

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo

for CFG in "small 128 4 4" "large 256 8 6"; do
  set -- $CFG
  NAME=$1; D=$2; H=$3; L=$4
  echo "=============================================================="
  echo "TRAIN $NAME  d=$D heads=$H layers=$L"
  echo "=============================================================="
  $PY lm.py train \
      --ref data/antibacterial.fasta \
      --out checkpoint/lm_${NAME}.pt \
      --d $D --heads $H --layers $L \
      --epochs 40 --batch 256 --lr 6e-4 --seed 20260930
  echo
  echo "--- sampling 50,000 from $NAME ---"
  $PY lm.py sample \
      --ckpt checkpoint/lm_${NAME}.pt \
      --ref data/antibacterial.fasta \
      --out out/library_${NAME}.fasta \
      --n 50000 --batch 2048 --temperature 1.0 --seed 20260930
  echo
done

echo "DONE"
ls -la checkpoint out
