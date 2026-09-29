#!/bin/bash
# Install ColabDesign into the BindCraft venv.
#
# BindCraft's functions/colabdesign_utils.py imports `colabdesign` (AF2 backpropagation). It is not on
# PyPI under that name; BindCraft's own conda installer pulls it from GitHub. Installing it directly.
#
# Load the module the venv was built against, or the compute node cannot find libpython3.10.so.1.0.

if ! type module >/dev/null 2>&1; then
  [ -f /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash
fi
module load python/gpu/3.10.10 >/dev/null 2>&1

export TMPDIR=/N/scratch/bsmensah/tmp
export PIP_CACHE_DIR=/N/scratch/bsmensah/pipcache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

PY=/N/scratch/bsmensah/bindcraft/.venv/bin/python

echo "START $(date)"
$PY -m pip install --no-cache-dir "git+https://github.com/sokrypton/ColabDesign.git" 2>&1 | tail -6

echo
echo "=== VERIFY (import, not exit code) ==="
$PY -c "import colabdesign; print('colabdesign OK', colabdesign.__file__)" 2>&1 | tail -3

echo
echo "=== BindCraft functions import ==="
cd /N/scratch/bsmensah/bindcraft || exit 1
$PY -c "import sys; sys.path.insert(0,'.'); import functions; print('BINDCRAFT FUNCTIONS IMPORT OK')" 2>&1 | tail -6
echo "DONE $(date)"
