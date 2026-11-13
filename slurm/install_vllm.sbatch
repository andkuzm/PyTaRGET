#!/usr/bin/bash
# One-off: separate venv with vLLM for the modern-model runs (keeps venv3.12 untouched).
#   sbatch slurm/install_vllm.sbatch      (run from the PyTaRGET directory)
#SBATCH -J install-vllm
#SBATCH -p amd
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
VENV="$HOME/venv_vllm"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install vllm
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
