#!/usr/bin/bash
#SBATCH -J build-matched
#SBATCH -p amd
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
cd "$HOME/PyTaRGET"
PYTHONPATH="$HOME/PyTaRGET" "$HOME/venv3.12/bin/python" -m data_processing.build_matched_split \
  --results "$HOME/eval_results_perproject" --out "$HOME/eval_results_perproject_llm/matched"
