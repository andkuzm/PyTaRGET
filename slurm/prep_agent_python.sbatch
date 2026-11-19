#!/usr/bin/bash
# CPU job (network needed: git clone, pip): prepare the Python agent cases (sibling-safe workspaces + one virtualenv per repository), resumable.
#   sbatch -p amd slurm/prep_agent_python.sbatch      ; then the agent job with --dependency=afterok:<this job>
#SBATCH -J prep-agent-py
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 05:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
ROOT="${ROOT:-$HOME/agent_work3}"
echo "=== start $(date) on $(hostname)"
PYTHONPATH="$HOME/PyTaRGET" "$HOME/venv3.12/bin/python" -u scripts/agent/prepare_cases.py --split-dir "$HOME/eval_results_perproject_llm/contrast300" --root "$ROOT" --n 100 --sibling-safe
echo "=== done $(date)"
