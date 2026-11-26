#!/usr/bin/bash
# CPU (needs network): leak-safe (sibling-safe) agent workspaces for the first N cases of the seed-7 sample of the Python contrast subset, same cases as the earlier agent runs.
#   sbatch --export=ALL,N=8 slurm/prep_agent_inkling.sbatch      WARNING: clones third-party repositories and installs / runs their code.
#SBATCH -J prep-agent-ink
#SBATCH -p amd
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
env -u GH_TOKEN -u HF_TOKEN PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 "$HOME/venv3.12/bin/python" scripts/agent/prepare_cases.py \
    --split-dir "$HOME/eval_results_perproject_llm/contrast300" --root "${ROOT:-$HOME/agent_inkling}" --n "${N:-8}" --sibling-safe
