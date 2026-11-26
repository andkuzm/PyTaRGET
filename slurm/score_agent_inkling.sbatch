#!/usr/bin/bash
# CPU: score an Inkling agent run: whole-function layout (scripts/agent/score_layout.py), then plausibility of the agent's final test function (reference control and span control as for the earlier runs).
#   sbatch --export=ALL,NAME=inkling_low_smoke slurm/score_agent_inkling.sbatch
#SBATCH -J score-agent-ink
#SBATCH -p amd
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
source "$HOME/PyTaRGET/slurm/env.sh"
cd "$ROOT"; N="${NAME:-inkling_low_smoke}"; OUT_L="$HOME/eval_agent_plaus/$N"; SCRATCH="/tmp/agentplaus_${USER}_${SLURM_JOB_ID:-manual}"; mkdir -p "$SCRATCH"
PYTHONPATH="$ROOT" "$PY" scripts/agent/score_layout.py "$HOME/eval_results_perproject_llm/contrast300" "$HOME/${AGENT_ROOT:-agent_inkling}" "$HOME/eval_agent/$N" "$OUT_L"
PYTHONPATH="$ROOT" "$PY" -u -m data_processing.plausibility_test --results "$HOME/eval_agent_plaus" --models "$N" --workdir "$SCRATCH" --workers 4 --k 1 --span-control --out-name plausibility_v3.json
echo "=== exit $? $(date)"; rm -rf "$SCRATCH"
