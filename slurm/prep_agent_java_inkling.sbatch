#!/usr/bin/bash
# CPU (needs network): rebuild the Java agent cases (same seed and rules as the earlier Qwen runs) and warm the shared Maven repository for the offline agent job. Resumable.
#   sbatch slurm/prep_agent_java_inkling.sbatch      WARNING: clones third-party repositories and runs their builds.
#SBATCH -J prep-agent-java
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
R="${ROOT:-$HOME/agent_java_ink}"; mkdir -p "$R"
run() { env -u GH_TOKEN -u HF_TOKEN PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 "$HOME/venv3.12/bin/python" "$@"; }
echo "=== prepare $(date) on $(hostname)"; run scripts/agent/prepare_cases_java.py --root "$R" --n "${N:-60}" || exit 1
echo "=== warm $(date)"; run scripts/agent/warm_cases_java.py --root "$R" || exit 1
echo "=== done $(date)"
