#!/usr/bin/bash
# CPU job (network needed: git clone, Maven downloads): the Java context file (failure output of the broken test, definitions of the changed symbols) for the 273-case Java contrast subset.
# Resumable. Then the Java ablation jobs read ~/java_ctx/context.json.
#   sbatch -p amd slurm/build_java_context.sbatch
#SBATCH -J java-ctx
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
mkdir -p "$HOME/java_ctx"
echo "=== start $(date) on $(hostname)"
PYTHONPATH="$HOME/PyTaRGET" "$HOME/venv3.12/bin/python" -u -m data_processing.java_context build --split-dir "$HOME/eval_results_java_llm/contrast273" \
    --out "$HOME/java_ctx/context.json" --workdir "$HOME/java_ctx/work" --m2 "$HOME/java_ctx/m2" --workers "${WORKERS:-4}"
echo "=== done $(date)"
