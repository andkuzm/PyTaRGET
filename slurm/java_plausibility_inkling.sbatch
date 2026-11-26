#!/usr/bin/bash
# CPU: run the Java predictions of Inkling-2-Mini (contrast273, 1 candidate) in the repaired project (data_processing/java_plausibility.py; reference control included). Resumable: submit again to continue.
#   sbatch --export=ALL,EFFORT=low slurm/java_plausibility_inkling.sbatch       WARNING: clones third-party repositories and runs their build and tests with model-generated code.
#SBATCH -J java-plaus-inkling
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
E="${EFFORT:-low}"; SCRATCH="/tmp/javaplaus_${USER}_${SLURM_JOB_ID:-manual}"; mkdir -p "$SCRATCH"
echo "=== java plausibility effort $E $(date) on $(hostname)"
env -u GH_TOKEN -u HF_TOKEN PYTHONPATH="$HOME/PyTaRGET" "$HOME/venv3.12/bin/python" -u -m data_processing.java_plausibility --split-dir "$HOME/eval_results_java_llm/contrast273" \
    --predictions "$HOME/eval_inkling/java273_$E/test_predictions.json" --out "$HOME/eval_inkling/java273_$E/java_plausibility.json" --workdir "$SCRATCH" --workers "${WORKERS:-4}" &
pid=$!; trap 'kill -INT "$pid" 2>/dev/null' TERM; wait "$pid"; rc=$?; if kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi
echo "=== exit $rc $(date)"
rm -rf "$SCRATCH"
