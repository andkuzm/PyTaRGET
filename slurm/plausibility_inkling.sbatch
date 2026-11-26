#!/usr/bin/bash
# CPU: plausibility (pytest) of Inkling-2-Mini's Python predictions, same settings as the earlier zero-shot LLM runs (top 5, span control). Results: ~/eval_llm_plaus/<MODEL>/0.8/plausibility_v3.json. Resumable.
#   sbatch --export=ALL,MODEL=full_inkling_low slurm/plausibility_inkling.sbatch      WARNING: clones third-party repositories and runs their tests with model-generated code (stripped test environment, see plausibility_test.py).
#SBATCH -J plaus-inkling
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
source "$HOME/PyTaRGET/slurm/env.sh"
M="${MODEL:-full_inkling_low}"; SCRATCH="/tmp/plaus_${USER}_${SLURM_JOB_ID:-manual}"; mkdir -p "$SCRATCH"; cd "$ROOT"
echo "=== $M $(date) on $(hostname)"
PYTHONPATH="$ROOT" "$PY" -u -m data_processing.plausibility_test --results "$HOME/eval_llm_plaus" --models "$M" --workdir "$SCRATCH" --workers "${WORKERS:-6}" --k 5 --span-control --out-name plausibility_v3.json &
pid=$!; trap 'kill -INT "$pid" 2>/dev/null' TERM; wait "$pid"; rc=$?; if kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi
echo "=== exit $rc $(date)"
rm -rf "$SCRATCH"
