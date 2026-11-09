#!/usr/bin/bash
# Plausibility (pytest) of the predictions of codet5p, plbart and codegen in $OUT: first the top 5 candidates
# (plausibility.json), then candidates 6-20 (plausibility_ranks6_20.json) for the cases judged in the first pass.
# Runs repositories in parallel inside this job; results are resumable, a job stopped by the time limit can be submitted again.
#   OUT=$HOME/eval_results_perproject sbatch --dependency=afterok:<test job ids> slurm/plausibility.sbatch
# WARNING: clones third-party repositories and runs their tests with model-generated code (see plausibility_test.py).
#SBATCH -J plausibility
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
SCRATCH="/tmp/plausibility_${USER}_${SLURM_JOB_ID:-manual}"
mkdir -p "$SCRATCH"
cd "$ROOT"
echo "plausibility on $(hostname), results $OUT, scratch $SCRATCH, workers ${WORKERS:-6}"
run_pl() {   # stop cleanly with SIGINT 10 minutes before the time limit; Slurm's TERM is passed on by run_stage-like trap
    PYTHONPATH="$ROOT" "$PY" -u -m data_processing.plausibility_test --results "$OUT" --workdir "$SCRATCH" \
        --workers "${WORKERS:-6}" "$@" &
    local pid=$! rc=0
    trap 'kill -INT "$pid" 2>/dev/null' TERM
    wait "$pid" || rc=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid" || rc=$?; fi
    trap - TERM
    return "$rc"
}
run_pl --k 5
echo "=== top 5 done, status $? ==="
run_pl --k 20 --first-rank 5 --extend plausibility.json --out-name plausibility_ranks6_20.json
echo "=== candidates 6-20 done, status $? ==="
rmdir "$SCRATCH" 2>/dev/null || true
