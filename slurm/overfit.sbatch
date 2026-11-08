#!/usr/bin/bash
# Diagnostic: encode + train for several epochs + test on the 162-case smoke sample, for the models in $MODELS
# (default plbart). Shows whether a model leaves the "empty output" state once it trains for more than one epoch.
#   sbatch --gres=gpu:h200-141g:1 --export=ALL,MODELS=plbart,EPOCHS=15 slurm/overfit.sbatch
# Results go to $DIAG_OUT (default ~/eval_overfit), never to OUT.
#SBATCH -J overfit
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
export OUT="${DIAG_OUT:-$HOME/eval_overfit}"
SMOKE_CSV="${SMOKE_CSV:-$HOME/annotated_cases_v2_pr59_smoke.csv}"
for model in $(echo "${MODELS:-plbart}" | tr ',' ' '); do
    for stage in encode train test; do
        echo "=== overfit $model $stage ==="
        args=(--model "$model" --csv "$SMOKE_CSV")
        [ "$stage" = train ] && args+=(--epochs "${EPOCHS:-15}" --early-stop 100 --learning-rate "${LR:-5e-5}")
        [ "$stage" = test ] && args+=(--limit "${LIMIT:-20}")
        run_stage "$stage" "${args[@]}" || { echo "=== overfit $model $stage FAILED with status $? ==="; break; }
    done
done
