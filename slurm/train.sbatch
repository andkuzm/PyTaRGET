#!/usr/bin/bash
# Fine-tune one model on one GPU. Safe to start again: continues from checkpoint-last after any interruption.
# Exit status 75 = stopped before the end (time limit): submit this script again.
#   sbatch --export=ALL,MODEL=codet5p slurm/train.sbatch       (GPU type: add --gres=gpu:h200-141g:1 etc.)
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
: "${MODEL:?set MODEL=codet5p|plbart|codegen, e.g. sbatch --export=ALL,MODEL=codet5p slurm/train.sbatch}"
echo "train $MODEL on $(hostname): $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
# EXTRA_ARGS: e.g. "--epochs 5 --batch-size 4" or "--restart" (ignore the saved state)
run_stage train --model "$MODEL" ${EXTRA_ARGS:-}
