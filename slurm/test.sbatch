#!/usr/bin/bash
# Generate the candidate repairs for the test split of one model and score them (writes metrics.json next to the
# predictions). Safe to start again: continues with the cases that have no prediction yet. Exit status 75 = not finished.
#   sbatch --export=ALL,MODEL=codet5p slurm/test.sbatch
#SBATCH -J test
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
: "${MODEL:?set MODEL=codet5p|plbart|codegen, e.g. sbatch --export=ALL,MODEL=codet5p slurm/test.sbatch}"
echo "test $MODEL on $(hostname): $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
run_stage test --model "$MODEL" ${EXTRA_ARGS:-}
