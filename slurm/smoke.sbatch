#!/usr/bin/bash
# Short end-to-end check with the REAL models on a small sample (40 commits, 1 epoch, 20 test cases), one model after
# the other in one job: shows the GPU/driver, peak memory, and seconds per stage ("finished in ... s, peak RAM ..., peak GPU
# memory ...") so the time limits of the full runs can be set. Writes to $SMOKE_OUT (default ~/eval_smoke), never to OUT.
#   sbatch slurm/smoke.sbatch                      (MODELS="codet5p plbart codegen" by default)
#SBATCH -J smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --signal=B:TERM@120
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
export OUT="${SMOKE_OUT:-$HOME/eval_smoke}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
"$PY" -I -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())"
for model in ${MODELS:-codet5p plbart codegen}; do
    echo "=== smoke $model ==="
    run_stage smoke --model "$model" || echo "=== smoke $model FAILED with status $? ==="
done
