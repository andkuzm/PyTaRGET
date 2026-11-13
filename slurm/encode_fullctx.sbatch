#!/usr/bin/bash
# Per-project split with a 32768-token window, so (almost) no case is dropped for length: the test set for the
# "full dataset, largest context" evaluation of the modern LLMs. Output: ~/eval_results_perproject_fullctx (separate
# from the matched 1024-token results). CPU only, same resources as encode.sbatch.
#   sbatch slurm/encode_fullctx.sbatch      (from the PyTaRGET directory)
#SBATCH -J encode-fullctx
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
export OUT="$HOME/eval_results_perproject_fullctx" SPLIT_MODE=per_project
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
echo "encode codet5p tokenizer, max-length 32768, per_project, $CSV -> $OUT on $(hostname)"
run_stage encode --model codet5p --max-length 32768 --split-mode per_project
