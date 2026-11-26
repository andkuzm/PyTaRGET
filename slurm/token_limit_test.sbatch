#!/usr/bin/bash
# How many empty answers does a higher output limit recover, and at what cost? Inkling-2-Mini, effort medium and low, output limits 12288 / 24576 / 32768, on the 150 VALIDATION cases, 2 candidates each.
# Only the output limit differs between runs (same max model length, seed, batches). Why every candidate stopped is saved (candidate_meta.jsonl).     sbatch slurm/token_limit_test.sbatch
#SBATCH -J token-limit
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_inkling/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
for e in medium low; do for lim in 12288 24576 32768; do
  name=${e}_$lim; echo "=== $name $(date)"; t0=$(date +%s)
  "$HOME/venv_inkling/bin/python" -m data_processing.vllm_run --model thinkingmachines/Inkling-2-Mini --split-dir "$HOME/eval_inkling/val150" --out-dir "$HOME/eval_inkling/token_limit/$name" \
      --n 2 --prompt-version 1 --reasoning-effort $e --max-prompt-tokens 24000 --max-new-tokens $lim --max-model-len 57344 --chunk 150 --save-meta
  echo "=== $name exit $? seconds $(( $(date +%s) - t0 ))"
done; done
echo "=== done $(date)"
