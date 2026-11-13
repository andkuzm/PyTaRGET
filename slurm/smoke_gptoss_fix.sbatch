#!/usr/bin/bash
# gpt-oss-20b on the first 50 per-project test cases with the fixes (low reasoning effort, answer parsing, reindent).
#SBATCH -J smoke-gptoss
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_vllm/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
"$HOME/venv_vllm/bin/python" -m data_processing.vllm_run --model openai/gpt-oss-20b \
  --split-dir "$HOME/eval_results_perproject/codet5p/0.8" --out-dir "$HOME/eval_smoke_llm/gptoss20b_fix" \
  --n 5 --limit 50 --max-new-tokens 2048 --reasoning-effort low --reindent
