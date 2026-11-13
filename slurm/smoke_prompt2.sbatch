#!/usr/bin/bash
# Prompt version 2 (full broken part, any number of lines, indentation kept) on the first 50 per-project test cases,
# for Qwen3-Coder and gpt-oss-20b (low reasoning effort, no --reindent). One H200; models are already downloaded.
#SBATCH -J smoke-prompt2
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_vllm/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SPLIT="$HOME/eval_results_perproject/codet5p/0.8"
run() { "$HOME/venv_vllm/bin/python" -m data_processing.vllm_run --split-dir "$SPLIT" --n 5 --limit 50 --prompt-version 2 "$@"; }
echo "=== qwen3coder $(date)"
run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --out-dir "$HOME/eval_smoke_llm/qwen3coder_p2" --max-new-tokens 512
echo "=== gptoss $(date)"
run --model openai/gpt-oss-20b --out-dir "$HOME/eval_smoke_llm/gptoss20b_p2" --max-new-tokens 2048 --reasoning-effort low
