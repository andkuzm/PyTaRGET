#!/usr/bin/bash
# Inkling-2-Mini full passes with the effort chosen on the validation sample (low): Java contrast273 (1 candidate, 4096-token prompts, as the earlier local Java runs), then the Python per-project test set
# (1768 cases, 5 candidates, prompts up to 24000 tokens, as the earlier Qwen3-Coder / gpt-oss runs). Resumable: submit again to continue.     sbatch slurm/inkling_full.sbatch
#SBATCH -J inkling-full
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_inkling/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
run() { "$HOME/venv_inkling/bin/python" -m data_processing.vllm_run --model thinkingmachines/Inkling-2-Mini --prompt-version 1 --reasoning-effort low --max-new-tokens 12288 --max-model-len 40960 "$@"; }
echo "=== java $(date)"; run --split-dir "$HOME/eval_results_java_llm/contrast273" --out-dir "$HOME/eval_inkling/java273_low" --n 1 --max-prompt-tokens 4096 --java --dump-raw 2; echo "=== java exit $? $(date)"
echo "=== python $(date)"; run --split-dir "$HOME/eval_results_perproject_fullctx/codet5p/0.8" --out-dir "$HOME/eval_inkling/python_full_low" --n 5 --max-prompt-tokens 24000 --chunk 96 --dump-raw 2; echo "=== python exit $? $(date)"
