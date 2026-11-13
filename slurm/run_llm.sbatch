#!/usr/bin/bash
# Qwen3-Coder-30B-A3B and gpt-oss-20b (vLLM, prompt version 1, 5 candidates at temperature 0.7) on one test set.
# Resumable: finished cases are kept, so submitting the same job again continues.
#   sbatch --export=ALL,SPLIT_DIR=<dir with splits/test.json>,OUT_ROOT=<output dir>,MAX_PROMPT=4096,MAX_LEN=8192 slurm/run_llm.sbatch
#SBATCH -J run-llm
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
: "${SPLIT_DIR:?}" "${OUT_ROOT:?}"; MAX_PROMPT="${MAX_PROMPT:-4096}"; MAX_LEN="${MAX_LEN:-8192}"
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_vllm/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
run() { "$HOME/venv_vllm/bin/python" -m data_processing.vllm_run --split-dir "$SPLIT_DIR" --n 5 --prompt-version 1 \
          --max-prompt-tokens "$MAX_PROMPT" --max-model-len "$MAX_LEN" "$@"; }
echo "=== qwen3coder $(date)"
run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --out-dir "$OUT_ROOT/qwen3coder" --max-new-tokens 512
echo "=== gptoss20b $(date)"
run --model openai/gpt-oss-20b --out-dir "$OUT_ROOT/gptoss20b" --max-new-tokens 2048 --reasoning-effort low --reindent
echo "=== done $(date)"
