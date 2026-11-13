#!/usr/bin/bash
# Smoke test of the modern local models (vLLM) on the first 50 per-project test cases; one H200.
#   sbatch slurm/smoke_llm.sbatch      (from the PyTaRGET directory; logs slurm/logs/, output ~/eval_smoke_llm/<name>)
#SBATCH -J smoke-llm
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_vllm/bin:$PATH"
export PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SPLIT="$HOME/eval_results_perproject/codet5p/0.8"
for spec in "qwen3coder Qwen/Qwen3-Coder-30B-A3B-Instruct 512" "gptoss20b openai/gpt-oss-20b 2048"; do
  set -- $spec
  echo "=== $1 ($2) $(date)"
  "$HOME/venv_vllm/bin/python" -m data_processing.vllm_run --model "$2" --split-dir "$SPLIT" \
      --out-dir "$HOME/eval_smoke_llm/$1" --n 5 --limit 50 --max-new-tokens "$3"
  echo "=== $1 exit $? $(date)"
done
