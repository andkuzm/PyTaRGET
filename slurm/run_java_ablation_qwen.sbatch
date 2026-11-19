#!/usr/bin/bash
# Java context ablation with Qwen3-Coder-30B-A3B (vLLM, Java prompt, ONE candidate at temperature 0.7, same settings as the earlier Java prompt-only runs: prompt budget 4096 tokens):
# B + failure output of the broken test, C + definitions of the changed symbols, BC both, on the 273-case Java contrast subset. Needs ~/java_ctx/context.json (slurm/build_java_context.sbatch). Resumable.
#   sbatch --dependency=afterok:<context job> slurm/run_java_ablation_qwen.sbatch
#SBATCH -J abl-java-qwen
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
CONTEXT_FILE="${CONTEXT_FILE:-$HOME/java_ctx/context.json}"; SPLIT_DIR="${SPLIT_DIR:-$HOME/eval_results_java_llm/contrast273}"; OUT_ROOT="${OUT_ROOT:-$HOME/eval_ablation_java}"
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_vllm/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
run() { # variant
  echo "=== $1 $(date)"
  "$HOME/venv_vllm/bin/python" -m data_processing.vllm_run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --split-dir "$SPLIT_DIR" --n 1 --java \
    --prompt-version 1 --max-prompt-tokens 4096 --max-model-len 8192 --max-new-tokens 512 --out-dir "$OUT_ROOT/qwen3coder_$1" --context-variant "$1" --context-file "$CONTEXT_FILE"
}
run B
run C
run BC
echo "=== done $(date)"
