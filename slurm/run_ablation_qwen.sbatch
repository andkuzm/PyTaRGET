#!/usr/bin/bash
# WP5 context ablation with Qwen3-Coder-30B-A3B (vLLM, prompt version 1, ONE candidate at temperature 0.7, same settings in every run):
# A baseline, B + failure output, C + definitions of changed symbols, BC both, on the 300-case Python contrast subset.
# The prompt budget is 6144 tokens in ALL runs (also A) so that the extra sections do not cost hunks unevenly. Resumable.
#   sbatch --export=ALL,CONTEXT_FILE=$HOME/eval_ablation/context.json slurm/run_ablation_qwen.sbatch
#SBATCH -J abl-qwen
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
: "${CONTEXT_FILE:?}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/eval_results_perproject_llm/contrast300}"; OUT_ROOT="${OUT_ROOT:-$HOME/eval_ablation}"
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_vllm/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
run() { # name [extra args]
  name="$1"; shift
  echo "=== $name $(date)"
  "$HOME/venv_vllm/bin/python" -m data_processing.vllm_run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --split-dir "$SPLIT_DIR" --n 1 \
    --prompt-version 1 --max-prompt-tokens 6144 --max-model-len 8192 --max-new-tokens 512 --out-dir "$OUT_ROOT/qwen3coder_$name" "$@"
}
run A
run B  --context-variant B  --context-file "$CONTEXT_FILE"
run C  --context-variant C  --context-file "$CONTEXT_FILE"
run BC --context-variant BC --context-file "$CONTEXT_FILE"
echo "=== done $(date)"
