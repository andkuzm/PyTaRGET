#!/usr/bin/bash
# Reasoning-effort comparison on 150 VALIDATION cases (the effort is chosen here, never on the test set): Inkling-2-Mini none/low/medium/high, Qwen3-Coder (has no reasoning setting), gpt-oss-20b low/medium/high.
# One candidate per case, prompt version 1. Resumable. Output ~/eval_inkling/sweep/<name>.     sbatch slurm/reasoning_sweep.sbatch
#SBATCH -J reasoning-sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:30:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
export PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
S="$HOME/eval_inkling/val150"; O="$HOME/eval_inkling/sweep"
run() { venv="$1"; name="$2"; model="$3"; shift 3
  echo "=== $name $(date)"; t0=$(date +%s)
  PATH="$HOME/$venv/bin:$PATH" "$HOME/$venv/bin/python" -m data_processing.vllm_run --model "$model" --split-dir "$S" --out-dir "$O/$name" --n 1 --prompt-version 1 \
      --max-prompt-tokens 24000 --max-model-len 40960 --chunk 150 --dump-raw 2 "$@"
  echo "=== $name exit $? seconds $(( $(date +%s) - t0 ))"; }
for e in none low medium high; do run venv_inkling inkling_$e thinkingmachines/Inkling-2-Mini --reasoning-effort $e --max-new-tokens 12288; done
run venv_vllm qwen3coder Qwen/Qwen3-Coder-30B-A3B-Instruct --max-new-tokens 512
for e in low medium high; do run venv_vllm gptoss20b_$e openai/gpt-oss-20b --reasoning-effort $e --max-new-tokens 12288 --reindent; done
echo "=== done $(date)"
