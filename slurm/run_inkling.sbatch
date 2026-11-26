#!/usr/bin/bash
# Inkling-2-Mini (vLLM nightly in venv_inkling, same prompt as the Qwen3-Coder / gpt-oss runs) on one test set; one H200. Resumable: finished cases are kept.
#   sbatch --export=ALL,SPLIT_DIR=<dir with splits/test.json>,OUT_DIR=<output dir>[,EFFORT=low,N_CANDIDATES=5,MAX_PROMPT=24000,MAX_LEN=32768,MAX_NEW=4096,EXTRA_ARGS="--limit 20"] slurm/run_inkling.sbatch
#SBATCH -J run-inkling
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
: "${SPLIT_DIR:?}" "${OUT_DIR:?}"
cd "$HOME/PyTaRGET"
export PATH="$HOME/venv_inkling/bin:$PATH" PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
echo "=== inkling effort=${EFFORT:-low} $(date) on $(hostname)"
"$HOME/venv_inkling/bin/python" -m data_processing.vllm_run --model thinkingmachines/Inkling-2-Mini --split-dir "$SPLIT_DIR" --out-dir "$OUT_DIR" \
    --n "${N_CANDIDATES:-5}" --prompt-version 1 --reasoning-effort "${EFFORT:-low}" --max-prompt-tokens "${MAX_PROMPT:-24000}" --max-model-len "${MAX_LEN:-32768}" --max-new-tokens "${MAX_NEW:-4096}" ${EXTRA_ARGS:-}
echo "=== exit $? $(date)"
