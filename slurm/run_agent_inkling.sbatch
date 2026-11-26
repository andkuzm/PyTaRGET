#!/usr/bin/bash
# mini-SWE-agent (tool-calling bash agent) with Inkling-2-Mini served by the vLLM nightly (venv_inkling), on the prepared Python cases (scripts/agent/prepare_cases.py).
# The WHOLE job runs in a private network namespace (`unshare -rn`): no internet, so the agent cannot fetch the repaired test; vLLM and the agent talk over the
# loopback. Each case runs in its own mount namespace that hides home, datasets and tokens (scripts/agent/sandbox_inner.sh).
# Resumable: finished cases (result.json) are skipped. MAX_CASES limits a trial run.
#   sbatch --export=ALL,MAX_CASES=3 slurm/run_agent_inkling.sbatch      # first trial
#   sbatch slurm/run_agent_inkling.sbatch                                # all prepared cases
#SBATCH -J agent-inkling
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
SPLIT_DIR="${SPLIT_DIR:-$HOME/eval_results_perproject_llm/contrast300}"; ROOT="${ROOT:-$HOME/agent_inkling}"; OUT="${OUT:-$HOME/eval_agent/inkling_low_smoke}"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/PyTaRGET" VENV_VLLM="$HOME/venv_vllm" VENV_INK="$HOME/venv_inkling" MAX_CASES="${MAX_CASES:-}"
export PATH="$HOME/venv_inkling/bin:$PATH" EFFORT="${EFFORT:-low}" WORKERS="${WORKERS:-6}" SPLIT_DIR ROOT OUT
echo "=== start $(date) on $(hostname)"
# everything below runs without network
unshare -rn bash -c '
  ip link set lo up
  vllm serve thinkingmachines/Inkling-2-Mini --served-model-name inkling --port 8000 --max-model-len "${MODEL_LEN:-131072}" --gpu-memory-utilization 0.90 \
      --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling --default-chat-template-kwargs "{\"reasoning_effort\": \"$EFFORT\"}" > "$OUT/vllm.log" 2>&1 &
  server=$!
  for i in $(seq 1 90); do                       # up to 15 minutes for the model to load
    if python3 -c "import urllib.request;urllib.request.urlopen(\"http://localhost:8000/v1/models\",timeout=3)" 2>/dev/null; then ready=1; break; fi
    kill -0 $server 2>/dev/null || { echo "vllm died, see $OUT/vllm.log"; exit 1; }
    sleep 10
  done
  [ -n "${ready:-}" ] || { echo "vllm not ready after 15 minutes"; kill $server; exit 1; }
  echo "=== vllm ready $(date)"
  curl -s -m 4 -o /dev/null -w "network probe (expected 000): %{http_code}\n" https://github.com || true
  limit=""; [ -n "$MAX_CASES" ] && limit="--limit $MAX_CASES"
  $VENV_VLLM/bin/python scripts/agent/run_cases.py --split-dir "$SPLIT_DIR" --root "$ROOT" --out "$OUT" --workers "$WORKERS" --step-limit "${STEP_LIMIT:-80}" --model hosted_vllm/inkling --config mini_inkling.yaml $limit
  kill $server; wait $server 2>/dev/null
'
echo "=== done $(date)"
