#!/usr/bin/bash
# WP5 level 3, JAVA: mini-SWE-agent with Qwen3-Coder-30B-A3B served by vLLM, on the prepared Java cases (scripts/agent/prepare_cases_java.py); Maven runs offline.
# The WHOLE job runs in a private network namespace (`unshare -rn`): no internet, so the agent cannot fetch the repaired test; vLLM and the agent talk over the
# loopback. Each case runs in its own mount namespace that hides home, datasets and tokens (scripts/agent/sandbox_inner.sh).
# Resumable: finished cases (result.json) are skipped. MAX_CASES limits a trial run.
#   sbatch --export=ALL,MAX_CASES=3 slurm/run_agent_java_qwen.sbatch      # first trial
#   sbatch slurm/run_agent_java_qwen.sbatch                                # all prepared cases
#SBATCH -J agent-java-qwen
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
SPLIT_DIR="${SPLIT_DIR:-$HOME/eval_results_java_llm/contrast300}"; ROOT="${ROOT:-$HOME/agent_java}"; OUT="${OUT:-$HOME/eval_agent/java_qwen3coder}"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/PyTaRGET" VENV_VLLM="$HOME/venv_vllm" MAX_CASES="${MAX_CASES:-}"
export PATH="$HOME/venv_vllm/bin:$PATH" WORKERS="${WORKERS:-6}" SPLIT_DIR ROOT OUT
echo "=== start $(date) on $(hostname)"
# everything below runs without network
unshare -rn bash -c '
  ip link set lo up
  vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name qwen3-coder --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder > "$OUT/vllm.log" 2>&1 &
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
  $VENV_VLLM/bin/python scripts/agent/run_cases_java.py --split-dir "$SPLIT_DIR" --root "$ROOT" --out "$OUT" --workers "$WORKERS" $limit
  kill $server; wait $server 2>/dev/null
'
echo "=== done $(date)"
