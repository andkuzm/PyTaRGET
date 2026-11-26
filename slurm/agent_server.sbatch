#!/usr/bin/bash
# Inkling-2-Mini as an AGENT delegate for the repair pipeline: vLLM and mini-SWE-agent in one GPU job without network (`unshare -rn`); tools/delegate (alias "inkling-agent") talks to it only through files in
# ~/delegation_exp/agent_queue (scripts/agent/delegate_worker.py). While the job is ready, <queue>/heartbeat is refreshed (also by the sidecar during long requests). At most one such job at a time.
#   sbatch slurm/agent_server.sbatch          (6 h)
#SBATCH -J agent-server
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
export QUEUE="$HOME/delegation_exp/agent_queue" M2="/tmp/agent_m2_${USER}_${SLURM_JOB_ID}" EFFORT="${EFFORT:-low}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/PyTaRGET" PATH="$HOME/venv_inkling/bin:$PATH"
mkdir -p "$QUEUE"; rm -f "$QUEUE/heartbeat"
cp -r "$HOME/delegation_exp/m2" "$M2"                       # private copy: commands chosen by the model cannot alter the cache the scorer and the pipeline use
echo "=== start $(date) on $(hostname)"
unshare -rn bash -c '
  ip link set lo up
  vllm serve thinkingmachines/Inkling-2-Mini --served-model-name inkling --port 8000 --max-model-len 131072 --gpu-memory-utilization 0.90 \
      --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling --default-chat-template-kwargs "{\"reasoning_effort\": \"$EFFORT\"}" > "$QUEUE/vllm.log" 2>&1 &
  server=$!
  for i in $(seq 1 90); do
    if python3 -c "import urllib.request;urllib.request.urlopen(\"http://localhost:8000/v1/models\",timeout=3)" 2>/dev/null; then ready=1; break; fi
    kill -0 $server 2>/dev/null || { echo "vllm died, see $QUEUE/vllm.log"; exit 1; }
    sleep 10
  done
  [ -n "${ready:-}" ] || { echo "vllm not ready after 15 minutes"; kill $server; exit 1; }
  echo "=== vllm ready $(date)"
  curl -s -m 4 -o /dev/null -w "network probe (expected 000): %{http_code}\n" https://github.com || true
  ( while true; do date +%s > "$QUEUE/alive"; sleep 20; done ) & beat=$!
  "$HOME/venv3.12/bin/python" scripts/agent/delegate_worker.py "$QUEUE" "$M2"
  kill $beat $server; wait $server 2>/dev/null
'
rm -rf "$M2"
echo "=== done $(date)"
