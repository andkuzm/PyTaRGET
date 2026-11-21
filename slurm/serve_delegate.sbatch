#!/usr/bin/bash
# Local delegate for the delegation experiment: serves Qwen3-Coder-30B-A3B with vLLM (OpenAI-compatible) on one GPU so that `tools/delegate` (alias "local") can use it instead of OpenRouter.
# The server listens on the cluster network (the delegate runs on another node) and is protected by a random API token (~/.config/delegation/local_token, mode 600).
# When the model is ready, the address is written to ~/delegation_exp/local_delegate_endpoint.txt (removed again when the job ends). At most ONE such job at a time.
#   sbatch slurm/serve_delegate.sbatch          # 6 h on one H200
#SBATCH -J serve-delegate
#SBATCH -p gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "$HOME/PyTaRGET"
EP="$HOME/delegation_exp/local_delegate_endpoint.txt"; TOKEN_FILE="$HOME/.config/delegation/local_token"
mkdir -p "$(dirname "$TOKEN_FILE")" "$HOME/delegation_exp/scratch"; chmod 700 "$(dirname "$TOKEN_FILE")"
[ -s "$TOKEN_FILE" ] || { umask 077; python3 -c "import secrets;print(secrets.token_hex(24))" > "$TOKEN_FILE"; }; chmod 600 "$TOKEN_FILE"
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1])")
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PATH="$HOME/venv_vllm/bin:$PATH"
rm -f "$EP"
echo "=== start $(date) on $(hostname) port $PORT"
LOG="$HOME/delegation_exp/scratch/vllm_delegate_${SLURM_JOB_ID}.log"
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name qwen3-coder --host 0.0.0.0 --port "$PORT" --api-key "$(cat "$TOKEN_FILE")" \
    --max-model-len 32768 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
server=$!
cleanup() { rm -f "$EP"; kill "$server" 2>/dev/null; wait "$server" 2>/dev/null; echo "=== stopped $(date)"; }
trap cleanup TERM EXIT
for i in $(seq 1 120); do                                  # up to 20 minutes for the model to load
    if curl -s -m 3 -H "Authorization: Bearer $(cat "$TOKEN_FILE")" "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$server" 2>/dev/null || { echo "vllm died, see $LOG"; exit 1; }
    sleep 10
done
[ -n "${ready:-}" ] || { echo "vllm not ready after 20 minutes, see $LOG"; exit 1; }
echo "http://$(hostname):$PORT/v1" > "$EP"
echo "=== ready $(date): $(cat "$EP")"
wait "$server"
