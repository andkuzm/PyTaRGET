#!/usr/bin/bash
# start nemotron-3-ultra (free) on the contrast subset when the ernie and particle runs have both ended
while pgrep -f "openrouter_run --model baidu" >/dev/null || pgrep -f "openrouter_run --model particle-ai" >/dev/null; do sleep 20; done
cd /gpfs/helios/home/kandrei/PyTaRGET
HF_HUB_OFFLINE=1 PYTHONPATH=. ~/venv_vllm/bin/python -u -m data_processing.openrouter_run --model nvidia/nemotron-3-ultra-550b-a55b:free \
  --split-dir ~/eval_results_perproject_llm/contrast300 --out-dir ~/eval_openrouter/contrast300_nvidia_nemotron-3-ultra-550b-a55b_free \
  --n 1 --reasoning-effort low --max-new-tokens 8192 --requests-per-minute 8 --workers 3 --max-requests 450 2>&1 | grep -v Warning | cut -c1-220
