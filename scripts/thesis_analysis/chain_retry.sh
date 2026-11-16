#!/usr/bin/bash
# after the nemotron chain ended: ask again for the empty answers of the three free-model runs with a larger token limit
while pgrep -f "chain_nemotron" >/dev/null || pgrep -f "openrouter_run --model" >/dev/null; do sleep 30; done
cd /gpfs/helios/home/kandrei/PyTaRGET
for M in baidu/ernie-5.5-flash-preview:free particle-ai/particle-1:free nvidia/nemotron-3-ultra-550b-a55b:free; do
  D=~/eval_openrouter/contrast300_$(echo $M | tr '/:' '__')
  echo "=== retry empty answers of $M with 16384 tokens"
  HF_HUB_OFFLINE=1 PYTHONPATH=. ~/venv_vllm/bin/python -u -m data_processing.openrouter_run --model $M \
    --split-dir ~/eval_results_perproject_llm/contrast300 --out-dir $D --retry-empty --n 1 --reasoning-effort low \
    --max-new-tokens 16384 --requests-per-minute 8 --workers 2 --max-requests 150 2>&1 | grep -v Warning | tail -4 | cut -c1-220
done
echo "=== retries finished"
