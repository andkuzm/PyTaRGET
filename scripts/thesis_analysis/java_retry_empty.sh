#!/usr/bin/bash
# Ask again (16384 new tokens) for the empty answers of ernie and particle on the Java contrast273 subset. Free models only; resumable.
cd "$HOME/PyTaRGET" || exit 1
for pair in "baidu/ernie-5.5-flash-preview:free java273_baidu_ernie-5.5-flash-preview_free" "particle-ai/particle-1:free java273_particle-ai_particle-1_free"; do
  set -- $pair
  HF_HUB_OFFLINE=1 PYTHONPATH=. "$HOME/venv_vllm/bin/python" -u -m data_processing.openrouter_run --model "$1" --java --retry-empty \
    --split-dir "$HOME/eval_results_java_llm/contrast273" --out-dir "$HOME/eval_openrouter/$2" --n 1 --reasoning-effort low \
    --max-new-tokens 16384 --requests-per-minute 6 --workers 2 --max-requests 150 2>&1 | grep -v Warning | tail -3 | cut -c1-200
done
echo "retry done $(date '+%F %H:%M')"
