#!/usr/bin/bash
# Free OpenRouter models on the Java contrast subset without LGPL projects (273 cases, WP2): ernie and particle-1 in parallel, then nemotron.
# One candidate per case. Resumable (the same command continues). Only ':free' slugs are accepted by the runner.
cd "$HOME/PyTaRGET" || exit 1
run() { # slug dir workers
  HF_HUB_OFFLINE=1 PYTHONPATH=. "$HOME/venv_vllm/bin/python" -u -m data_processing.openrouter_run --model "$1" --java \
    --split-dir "$HOME/eval_results_java_llm/contrast273" --out-dir "$HOME/eval_openrouter/$2" --n 1 --reasoning-effort low \
    --max-new-tokens 8192 --requests-per-minute 8 --workers "$3" --max-requests 450 2>&1 | grep -v Warning | tail -3 | cut -c1-200
}
echo "=== $(date '+%F %H:%M') ernie + particle-1"
run baidu/ernie-5.5-flash-preview:free java273_baidu_ernie-5.5-flash-preview_free 2 &
run particle-ai/particle-1:free java273_particle-ai_particle-1_free 2 &
wait
echo "=== $(date '+%F %H:%M') nemotron-3-ultra"
run nvidia/nemotron-3-ultra-550b-a55b:free java273_nvidia_nemotron-3-ultra-550b-a55b_free 3
echo "=== $(date '+%F %H:%M') all three ended"
