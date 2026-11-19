#!/usr/bin/bash
# Free particle-1 on the 273 Java cases with the context variants B and C (failure output / definitions), one after the other; each run is followed by a retry of its empty answers
# (16384 tokens). Resumable (same commands continue). Only ':free' models; ~600 requests in total.
cd "$HOME/PyTaRGET" || exit 1
for V in B C; do
  OUT="$HOME/eval_ablation_java/particle1_$V"
  echo "=== $V first pass $(date '+%F %H:%M')"
  HF_HUB_OFFLINE=1 PYTHONPATH=. "$HOME/venv_vllm/bin/python" -u -m data_processing.openrouter_run --model particle-ai/particle-1:free --java \
    --split-dir "$HOME/eval_results_java_llm/contrast273" --out-dir "$OUT" --n 1 --reasoning-effort low --max-new-tokens 8192 --requests-per-minute 8 --workers 2 \
    --max-requests 380 --context-variant "$V" --context-file "$HOME/java_ctx/context.json" 2>&1 | grep -v Warning | tail -2 | cut -c1-220
  echo "=== $V retry of empty answers $(date '+%F %H:%M')"
  HF_HUB_OFFLINE=1 PYTHONPATH=. "$HOME/venv_vllm/bin/python" -u -m data_processing.openrouter_run --model particle-ai/particle-1:free --java --retry-empty \
    --split-dir "$HOME/eval_results_java_llm/contrast273" --out-dir "$OUT" --n 1 --reasoning-effort low --max-new-tokens 16384 --requests-per-minute 6 --workers 2 \
    --max-requests 60 --context-variant "$V" --context-file "$HOME/java_ctx/context.json" 2>&1 | grep -v Warning | tail -2 | cut -c1-220
done
echo "=== chain done $(date '+%F %H:%M')"
