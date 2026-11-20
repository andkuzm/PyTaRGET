#!/usr/bin/bash
# Paid OpenRouter models on the Java and Python contrast cases, variants A (baseline prompt) and B (+ failure output), one candidate each, 8192 output tokens (empty answers retried at 16384).
# Stage 1: the 59 Java / 61 Python cases of the agent comparison; stage 2: the rest of the subsets (257 Java / 300 Python). Every run has a --max-cost cap and a GLOBAL guard: before each run
# the real account usage is read and nothing more is started once it reaches GUARD (default 8.50 USD). Resumable: the same command continues. Needs ~/eval_paid/ids_*.json.
cd "$HOME/PyTaRGET" || exit 1
GUARD="${GUARD:-8.50}"
PY="$HOME/venv_vllm/bin/python"
MODELS=("qwen/qwen3-coder-next:0.12:0.80:no:qn:0.5" "qwen/qwen3-coder:0.30:1.00:no:q3c:0.5" "mistralai/devstral-2512:0.40:2.00:no:dev:0.6" "z-ai/glm-5.3:0.07:1.40:yes:glm:0.8" "moonshotai/kimi-k2.5:0.50:2.50:yes:kimi:1.5")
over() { u=$(PYTHONPATH=. $HOME/venv3.12/bin/python scripts/thesis_analysis/or_usage.py 2>/dev/null || echo 99); $HOME/venv3.12/bin/python -c "import sys; sys.exit(0 if float('$u') >= $GUARD else 1)"; r=$?; echo "   account usage $u USD (guard $GUARD)"; return $r; }
run() { # model pin pout reasoning short cap lang variant stage
  IFS=: read -r model pin pout reas short cap <<< "$1"; lang="$2"; variant="$3"; stage="$4"
  if over; then echo "GUARD reached: skipping $short $lang$variant"; return; fi
  out="$HOME/eval_paid/${short}_${lang}${variant}"
  args=(--model "$model" --allow-paid --out-dir "$out" --n 1 --max-new-tokens 8192 --requests-per-minute 30 --workers 3 --max-cost "$cap" --price-in "$pin" --price-out "$pout")
  [ "$reas" = yes ] && args+=(--reasoning-effort low)
  if [ "$lang" = java ]; then args+=(--java --split-dir "$HOME/eval_results_java_llm/contrast273"); ctx="$HOME/java_ctx/context.json"; ids="$HOME/eval_paid/ids_java59.json"
  else args+=(--split-dir "$HOME/eval_results_perproject_llm/contrast300"); ctx="$HOME/eval_ablation/context.json"; ids="$HOME/eval_paid/ids_python61.json"; fi
  [ "$variant" = B ] && args+=(--context-variant B --context-file "$ctx")
  [ "$stage" = 1 ] && args+=(--only-ids "$ids")
  echo "=== $short $lang $variant stage $stage $(date '+%H:%M')"
  HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -u -m data_processing.openrouter_run "${args[@]}" --max-requests 500 2>&1 | grep -v Warning | tail -1 | cut -c1-200
  HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -u -m data_processing.openrouter_run "${args[@]}" --retry-empty --max-new-tokens 16384 --max-requests 80 2>&1 | grep -v Warning | tail -1 | cut -c1-200
}
for stage in 1 2; do
  for m in "${MODELS[@]}"; do
    # stage 2 (full subsets) skips kimi: its reasoning writes 6-12k tokens per answer (~$0.02 per case, ~$5.5 for the 240 stage-1 runs); set KIMI_STAGE2=1 to extend it
    if [ "$stage" = 2 ] && [[ "$m" == moonshotai/* ]] && [ "${KIMI_STAGE2:-0}" != 1 ]; then echo "skipping kimi stage 2"; continue; fi
    for lang in java py; do for v in A B; do run "$m" "$lang" "$v" "$stage"; done; done
  done
done
echo "=== chain done $(date '+%F %H:%M')"; PYTHONPATH=. $HOME/venv3.12/bin/python scripts/thesis_analysis/or_usage.py
