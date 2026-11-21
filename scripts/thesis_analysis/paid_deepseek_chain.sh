#!/usr/bin/bash
# deepseek/deepseek-v4.1-flash (0.05 / 0.20 USD per 1M tokens, reasoning, effort low) via OpenRouter, variants A and B, one candidate, 8192 output tokens (empty answers retried at 16384).
# Stage 1: the 59 Java / 61 Python cases of the agent comparison; stage 2: the rest of the Java subset only (Python full set skipped to stay within the credit). GLOBAL guard on the real account usage
# (default 9.30 USD) before every run; per-run --max-cost caps. Resumable.
cd "$HOME/PyTaRGET" || exit 1
GUARD="${GUARD:-9.30}"; PY="$HOME/venv_vllm/bin/python"
over() { u=$(PYTHONPATH=. $HOME/venv3.12/bin/python scripts/thesis_analysis/or_usage.py 2>/dev/null || echo 99); $HOME/venv3.12/bin/python -c "import sys; sys.exit(0 if float('$u') >= $GUARD else 1)"; r=$?; echo "   account usage $u USD (guard $GUARD)"; return $r; }
run() { lang="$1"; variant="$2"; stage="$3"; cap="$4"
  if over; then echo "GUARD reached: skipping ds $lang$variant"; return; fi
  out="$HOME/eval_paid/ds_${lang}${variant}"
  args=(--model deepseek/deepseek-v4.1-flash --allow-paid --out-dir "$out" --n 1 --max-new-tokens 8192 --requests-per-minute 30 --workers 3 --max-cost "$cap" --price-in 0.05 --price-out 0.20 --reasoning-effort low)
  if [ "$lang" = java ]; then args+=(--java --split-dir "$HOME/eval_results_java_llm/contrast273"); ctx="$HOME/java_ctx/context.json"; ids="$HOME/eval_paid/ids_java59.json"
  else args+=(--split-dir "$HOME/eval_results_perproject_llm/contrast300"); ctx="$HOME/eval_ablation/context.json"; ids="$HOME/eval_paid/ids_python61.json"; fi
  [ "$variant" = B ] && args+=(--context-variant B --context-file "$ctx")
  [ "$stage" = 1 ] && args+=(--only-ids "$ids")
  echo "=== ds $lang $variant stage $stage $(date '+%H:%M')"
  HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -u -m data_processing.openrouter_run "${args[@]}" --max-requests 500 2>&1 | grep -v Warning | tail -1 | cut -c1-200
  HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -u -m data_processing.openrouter_run "${args[@]}" --retry-empty --max-new-tokens 16384 --max-requests 80 2>&1 | grep -v Warning | tail -1 | cut -c1-200
}
for lang in java py; do for v in A B; do run "$lang" "$v" 1 0.15; done; done
for v in A B; do run java "$v" 2 0.70; done
echo "=== chain done $(date '+%F %H:%M')"; PYTHONPATH=. $HOME/venv3.12/bin/python scripts/thesis_analysis/or_usage.py
