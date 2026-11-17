#!/usr/bin/bash
# Finish the plausibility scoring of the training-size experiment: CodeGen sets first (resume), then PLBART sets. Resumable.
#   bash scripts/thesis_analysis/trainfrac_scoring_chain.sh <seconds for CodeGen> <seconds for PLBART>
cd "$HOME/PyTaRGET" || exit 1
for pass in "codegen $1" "plbart $2"; do
  set -- $pass; name=$1; secs=$2
  models=""; for f in 20 40 60 80; do models="$models ${name}_f$f"; done
  echo "=== $(date '+%F %H:%M') pass $name starts (limit $secs s)"
  PYTHONPATH=. timeout -s INT "$secs" "$HOME/venv3.12/bin/python" -u -m data_processing.plausibility_test \
     --results "$HOME/eval_trainfrac_plaus" --models $models --workdir "${CLAUDE_JOB_DIR:-/tmp}/tmp/plaus_trainfrac" \
     --workers 3 --k 5 --out-name plausibility_v3.json --span-control 2>&1 | tail -3 | cut -c1-200
  echo "=== $(date '+%F %H:%M') pass $name ended"
done
