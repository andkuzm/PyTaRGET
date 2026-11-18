#!/usr/bin/bash
# Plausibility (checked rule inputs: --span-control) of the context-ablation runs, one run after the other (model-by-model passes are faster than one big pass).
# usage: ablation_plaus_chain.sh <workers> <run name>...     (layouts made by prep_ablation_plaus.py)
cd "$HOME/PyTaRGET" || exit 1
workers="$1"; shift
for name in "$@"; do
  echo "=== $name $(date '+%F %H:%M')"
  PYTHONPATH=. "$HOME/venv3.12/bin/python" -u -m data_processing.plausibility_test --results "$HOME/eval_ablation_plaus" --models "$name" \
      --workdir "$CLAUDE_JOB_DIR/tmp/plaus_work_abl" --workers "$workers" --k 1 --out-name plausibility_v3.json --span-control 2>&1 | tail -4 | cut -c1-300
done
echo "=== all done $(date '+%F %H:%M')"
