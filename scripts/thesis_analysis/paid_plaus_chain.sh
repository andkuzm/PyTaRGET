#!/usr/bin/bash
# Python plausibility (with span control) of the paid-model runs prepared by prep_paid_plaus.py, one run after the other (61 cases each).
cd "$HOME/PyTaRGET" || exit 1
for name in "$@"; do
  echo "=== $name $(date '+%F %H:%M')"
  PYTHONPATH=. "$HOME/venv3.12/bin/python" -u -m data_processing.plausibility_test --results "$HOME/eval_paid_plaus" --models "$name" \
      --workdir "$CLAUDE_JOB_DIR/tmp/plaus_work_paid" --workers 3 --k 1 --out-name plausibility_v3.json --span-control 2>&1 | tail -2 | cut -c1-300
done
echo "=== all done $(date '+%F %H:%M')"
