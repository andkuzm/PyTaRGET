"""Prepare a plausibility layout for an OpenRouter run: fence-stripped copy of its predictions next to a link to the subset's test split."""
import json, os, sys
from pathlib import Path
from data_processing.openrouter_run import strip_code_fence
from data_processing import prediction_files
name, run_dir = sys.argv[1], Path(sys.argv[2])
out = Path.home() / "eval_openrouter_plaus" / name / "0.8"
(out / "splits").mkdir(parents=True, exist_ok=True)
link = out / "splits" / "test.json"
if not link.exists():
    link.symlink_to(Path.home() / "eval_results_perproject_llm" / "contrast300" / "splits" / "test.json")
recs = prediction_files.load(run_dir / "test_predictions.json")
changed = 0
for r in recs:
    new = [strip_code_fence(x) for x in r["preds"]]
    changed += new != r["preds"]
    r["preds"] = new
prediction_files.save(recs, out / "test_predictions.json")
print(f"{name}: {len(recs)} cases, {changed} with a code fence removed -> {out}")
