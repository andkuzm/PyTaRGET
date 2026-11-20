"""Plausibility layouts for the paid-model PYTHON runs: ~/eval_paid_plaus/<run>/0.8/{splits/test.json (link to contrast300), test_predictions.json}, restricted to the 61 common
cases (positions in ~/eval_paid/ids_python61.json) so that every run is scored on the same cases quickly; fences stripped. Usage: python prep_paid_plaus.py [--all] <run>..."""
import json, sys
from pathlib import Path
from data_processing.openrouter_run import strip_code_fence
from data_processing import prediction_files
H = Path.home()
args = [a for a in sys.argv[1:] if a != "--all"]
keep_all = "--all" in sys.argv
ids = set(json.load(open(H / "eval_paid/ids_python61.json")))
for name in args:
    out = H / "eval_paid_plaus" / name / "0.8"
    (out / "splits").mkdir(parents=True, exist_ok=True)
    link = out / "splits" / "test.json"
    if not link.exists():
        link.symlink_to(H / "eval_results_perproject_llm/contrast300/splits/test.json")
    recs = [r for r in prediction_files.load(H / "eval_paid" / name / "test_predictions.json") if keep_all or r["ID"] in ids]
    for r in recs:
        r["preds"] = [strip_code_fence(x) for x in r["preds"]]
    prediction_files.save(recs, out / "test_predictions.json")
    print(f"{name}: {len(recs)} cases -> {out}")
