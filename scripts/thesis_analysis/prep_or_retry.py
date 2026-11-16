"""Plausibility input for ONLY the cases whose first-pass verdict was 'empty' (or that had no answer then), taken from the
retried run directory, with code fences stripped. Output layout: ~/eval_openrouter_plaus/<name>_retry/0.8"""
import json, sys
from pathlib import Path
from data_processing.openrouter_run import strip_code_fence
from data_processing import prediction_files
name, run_dir = sys.argv[1], Path(sys.argv[2])
H = Path.home()
first = json.load(open(H / f"eval_openrouter_plaus/{name}/0.8/plausibility_v3.json"))["results"]
redo = {int(k) for k, v in first.items() if v["status"] == "evaluated" and v["verdicts"] and v["verdicts"][0] == "empty"}
scored = {int(k) for k in first}
out = H / f"eval_openrouter_plaus/{name}_retry/0.8"
(out / "splits").mkdir(parents=True, exist_ok=True)
link = out / "splits" / "test.json"
if not link.exists():
    link.symlink_to(H / "eval_results_perproject_llm/contrast300/splits/test.json")
recs = prediction_files.load(run_dir / "test_predictions.json")
keep = []
for r in recs:
    if r["ID"] in redo or r["ID"] not in scored:
        r["preds"] = [strip_code_fence(x) for x in r["preds"]]
        keep.append(r)
now_nonempty = sum(1 for r in keep if any(x.strip() for x in r["preds"]))
prediction_files.save(keep, out / "test_predictions.json")
print(f"{name}: {len(redo)} cases were empty in the first pass; {len(keep)} to score again, of which {now_nonempty} now have an answer -> {out}")
