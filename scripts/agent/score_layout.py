"""Layout for scoring agent results with plausibility_test.py: a derived split whose reference is the WHOLE repaired test function and whose candidate is
the agent's final test function (cases with no function after the agent's edits are failures and are not in the layout).
Output: <out>/0.8/{splits/test.json, test_predictions.json, case_map.json}; ID of a prediction = position in the derived split.
Usage: PYTHONPATH=. python scripts/agent/score_layout.py <split dir> <agent work root> <agent output dir> <out dir>"""
import json, subprocess, sys
from pathlib import Path
from data_processing import agent_workspace as aw, prediction_files
from data_processing.plausibility_test import find_repair_site, patch_source
split_dir, root, agent_out, out = (Path(a) for a in sys.argv[1:5])
rows = json.load(open(split_dir / "splits" / "test.json"))
manifest = json.load(open(root / "manifest.json"))
derived, preds, case_map, skipped = [], [], {}, {}
for result_file in sorted(agent_out.glob("*/result.json"), key=lambda f: int(f.parent.name)):
    i = result_file.parent.name
    result, row, ws = json.loads(result_file.read_text()), rows[int(i)], Path(manifest[i]["ws"])
    if not result.get("function_found") or not result.get("function"):
        skipped[i] = "function missing"
        continue
    name = aw.test_name_of(row)
    snapshot = subprocess.run(["git", "show", f"HEAD:{row['relative_path']}"], cwd=ws, capture_output=True, text=True).stdout
    site = find_repair_site(snapshot, name, aw.broken_lines(row))
    repaired = function_text = None
    if site is not None:
        repaired = patch_source(snapshot, site, row["output"])                 # snapshot with the reference repair put back
        function_text = aw.function_source(repaired, name)
    if function_text is None:
        skipped[i] = "repaired function not reconstructable"
        continue
    new = dict(row, output=function_text[0])
    case_map[len(derived)] = i
    derived.append(new)
    preds.append({"ID": len(derived) - 1, "target": new["output"], "preds": [result["function"]]})
(out / "0.8" / "splits").mkdir(parents=True, exist_ok=True)
json.dump(derived, open(out / "0.8" / "splits" / "test.json", "w"))
prediction_files.save(preds, out / "0.8" / "test_predictions.json")
json.dump({"case_map": case_map, "skipped": skipped}, open(out / "0.8" / "case_map.json", "w"))
print(f"{len(derived)} cases in the layout, skipped {len(skipped)}: {skipped}")
