"""Warm the shared Maven repository for the OFFLINE agent run: for every usable Java case, put the REPAIRED method into the workspace, run the test once with Maven
ONLINE (this downloads what only a test that actually runs needs, e.g. the surefire provider; it also checks that the reference passes in this workspace), and restore
the broken text. Writes warm.json next to the manifest ({case: status}; the preparation owns manifest.json); the agent runner only takes cases whose warm status is "success". Needs the NETWORK. Resumable.
  PYTHONPATH=. ~/venv3.12/bin/python scripts/agent/warm_cases_java.py --root ~/agent_java"""
import argparse
import json
import subprocess
from pathlib import Path

from data_processing import agent_workspace_java as aj
from data_processing.java_plausibility import JavaPlausibility, SUCCESS

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
args = ap.parse_args()
root = Path(args.root)
rows = json.load(open(Path.home() / "eval_results_java_llm/contrast300/splits/test.json"))
manifest_path, warm_path = root / "manifest.json", root / "warm.json"
runner = JavaPlausibility(root / "jwork", timeout=900, m2=root / "m2")
while True:                                           # re-read the manifest: the preparation may still be adding cases
    manifest = json.loads(manifest_path.read_text())
    warm = json.loads(warm_path.read_text()) if warm_path.exists() else {}
    todo = [i for i, e in manifest.items() if e["status"] == "ok" and i not in warm]
    if not todo:
        break
    i = todo[0]
    e, row = manifest[i], rows[int(i)]
    ws = Path(e["ws"])
    cls, method = aj.test_class_and_method(row)
    aj.swap_method(ws, row, e["method_lines"], "after")
    try:
        status, log = runner.run_test(ws, Path(row["bPath"]), cls, method, forced_java=e["java"])
    finally:
        subprocess.run(["git", "checkout", "--quiet", "--", row["bPath"]], cwd=ws)          # back to the broken snapshot text
    warm[i] = status
    warm_path.write_text(json.dumps(warm))
    print(i, e["repo"], status, flush=True)
print("warm-up finished")
