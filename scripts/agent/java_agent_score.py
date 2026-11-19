"""Score the Java agent results with the Java plausibility runner: the agent's final METHOD replaces the whole after-method (data_processing/agent_workspace_java.py),
the reference method is the control. Cases without a usable method (missing, edited outside the method, other tracked files changed) are failures and are
not run. Output: <out>, a result file in the format of JavaPlausibility.run, plus <out>.skipped.json.
  PYTHONPATH=. python scripts/agent/java_agent_score.py --root ~/agent_java --agent-out ~/eval_agent/java_qwen3coder --out ~/eval_agent_plaus/java_qwen3coder.json --workdir <scratch>
Runs third-party build code: not on a login node."""
import argparse
import json
from pathlib import Path

from data_processing import agent_workspace_java as aj
from data_processing.java_plausibility import JavaPlausibility, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--agent-out", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workdir", required=True)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--m2", default=None, help="shared Maven repository (default: <workdir>/m2)")
ap.add_argument("--relaxed", action="store_true", help="score the agent's WHOLE final test file (edits outside the method, e.g. imports or helpers, are allowed); "
                "only cases that changed other tracked files stay failures")
args = ap.parse_args()
rows = json.load(open(Path.home() / "eval_results_java_llm/contrast300/splits/test.json"))
manifest = json.load(open(Path(args.root) / "manifest.json"))
cases, skipped = [], {}
for result_file in sorted(Path(args.agent_out).glob("*/result.json"), key=lambda f: int(f.parent.name)):
    i = result_file.parent.name
    r = json.loads(result_file.read_text())
    manifest_entry = manifest[i]
    if r.get("other_files_touched"):
        skipped[i] = "changed other tracked files"
    elif args.relaxed:
        import subprocess
        ws, row = Path(manifest_entry["ws"]), rows[int(i)]
        snapshot = subprocess.run(["git", "show", f"HEAD:{row['bPath']}"], cwd=ws, capture_output=True, text=True).stdout.split("\n")
        final = (ws / row["bPath"]).read_text(encoding="utf-8", errors="ignore")
        n_after = len(snapshot) - len(row["bSource"]["code"].split("\n")) + len(row["aSource"]["code"].split("\n"))     # lines of the file at the after commit
        derived = dict(row, hunk=dict(row["hunk"], targetChanges=[{"lineNo": n, "type": "ADD", "line": ""} for n in range(1, n_after + 1)]))
        first, last = manifest_entry["method_lines"]
        code = row["aSource"]["code"].split("\n")
        indent = snapshot[first - 1][:len(snapshot[first - 1]) - len(snapshot[first - 1].lstrip())]
        if code[0] == code[0].lstrip():
            code[0] = indent + code[0]
        after_file = "\n".join(snapshot[:first - 1] + code + snapshot[last:])            # the file at the after commit (the control)
        cases.append({"row": derived, "key": i, "preds": [final.rstrip("\n")], "target": after_file.rstrip("\n")})
    elif r.get("test_file_changed_outside_method"):
        skipped[i] = "edited outside the method"
    elif not r.get("method_found"):
        skipped[i] = "method missing"
    else:
        row = rows[int(i)]
        cases.append({"row": aj.derived_row(row), "key": i, "preds": [r["method"]], "target": row["aSource"]["code"]})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(skipped, open(args.out + ".skipped.json", "w"))
print(f"{len(cases)} cases to score, {len(skipped)} skipped by rule: {skipped}", flush=True)
results = JavaPlausibility(args.workdir, m2=args.m2, log_dir=Path(args.out).with_suffix(".logs")).run(cases, args.out, args.workers)
print(json.dumps(summarize(results)))
