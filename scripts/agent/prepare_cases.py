"""
Prepare agent cases (needs the NETWORK: git clone, pip): for the nested sample of a split, a leak-safe workspace per case (data_processing/agent_workspace.py)
and one virtualenv per repository (pytest plugins + requirements*.txt; the project itself comes from PYTHONPATH=<workspace>[:<workspace>/src], not from an
editable install, so a venv shared by several commits never imports another case's code). Each workspace's test is run once: a prepared case must FAIL.
  PYTHONPATH=. ~/venv3.12/bin/python scripts/agent/prepare_cases.py --split-dir ~/eval_results_perproject_llm/contrast300 --root ~/agent_work --n 100
Resumable (manifest.json). Runs third-party install/test code: not on a login node.
"""
import argparse
import json
import sys
from pathlib import Path

from data_processing import agent_workspace as aw
from data_processing.openrouter_run import sample_positions
from data_processing.plausibility_test import PlausibilityTester


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sibling-safe", action="store_true", help="revert ALL test files changed by the repair to their broken version (other fixed tests cannot be copied)")
    ap.add_argument("--venvs-from", default=None, help="root of an earlier preparation whose virtualenvs are reused (linked)")
    args = ap.parse_args()
    root = Path(args.root)
    if args.venvs_from and not (root / "venvs").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "venvs").symlink_to(Path(args.venvs_from) / "venvs")
    rows = json.load(open(Path(args.split_dir) / "splits" / "test.json"))
    chosen = sorted(sample_positions(len(rows), args.n, args.seed, nested=True))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tester = PlausibilityTester(root, timeout=180)
    for i in chosen:
        if str(i) in manifest:
            continue
        row, ws = rows[i], root / "ws" / str(i)
        entry = {"repo": row["repository_name"], "test_file": row["relative_path"]}
        try:
            info = aw.prepare_workspace(row, ws, sibling_safe=args.sibling_safe)
            python = tester.python_for(row["repository_name"], ws)
            nodeid = "::".join([row["relative_path"]] + info["nodeid_parts"])
            verdict = tester.run_test(python, ws, nodeid)
            entry.update({"status": "ok" if verdict == "fail" else f"not_prepared_{verdict}", "ws": str(ws), "python": python, "nodeid": nodeid,
                          "broken_verdict": verdict, "test_name": info["test_name"], "repaired_function": info["repaired_function"]})
        except Exception as error:
            entry.update({"status": "error", "detail": repr(error)[:300]})
        manifest[str(i)] = entry
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))
        print(i, entry["repo"], entry["status"], flush=True)
    ok = sum(1 for e in manifest.values() if e["status"] == "ok")
    print(f"{ok} of {len(manifest)} cases are ready (broken test fails in its environment)")


if __name__ == "__main__":
    sys.exit(main())
