"""
Prepare Java agent cases (needs the NETWORK: git clone, Maven downloads): for a nested sample of the Java contrast subset whose reference passes in our environment,
a sibling-safe workspace per case (data_processing/agent_workspace_java.py) and one run of the broken test with Maven ONLINE, which (a) shows the case is really
broken and runnable (failure or compile_error) and (b) fills the shared Maven repository, so that the agent can later work OFFLINE (`mvn -o`).
  PYTHONPATH=. ~/venv3.12/bin/python scripts/agent/prepare_cases_java.py --root ~/agent_java --n 60
Resumable (manifest.json). Runs third-party build code: not on a login node.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from data_processing import agent_workspace_java as aj
from data_processing.java_plausibility import JavaPlausibility, VALID, find_parent_pom, detect_java_versions, pick_java_home, MVN_SKIPS

R = Path.home() / "eval_results_java_llm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=60, help="usable cases wanted")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-per-repo", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.root)
    rows = json.load(open(R / "contrast300/splits/test.json"))
    union = json.load(open(Path.home() / "eval_java_plaus/union.json"))
    in273 = {int(v["index300"]) for v in json.load(open(R / "contrast273/subset_meta.json"))["cases"].values()}
    pool = sorted(i for i in in273 if union.get(str(i), {}).get("status") == "evaluated")      # reference passes in our environment
    random.Random(args.seed).shuffle(pool)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    runner = JavaPlausibility(root / "jwork", timeout=900, m2=root / "m2")
    per_repo = {}
    for e in manifest.values():
        if e["status"] == "ok":
            per_repo[e["repo"]] = per_repo.get(e["repo"], 0) + 1
    for i in pool:
        if sum(1 for e in manifest.values() if e["status"] == "ok") >= args.n:
            break
        if str(i) in manifest:
            continue
        row, ws = rows[i], root / "ws" / str(i)
        repo = row["ID"].split(":")[0]
        if per_repo.get(repo, 0) >= args.max_per_repo:
            continue
        entry = {"repo": repo, "id": row["ID"]}
        try:
            info = aj.prepare_workspace(row, ws)
            cls, method = info["class"], info["method"]
            status, log = runner.run_test(ws, Path(row["bPath"]), cls, method)
            java = log.split("\n", 1)[0]
            pom = find_parent_pom(ws / row["bPath"], ws)
            module = str(pom.parent.relative_to(ws)) if pom is not None else "."
            command = f"mvn test -Dtest={cls}#{method}" if module == "." else f"mvn test -pl {module} --also-make -Dtest={cls}#{method}"
            versions = detect_java_versions((ws / "pom.xml").read_text(errors="ignore")) if (ws / "pom.xml").exists() else []
            java_version, java_home = pick_java_home(runner.homes, versions, java if java in runner.homes else None)
            entry.update({"status": "ok" if status in ("failure", "compile_error") else f"not_prepared_{status}", "ws": str(ws), "java": java_version,
                          "java_home": java_home, "command": command, "verdict": status, "method_lines": info["method_lines"],
                          "other_test_lines_changed": info["other_test_lines_changed"]})
            # keep the pom edits the build applied (plugins stripped like the scorer does) in the snapshot; compiled classes stay (faster offline builds)
            aj._run(["git", "add", "-u"], cwd=ws)
            aj._run(["git", *aj.GIT_USER, "commit", "--quiet", "--amend", "--no-edit"], cwd=ws)
            if entry["status"] == "ok":
                per_repo[repo] = per_repo.get(repo, 0) + 1
        except Exception as error:
            entry.update({"status": "error", "detail": repr(error)[:300]})
        manifest[str(i)] = entry
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))
        print(i, repo, entry["status"], flush=True)
    ok = sum(1 for e in manifest.values() if e["status"] == "ok")
    print(f"{ok} usable cases of {len(manifest)} tried")


if __name__ == "__main__":
    sys.exit(main())
