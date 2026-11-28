"""Build a delegation task: a Java project at the AFTER state of its source with the OLD tests, a SOURCE_CHANGES.diff of the main-source changes (what the "developer" just changed), tools, and the
reference material for scoring kept OUTSIDE the workspace.
  python prepare_task.py synth <task id>                      [<project>]   # a synthetic project (~/delegation_exp/synth/<project>, default shop; also ledger)
  python prepare_task.py real <task id> <repo> <aCommit> <bCommit> [<module dir>]   # a commit of a real project (TaRBench): all test files changed by the repair are put back to their old version
Task directory ~/delegation_exp/tasks/<id>/: ws/ (the workspace template), ref/ (reference tests), old_tests/, source_hashes.json, meta.json."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

H = Path.home()
EXP = H / "delegation_exp"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dtools import common  # noqa: E402

WRAPPER = '#!/usr/bin/bash\nexec {py} {tool} "$(cd "$(dirname "$0")/.." && pwd)" "$@"\n'


def run(cmd, cwd=None, timeout=3600, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def hashes(ws, module):
    out = {}
    for p in sorted((Path(ws) / module).rglob("*")):
        rel = p.relative_to(ws)
        if p.is_file() and ("src/main" in str(rel) or rel.name == "pom.xml"):
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def finalize(task, ws, ref_files, old_tests, diff_text, module, java_version, test_root):
    homes = json.load(open(H / "tools/java/java_homes.json"))
    (task / "ref").mkdir(parents=True, exist_ok=True); (task / "old_tests").mkdir(parents=True, exist_ok=True)
    for rel, text in ref_files.items():
        p = task / "ref" / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    for rel, text in old_tests.items():
        p = task / "old_tests" / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    (ws / "SOURCE_CHANGES.diff").write_text(diff_text)
    (ws / "tools").mkdir(exist_ok=True)
    py = str(H / "venv3.12/bin/python")
    for name, script in (("run_tests", "run_tests.py"), ("delegate", "delegate_guard.py"), ("review", "review.py"), ("start", "start.py"), ("repair", "repair.py")):
        f = ws / "tools" / name
        f.write_text(WRAPPER.format(py=py, tool=HERE / "dtools" / script)); f.chmod(0o755)
    cfg = {"java_home": homes[str(java_version)], "maven_home": homes["maven"], "m2": str(EXP / "m2"), "test_roots": [test_root], "module": module,
           "mvn_extra": (["-pl", module, "--also-make"] if module not in ("", ".") else []), "key_file": "~/.config/openrouter/key", "default_delegate": "particle-free",
           "delegates": {
               "particle-free": {"slug": "particle-ai/particle-1:free", "base_url": "https://openrouter.ai/api/v1", "reasoning": "low", "price_in": 0, "price_out": 0},
               "particle": {"slug": "particle-ai/particle-1", "base_url": "https://openrouter.ai/api/v1", "reasoning": "low", "price_in": 0.07, "price_out": 0.50},
               "qwen3-coder": {"slug": "qwen/qwen3-coder", "base_url": "https://openrouter.ai/api/v1", "price_in": 0.30, "price_out": 1.00},
               "local": {"slug": "qwen3-coder", "base_url_file": "~/delegation_exp/local_delegate_endpoint.txt", "key_file": "~/.config/delegation/local_token", "max_tokens": 8192, "price_in": 0, "price_out": 0}}}
    (ws / ".delegation").mkdir(exist_ok=True); (ws / ".delegation" / "config.json").write_text(json.dumps(cfg, indent=1))
    (ws / ".gitignore").write_text("target/\n.delegation/\n")
    shutil.rmtree(ws / ".git", ignore_errors=True)
    run(["git", "init", "-q"], cwd=ws); run(["git", "add", "-A"], cwd=ws)
    run(["git", "-c", "user.name=dev", "-c", "user.email=dev@example.invalid", "commit", "-q", "-m", "current state"], cwd=ws)
    (task / "source_hashes.json").write_text(json.dumps(hashes(ws, module if module != "." else "")))
    return cfg


def synth(task_id, project="shop"):
    S = EXP / "synth" / project; task = EXP / "tasks" / task_id
    shutil.rmtree(task, ignore_errors=True); task.mkdir(parents=True)
    run([str(S / "build_variants.sh"), str(S / "variants")])
    ws = task / "ws"; shutil.copytree(S / "variants" / "task", ws)
    tmp = task / "_tmp"; (tmp / "a").mkdir(parents=True); (tmp / "b").mkdir()
    shutil.copytree(S / "main_before", tmp / "a/src/main/java", dirs_exist_ok=True); shutil.copytree(S / "main_after", tmp / "b/src/main/java", dirs_exist_ok=True)
    diff = run(["diff", "-ruN", "a", "b"], cwd=tmp).stdout; shutil.rmtree(tmp)
    ref, old = {}, {}
    for f in (S / "tests_after").rglob("*.java"): ref[f"src/test/java/{f.relative_to(S / 'tests_after')}"] = f.read_text()
    for f in (S / "tests_before").rglob("*.java"): old[f"src/test/java/{f.relative_to(S / 'tests_before')}"] = f.read_text()
    cfg = finalize(task, ws, ref, old, diff, ".", 17, "src/test/java")
    meta = {"id": task_id, "kind": "synthetic", "java": 17, "test_files": len(old), "diff_chars": len(diff)}
    (task / "meta.json").write_text(json.dumps(meta, indent=1)); print(meta)


def real(task_id, repo, a, b, module):
    task = EXP / "tasks" / task_id; shutil.rmtree(task, ignore_errors=True); task.mkdir(parents=True)
    scratch = EXP / "scratch" / task_id; shutil.rmtree(scratch, ignore_errors=True); scratch.parent.mkdir(parents=True, exist_ok=True)
    assert run(["git", "clone", "-q", "--filter=blob:none", f"https://github.com/{repo}.git", str(scratch)]).returncode == 0, "clone failed"
    assert run(["git", "checkout", "-q", a], cwd=scratch).returncode == 0, "checkout failed"
    prefix = (module.rstrip("/") + "/") if module not in ("", ".") else ""
    names = run(["git", "diff", "--name-status", b, a], cwd=scratch).stdout.splitlines()
    ref, old = {}, {}
    for line in names:
        st, path = line.split("\t")[0][0], line.split("\t")[-1]
        if not (path.startswith(prefix + "src/test/") and path.endswith(".java")): continue
        rel = path[len(prefix):]
        ref[rel] = (scratch / path).read_text() if (scratch / path).exists() else ""
        if st == "A":
            (scratch / path).unlink()
        else:
            old[rel] = run(["git", "show", f"{b}:{path}"], cwd=scratch).stdout; (scratch / path).parent.mkdir(parents=True, exist_ok=True); (scratch / path).write_text(old[rel])
    diff = run(["git", "diff", b, a, "--", prefix + "src/main", prefix + "pom.xml"], cwd=scratch).stdout
    homes = json.load(open(H / "tools/java/java_homes.json"))
    sys.path.insert(0, str(H / "PyTaRGET"))
    from data_processing.java_plausibility import detect_java_versions, pick_java_home
    pom = scratch / "pom.xml"
    jv, _ = pick_java_home(homes, detect_java_versions(pom.read_text(errors="ignore")) if pom.exists() else [])
    ws = task / "ws"; shutil.copytree(scratch, ws, ignore=shutil.ignore_patterns(".git", "target")); shutil.rmtree(scratch)
    ref_in_ws = {prefix + k: v for k, v in ref.items()}; old_in_ws = {prefix + k: v for k, v in old.items()}
    cfg = finalize(task, ws, ref_in_ws, old_in_ws, diff, module or ".", jv, prefix + "src/test/java")
    meta = {"id": task_id, "kind": "real", "repo": repo, "aCommit": a, "bCommit": b, "module": module, "java": jv, "changed_test_files": len(ref), "diff_chars": len(diff)}
    (task / "meta.json").write_text(json.dumps(meta, indent=1)); print(meta)


if __name__ == "__main__":
    if sys.argv[1] == "synth": synth(sys.argv[2], *(sys.argv[3:4]))
    else: real(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else "")
