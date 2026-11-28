"""Prepare a NOT-YET-USED real project as a delegation task whose builds and tests run ONLY in the sandbox (dtools/jail.py).
usage: new_task.py <task id> <owner/repo> <aCommit (after)> <bCommit (parent)> [<module dir>] [--java N]
Steps: prepare_task.real (source clone; no project code runs) -> own Maven cache -> fill it in the sandbox WITH network but WITHOUT running tests (-Dtest=<no such test>) ->
offline, no network: (1) old code + old tests: which tests fail from the environment alone, (2) new code + reference tests, (3) new code + old tests must be broken.
Writes check.json in the task directory; ws/.delegation/env_failing.json is copied into every run (both arms see the same note)."""
import json
import re
import shutil
import sys
from pathlib import Path

H = Path.home(); HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(H / "PyTaRGET"), str(HERE.parent / "agent")]
import baseline  # noqa: E402
import prepare_task  # noqa: E402
from data_processing.java_plausibility import MVN_SKIPS  # noqa: E402
from dtools import common  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--java"]
java = None
if "--java" in sys.argv:
    java = sys.argv[sys.argv.index("--java") + 1]; args.remove(java)
tid, repo, a, b = args[:4]; module = args[4] if len(args) > 4 else ""
task = prepare_task.EXP / "tasks" / tid
prepare_task.real(tid, repo, a, b, module)
ws = task / "ws"; cfg = common.load_config(ws)
if java: cfg["java_home"] = json.load(open(H / "tools/java/java_homes.json"))[java]
cfg.update({"jail": True, "m2": str(task / "m2"), "mvn_extra": [*cfg.get("mvn_extra", []), *MVN_SKIPS, "-Dsurefire.printSummary=false", "-Dtoolchain.skip=true", "-Dgitbuildhook.skip=true"]})
for k in ("key_file", "delegates", "default_delegate"): cfg.pop(k, None)
cfg["delegates"] = {}
(ws / ".delegation/config.json").write_text(json.dumps(cfg, indent=1))


def variant(name, old_main=False, ref_tests=False):
    d = task / "_check" / name / "ws"; shutil.rmtree(d.parent, ignore_errors=True); shutil.copytree(ws, d, ignore=shutil.ignore_patterns(".git", "target"))
    if old_main:
        ok, err = baseline.reverse_changes(d); shutil.rmtree(d / ".git", ignore_errors=True)
        if not ok: sys.exit("cannot rebuild the old code: " + err)
    if ref_tests:
        root = cfg["test_roots"][0]
        for rel in [str(p.relative_to(task / "old_tests")) for p in (task / "old_tests").rglob("*.java")]:
            (d / rel).unlink(missing_ok=True)
        for p in (task / "ref").rglob("*.java"):
            if p.stat().st_size:
                q = d / p.relative_to(task / "ref"); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, q)
    return d


def fetch_artifact(coord):
    """download ONE artifact (and its dependencies) into the project's cache, in the sandbox with network; no project code runs."""
    from dtools import jail
    d = task / "_check" / "fetch" / "ws"; d.mkdir(parents=True, exist_ok=True)
    return jail.run(d, cfg, [str(Path(cfg["maven_home"]) / "bin/mvn"), "--batch-mode", f"-Dmaven.repo.local={cfg['m2']}", "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get", f"-Dartifact={coord}"], network=True, timeout=600).returncode


def go(d, offline=True, tests=None, timeout=900):
    seen = set()
    while True:                                      # the test provider is only asked for when tests run: fetch what the offline run names as missing, then run again
        rc, log = common.mvn(d, tests, offline=offline, timeout=timeout)
        need = [c for c in re.findall(r"artifact ([\w.-]+:[\w.-]+:(?:jar|pom):[\w.-]+) has not been downloaded", log) if c not in seen]
        if not (offline and rc and need) or len(seen) > 25: break
        seen.add(need[0]); g, art, kind_, ver = need[0].split(":"); print("fetching missing library", art, ver, "->", fetch_artifact(f"{g}:{art}:{ver}"), flush=True)
    kind, data = common.result_of(d, rc, log)
    return rc, log, kind, data


report = {"java_home": Path(cfg["java_home"]).name[:20]}
old, new_ref, new_old = variant("old", old_main=True), variant("ref", ref_tests=True), variant("task")
for name, d in (("fill_old", old), ("fill_ref", new_ref)):
    rc, log, *_ = go(d, offline=False, tests=["ZzNoSuchTestZz"])
    report[name] = {"rc": rc, "errors": [l[:220] for l in log.splitlines() if l.startswith("[ERROR]")][:4] if rc else []}
    if rc: print(json.dumps(report, indent=1)); sys.exit("download step failed")


def summary(kind, data):
    if kind == "tests": return {"run": len(data), "failed": len(baseline.failing_ids(data))}
    if kind == "compile_error": return {"compile_errors": sum(map(len, data.values())), "files": len(data)}
    return {"no_result": str(data)[-600:]}


rc, log, kind, data = go(old); report["old_code_old_tests"] = summary(kind, data)
if kind != "tests": print(json.dumps(report, indent=1)); sys.exit("the old code does not run offline")
env = baseline.failing_ids(data); report["env_failing"] = env
rc, log, kind, data = go(new_ref); report["new_code_ref_tests"] = summary(kind, data)
if kind != "tests": print(json.dumps(report, indent=1)); sys.exit("the reference tests do not run offline")
ref_extra = [x for x in baseline.failing_ids(data) if x not in env]; report["ref_failing_not_in_old"] = ref_extra
rc, log, kind, data = go(new_old); report["new_code_old_tests"] = summary(kind, data)
broken = kind == "compile_error" or (kind == "tests" and [x for x in baseline.failing_ids(data) if x not in env])
report["usable"] = bool(broken) and len(env) <= 15 and len(ref_extra) <= 5
(ws / ".delegation/env_failing.json").write_text(json.dumps(env))
(task / "env_excluded.json").write_text(json.dumps({"old_code": env, "ref_only": ref_extra}))
(task / "check.json").write_text(json.dumps(report, indent=1))
shutil.rmtree(task / "_check", ignore_errors=True)
print(json.dumps({**report, "env_failing": env[:8], "n_env": len(env), "ref_failing_not_in_old": ref_extra[:5]}, indent=1))
