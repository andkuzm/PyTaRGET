"""Agent delegate worker. Runs INSIDE the agent_server Slurm job, i.e. in a network namespace without internet, next to the vLLM server (slurm/agent_server.sbatch).
tools/delegate (alias with "agent": true) puts a request file into <queue>/requests/<id>.json {"ws","java_home","maven_home","test_roots","mvn_extra"}; for each request:
  1. a clean copy of the run workspace is made under <queue>/work/<id>/ws (without tools/, .delegation/, build output and git history) with ONE snapshot commit;
  2. mini-SWE-agent runs on it inside the mount sandbox (scripts/agent/sandbox_inner.sh: no home, no datasets, no tokens, no reference repair; a private copy of the Maven cache);
  3. <queue>/done/<id>.json reports usage and the changed files. Only changed or new files under the test roots, and deleted .java test files, are offered back; any other change is listed as ignored."""
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_processing import agent_workspace as aw  # noqa: E402
from data_processing.java_plausibility import MVN_SKIPS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_paths  # noqa: E402
import baseline  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delegation"))
from dtools import common  # noqa: E402

HERE = Path(__file__).resolve().parent
GIT = ["git", "-c", "user.name=a", "-c", "user.email=a@example.invalid"]
TASK = """The production code of this Java (Maven) project was changed and the test suite no longer compiles or passes. SOURCE_CHANGES.diff in the project root shows the change.
Task: repair the tests so that the whole test suite compiles and passes with the current production code.
Run the tests (offline Maven is already configured) with:  {command}
Rules:
1. Change ONLY files under {roots}. Never change src/main, pom.xml or any other file: the production code is correct, only the tests are outdated. Changes outside the test folders are discarded.
2. Do not weaken the tests: do not loosen assertions, do not comment tests out, do not add @Disabled/@Ignore, assumptions or try/catch blocks that swallow failures.
   When behaviour changed, update the expected values to the new behaviour; when an API changed, adapt the calls.
   Delete a test, or a single assertion, ONLY when what it checks no longer exists in the source code at all (the tested class, method or behaviour was removed); if any part of what it checks still exists, keep checking that part. A check that valid, compiling code can no longer trigger (for example because a parameter type now rules the bad input out, even if the throwing line is still in the source) also counts as gone; never use casts or raw types to force it. When a check cannot be kept, delete it: never replace it with one that cannot fail or that only tests the test's own code (for example a Java type test, or a cast in the test that throws by itself).
3. Do not use git history, the network or package installation; everything you need is in this directory. Work only inside it.
4. This environment has no network and some folders are not writable. Never rewrite a test to get around that (for example by replacing a remote server with a local fake, or by changing which paths the test uses): a test that fails only because of this environment stays as it is and keeps failing; say so when you submit.
When the whole test suite passes (apart from tests that fail only because of this environment), submit."""
ENV_NOTE = """

These tests ALREADY FAIL in this environment on the OLD code, before the source change, so their failure is not caused by the change:
{tests}
Do not try to make them pass. Edit them only as far as the source change requires (for example so that they compile); never change what they do (addresses, paths, set-up) and do not delete or disable them because they fail. They may keep failing: submit when every OTHER test passes."""


def sh(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


@functools.lru_cache(maxsize=None)
def _software(java_home, maven_home):
    return tuple(runtime_paths.needed(java_home, maven_home, str(Path.home() / "venv_agent")))


def software_for(req):
    """only the folders the JDK, Maven and the agent's Python need are visible (the whole software trees made `find /` hang)."""
    return list(_software(req["java_home"], req["maven_home"]))


def usage_of(traj):
    tin = tout = 0
    for m in traj.get("messages", []):
        u = ((m.get("extra") or {}).get("response") or {}).get("usage") if isinstance(m.get("extra"), dict) else None
        if u:
            tin += u.get("prompt_tokens", 0) or 0
            tout += u.get("completion_tokens", 0) or 0
    return tin, tout


def env_baseline(src, base, req, env, m2, timeout=900):
    """(sorted ids of the tests failing on the OLD code inside the sandbox, status text). Runs before the agent, in its own throw-away copy."""
    ws, out = base / "baseline_ws", base / "baseline_out"
    out.mkdir(parents=True)
    shutil.copytree(src, ws, symlinks=True, ignore=shutil.ignore_patterns(".git", ".delegation", "tools", "target"))
    ok, error = baseline.reverse_changes(ws)
    if not ok:
        return [], f"unavailable: the source change could not be reversed ({error})"
    cmd = ["unshare", "-m", str(HERE / "sandbox_inner.sh"), str(ws), str(out), str(m2), *software_for(req), "--",
           f"{req['maven_home']}/bin/mvn", "test", "-Dmaven.test.failure.ignore=true", *req.get("mvn_extra", [])]
    with open(out / "mvn.log", "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            return [], f"unavailable: the old tests did not finish in {timeout} s"
    results = common.parse_surefire(ws)
    if proc.returncode or not results:
        errors = [l for l in (out / "mvn.log").read_text(errors="ignore").splitlines() if l.startswith("[ERROR]")]
        return [], f"unavailable: the old code did not build and run cleanly (exit {proc.returncode}; {' '.join(errors[:2])[:300]})"
    return baseline.failing_ids(results), f"ok: {len(results)} tests run on the old code"


def handle(req_file, queue, m2, step_limit, timeout):
    rid = req_file.stem
    req = json.loads(req_file.read_text())
    base = queue / "work" / rid
    shutil.rmtree(base, ignore_errors=True)
    ws, out = base / "ws", base / "out"
    out.mkdir(parents=True)
    src = Path(req["ws"])
    shutil.copytree(src, ws, symlinks=True, ignore=shutil.ignore_patterns(".git", ".delegation", "tools", "target"))
    sh(["git", "init", "-q"], ws)
    (ws / ".git" / "info" / "exclude").write_text("target/\n")
    sh(["git", "add", "-A"], ws)
    sh([*GIT, "commit", "-q", "-m", "current state"], ws)
    roots = req.get("test_roots") or ["src/test/java"]
    command = " ".join(["mvn", "test", "-q", *req.get("mvn_extra", [])])
    shutil.copy(HERE / "mini_inkling.yaml", out / "mini_local.yaml")
    home = str(Path.home())
    maven_args = " ".join(["-o", "-nsu", "--batch-mode", "-Dsurefire.failIfNoSpecifiedTests=false", "-DfailIfNoTests=false", *MVN_SKIPS, f"-Dmaven.repo.local={m2}"])
    env = {"HOME": home, "PATH": f"{req['java_home']}/bin:{req['maven_home']}/bin:{home}/venv_agent/bin:/usr/bin:/bin", "JAVA_HOME": req["java_home"],
           "MAVEN_ARGS": maven_args, "MAVEN_OPTS": "-Xmx2g", "LANG": "C.UTF-8", "TERM": "dumb", "MSWEA_SILENT_STARTUP": "1", "MSWEA_CONFIGURED": "1",
           "MSWEA_GLOBAL_CONFIG_DIR": str(out / "home" / "cfg")}
    env["SANDBOX_NARROW"] = "1"
    try:
        env_failing, env_status = env_baseline(src, base, req, env, m2)
    except Exception as error:
        env_failing, env_status = [], f"unavailable: {error!r}"[:300]
    print(rid, "baseline", env_status, "| failing before the change:", len(env_failing), flush=True)
    task = TASK.format(command=command, roots=", ".join(roots)) + (ENV_NOTE.format(tests="\n".join("  - " + x for x in env_failing[:30])) if env_failing else "")
    cmd = ["unshare", "-m", str(HERE / "sandbox_inner.sh"), str(ws), str(out), str(m2), *software_for(req), "--",
           f"{home}/venv_agent/bin/mini", "-c", "mini.yaml", "-c", str(out / "mini_local.yaml"), "-c", f"agent.step_limit={step_limit}", "-c", "environment.timeout=600",
           "-m", "hosted_vllm/inkling", "--agent-class", "default", "-y", "-l", "0", "-t", task, "-o", str(out / "traj.json")]
    started = time.time()
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=open(out / "stdout.txt", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        status = "finished"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        status = "timeout"
    result = {"status": status, "seconds": round(time.time() - started), "returncode": proc.returncode, "changed": [], "deleted": [], "ignored": [],
              "env_failing": env_failing, "baseline": env_status}
    try:
        traj = json.load(open(out / "traj.json"))
        cmds = aw.commands_from_trajectory(traj)
        result.update({"api_calls": traj.get("info", {}).get("model_stats", {}).get("api_calls"), "exit_status": traj.get("info", {}).get("exit_status"),
                       "audit": aw.audit_trajectory(cmds), "n_commands": len(cmds)})
        result["tokens_in"], result["tokens_out"] = usage_of(traj)
    except Exception as error:
        result["trajectory_error"] = repr(error)[:200]
    sh(["git", "add", "-A"], ws)
    for line in sh(["git", "diff", "--cached", "--name-status", "--no-renames", "HEAD"], ws).stdout.splitlines():
        kind, rel = line.split("\t", 1)
        in_tests = any(rel.startswith(r.rstrip("/") + "/") for r in roots) and not (ws / rel).is_symlink()
        if in_tests and kind in ("A", "M"): result["changed"].append(rel)
        elif in_tests and kind == "D" and rel.endswith(".java"): result["deleted"].append(rel)   # allowed when the tested code is gone; tools/review reports every method of the file
        else: result["ignored"].append(f"{kind} {rel}")
    result["ws"] = str(ws)
    tmp = queue / "done" / f"{rid}.tmp"
    tmp.write_text(json.dumps(result))
    tmp.rename(queue / "done" / f"{rid}.json")
    print(rid, status, result.get("exit_status"), "calls", result.get("api_calls"), "changed", len(result["changed"]), "ignored", len(result["ignored"]), flush=True)


def main():
    queue, m2 = Path(sys.argv[1]), Path(sys.argv[2])
    step_limit, timeout = int(os.environ.get("STEP_LIMIT", 150)), int(os.environ.get("REQUEST_TIMEOUT", 1800))
    for d in ("requests", "done", "work"):
        (queue / d).mkdir(parents=True, exist_ok=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    print("worker ready", flush=True)
    while not stop:
        (queue / "heartbeat").write_text(str(time.time()))
        todo = sorted(f for f in (queue / "requests").glob("*.json") if not (queue / "done" / f.name).exists())
        if todo:
            try:
                handle(todo[0], queue, m2, step_limit, timeout)
            except Exception as error:
                (queue / "done" / todo[0].name).write_text(json.dumps({"status": "worker_error", "detail": repr(error)[:500], "changed": [], "ignored": []}))
                print(todo[0].stem, "worker error", repr(error)[:300], flush=True)
        else:
            time.sleep(3)


if __name__ == "__main__":
    main()
