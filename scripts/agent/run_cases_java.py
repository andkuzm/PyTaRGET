"""
Run mini-SWE-agent on the prepared Java cases, OFFLINE inside the job's private network namespace, each case in its own mount namespace that shows only its
workspace, its output directory, the agent environment, the cluster software (JDKs, Maven, Python) and the shared Maven repository (scripts/agent/sandbox_inner.sh).
Maven runs offline through MAVEN_ARGS. Afterwards (outside the sandbox) the final test method, touched files, the command audit and the usage are saved. Resumable.
  python run_cases_java.py --split-dir <dir> --root ~/agent_java --out ~/eval_agent/java_qwen3coder --workers 6 [--limit N]
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data_processing import agent_workspace as aw
from data_processing import agent_workspace_java as aj
from data_processing.java_plausibility import MVN_SKIPS

HERE = Path(__file__).resolve().parent


def run_case(i, row, entry, root, out_root, timeout, model, mini=None):
    out = out_root / str(i)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "mini_local.yaml", out / "mini_local.yaml")
    ws, home, m2 = Path(entry["ws"]), str(Path.home()), str(root / "m2")
    homes = json.load(open(Path.home() / "tools/java/java_homes.json"))
    maven_args = " ".join(["-o", "-nsu", "--batch-mode", "-Dsurefire.failIfNoSpecifiedTests=false", "-DfailIfNoTests=false", "-Dmaven.test.skip=false",
                           "-DskipTests=false", *MVN_SKIPS, f"-Dmaven.repo.local={m2}"])
    env = {"HOME": home, "PATH": f"{entry['java_home']}/bin:{homes['maven']}/bin:{home}/venv_agent/bin:/usr/bin:/bin", "JAVA_HOME": entry["java_home"],
           "MAVEN_ARGS": maven_args, "MAVEN_OPTS": "-Xmx2g", "LANG": "C.UTF-8", "TERM": "dumb", "MSWEA_SILENT_STARTUP": "1", "MSWEA_CONFIGURED": "1",
           "MSWEA_GLOBAL_CONFIG_DIR": str(out / "home" / "cfg")}
    cmd = ["unshare", "-m", str(HERE / "sandbox_inner.sh"), str(ws), str(out), m2, "/gpfs/helios/software", "/gpfs/space/software", "--",
           mini or f"{home}/venv_agent/bin/mini", "-c", "mini.yaml", "-c", str(out / "mini_local.yaml"), "-m", model, "--agent-class", "default", "-y", "-l", "0",
           "-t", aj.task_prompt(row, entry["command"]), "-o", str(out / "traj.json")]
    started = time.time()
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=open(out / "stdout.txt", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        status = "finished"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        status = "timeout"
    result = {"status": status, "seconds": round(time.time() - started), "returncode": proc.returncode}
    try:
        result.update(aj.collect_result(row, ws, entry.get("method_lines")))
    except Exception as error:
        result["collect_error"] = repr(error)[:200]
    try:
        traj = json.load(open(out / "traj.json"))
        commands = aw.commands_from_trajectory(traj)
        result["audit"] = aw.audit_trajectory(commands)
        result["api_calls"] = traj.get("info", {}).get("model_stats", {}).get("api_calls")
        result["exit_status"] = traj.get("info", {}).get("exit_status")
        result["n_commands"] = len(commands)
    except Exception as error:
        result["trajectory_error"] = repr(error)[:200]
    (out / "result.json").write_text(json.dumps(result))
    print(i, result["status"], result.get("exit_status"), "calls", result.get("api_calls"), "audit", len(result.get("audit", [])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per case")
    ap.add_argument("--model", default="hosted_vllm/qwen3-coder")
    args = ap.parse_args()
    rows = json.load(open(Path(args.split_dir) / "splits" / "test.json"))
    root, out_root = Path(args.root), Path(args.out)
    manifest = json.load(open(root / "manifest.json"))
    todo = [(int(i), e) for i, e in sorted(manifest.items(), key=lambda kv: int(kv[0])) if e["status"] == "ok" and not (out_root / i / "result.json").exists()][:args.limit]
    print(len(todo), "cases to run", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda item: run_case(item[0], rows[item[0]], item[1], root, out_root, args.timeout, args.model), todo))


if __name__ == "__main__":
    main()
