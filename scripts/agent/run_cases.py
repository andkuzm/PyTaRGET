"""
Run mini-SWE-agent on the prepared cases (OFFLINE, inside the job's private network namespace; see slurm/run_agent_qwen.sbatch). Per case the agent runs in
a mount namespace that shows only its workspace, its output directory, its virtualenv and the agent environment (scripts/agent/sandbox_inner.sh): no home
directory, no datasets, no tokens, no network. Afterwards (outside the sandbox) the final test function, the touched files, the audit of the commands and
the usage are saved. Resumable: a case with result.json is skipped.
  python run_cases.py --split-dir <dir> --root ~/agent_work --out ~/eval_agent/qwen3coder --workers 6 [--limit N]
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

HERE = Path(__file__).resolve().parent


def run_case(i, row, entry, out_root, timeout, model, mini=None, step_limit=40, config="mini_local.yaml"):
    out = out_root / str(i)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / config, out / "mini_local.yaml")
    ws, venv = Path(entry["ws"]), str(Path(entry["python"]).parent.parent)
    real = os.path.realpath(entry["python"])                                   # the interpreter the case venv was made from (cluster software)
    pybase = "/gpfs/space/software" if real.startswith("/gpfs/space/software/") else str(Path(real).parent.parent)   # spack pythons need sibling libraries
    home = str(Path.home())
    env = {"HOME": home, "PATH": f"{venv}/bin:{home}/venv_agent/bin:/usr/bin:/bin", "PYTHONPATH": f"{ws}:{ws}/src", "LANG": "C.UTF-8", "TERM": "dumb",
           "MSWEA_SILENT_STARTUP": "1", "MSWEA_CONFIGURED": "1", "MSWEA_GLOBAL_CONFIG_DIR": str(out / "home" / "cfg"), "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = ["unshare", "-m", str(HERE / "sandbox_inner.sh"), str(ws), str(out), venv, pybase, "--",
           mini or f"{home}/venv_agent/bin/mini", "-c", "mini.yaml", "-c", str(out / "mini_local.yaml"), "-c", f"agent.step_limit={step_limit}", "-m", model, "--agent-class", "default", "-y", "-l", "0",
           "-t", aw.task_prompt(row), "-o", str(out / "traj.json")]
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
        result.update(aw.collect_result(row, ws))
    except Exception as error:
        result["collect_error"] = repr(error)[:200]
    try:
        traj = json.load(open(out / "traj.json"))
        result["audit"] = aw.audit_trajectory(aw.commands_from_trajectory(traj))
        result["api_calls"] = traj.get("info", {}).get("model_stats", {}).get("api_calls")
        result["exit_status"] = traj.get("info", {}).get("exit_status")
        result["n_commands"] = len(aw.commands_from_trajectory(traj))
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
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per case")
    ap.add_argument("--model", default="hosted_vllm/qwen3-coder")
    ap.add_argument("--step-limit", type=int, default=40, help="model calls per case")
    ap.add_argument("--config", default="mini_local.yaml", help="override file in scripts/agent (mini_inkling.yaml for Inkling)")
    args = ap.parse_args()
    rows = json.load(open(Path(args.split_dir) / "splits" / "test.json"))
    manifest = json.load(open(Path(args.root) / "manifest.json"))
    out_root = Path(args.out)
    todo = [(int(i), e) for i, e in sorted(manifest.items(), key=lambda kv: int(kv[0])) if e["status"] == "ok" and not (out_root / i / "result.json").exists()]
    todo = todo[:args.limit]
    print(len(todo), "cases to run", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda item: run_case(item[0], rows[item[0]], item[1], out_root, args.timeout, args.model, step_limit=args.step_limit, config=args.config), todo))


if __name__ == "__main__":
    main()
