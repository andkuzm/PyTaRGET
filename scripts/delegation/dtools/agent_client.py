"""Client side of the agent delegate (alias with "agent": true): hands the run workspace to scripts/agent/delegate_worker.py (a Slurm job without network) through the queue folder,
waits, and copies back ONLY the changed files under the test roots (and removes the test files the agent deleted)."""
import json
import shutil
import time
import uuid
from pathlib import Path


def run_agent(ws, cfg, alias, timeout=2100):
    d = cfg["delegates"][alias]
    queue = Path(d["queue"]).expanduser()
    alive = queue / "alive"
    if not alive.exists() or time.time() - float(alive.read_text().strip() or 0) > 90:
        raise RuntimeError("the agent server job is not running (no recent heartbeat)")
    rid = f"{Path(ws).parent.name}_{uuid.uuid4().hex[:6]}"
    req = {"ws": str(Path(ws).resolve()), "java_home": cfg["java_home"], "maven_home": cfg["maven_home"], "test_roots": cfg.get("test_roots", ["src/test/java"]), "mvn_extra": cfg.get("mvn_extra", [])}
    tmp = queue / "requests" / f"{rid}.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(queue / "requests" / f"{rid}.json")
    done = queue / "done" / f"{rid}.json"
    t0 = time.time()
    while not done.exists():
        if time.time() - t0 > timeout:
            raise RuntimeError("no answer from the agent server in time")
        if time.time() - float(alive.read_text().strip() or 0) > 120:
            raise RuntimeError("the agent server job stopped while working")
        time.sleep(3)
    r = json.loads(done.read_text())
    if r.get("status") == "worker_error":
        raise RuntimeError("agent worker error: " + r.get("detail", ""))
    roots = [(Path(ws).resolve() / x).resolve() for x in req["test_roots"]]
    for rel in r.get("deleted", []):
        dst = (Path(ws).resolve() / rel).resolve()
        if rel.endswith(".java") and any(root in dst.parents for root in roots) and dst.is_file():
            dst.unlink()
    for rel in r["changed"]:
        src, dst = Path(r["ws"]) / rel, (Path(ws).resolve() / rel).resolve()
        if src.is_symlink() or not any(root == dst or root in dst.parents for root in roots):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return r
