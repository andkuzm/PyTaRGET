"""Build and verify real delegation tasks from TaRBench commit pairs, one after the other; results in ~/delegation_exp/real_prep.json (status per task). usage: prep_real_batch.py [task id ...] (default: all candidates)"""
import json
import subprocess
import sys
import time
from pathlib import Path

H = Path.home(); D = H / "PyTaRGET/scripts/delegation"; PY = str(H / "venv3.12/bin/python")
CANDS = {  # task id: (repo, aCommit, bCommit, module)
    "real_logicng_a": ("logic-ng/LogicNG", "803b406b", "dfa28508", "."),
    "real_logicng_b": ("logic-ng/LogicNG", "8d771456", "5cd053c0", "."),
    "real_ta4j": ("mdeverdelhan/ta4j-origins", "47b09ac", "c22a705", "ta4j"),
    "real_cactoos": ("yegor256/cactoos", "a23b69d8", "7a7e2215", "."),
    "real_tablesaw": ("jtablesaw/tablesaw", "05fe70d1", "de5860eb", "core"),
    "real_mockserver": ("mock-server/mockserver", "e4bf7451d", "78015e3af", "mockserver-core"),
    "real_cucumber": ("cucumber/cucumber-jvm", "ce19e3be6", "556fb3d75", "core"),
    "real_sonarcxx": ("SonarOpenCommunity/sonar-cxx", "57a61561", "756119a4", "cxx-checks"),
}
status_f = H / "delegation_exp/real_prep.json"; status = json.loads(status_f.read_text()) if status_f.exists() else {}
for tid in (sys.argv[1:] or list(CANDS)):
    repo, a, b, mod = CANDS[tid]; t0 = time.time()
    p = subprocess.run([PY, str(D / "prepare_task.py"), "real", tid, repo, a, b, mod], capture_output=True, text=True, timeout=3600)
    if p.returncode != 0: status[tid] = {"stage": "prepare", "ok": False, "err": (p.stderr or p.stdout)[-300:]}
    else:
        v = subprocess.run([PY, str(D / "verify_task.py"), tid], capture_output=True, text=True, timeout=5400); out = v.stdout
        ref_ok = "| passes" in out.split("\n")[0]; task_fails = "TASK state" in out and ("COMPILE ERRORS" in out or "failed" in out)
        status[tid] = {"stage": "verify", "ok": bool(ref_ok and task_fails), "reference_passes": ref_ok, "head": out[:400], "minutes": round((time.time() - t0) / 60, 1)}
    status_f.write_text(json.dumps(status, indent=1)); print(tid, status[tid]["ok"], flush=True)
