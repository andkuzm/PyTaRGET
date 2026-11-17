"""Events for the training-size plausibility scoring: a line per model group at every 25% of its cases (all four fractions), when a set
is complete, and a STALL warning if nothing grew for 20 minutes while a scoring process is alive. Exits when all sets are complete.
  python -u trainfrac_watch.py <state file> [model ...]"""
import json, subprocess, sys, time
from pathlib import Path
LAY = Path.home() / "eval_trainfrac_plaus"
TOTAL = {"codet5p": 1293, "plbart": 1342, "codegen": 1585}
state_file = Path(sys.argv[1]); models = sys.argv[2:] or ["codegen", "plbart"]
state = json.loads(state_file.read_text()) if state_file.exists() else {}
def done(name):
    p = LAY / name / "0.8" / "plausibility_v3.json"
    try: return len(json.load(open(p))["results"])
    except Exception: return 0
def alive():
    return subprocess.run(["pgrep", "-f", "plausibility_test --results"], capture_output=True).returncode == 0
last_total, last_change = None, time.time()
while True:
    grand = 0
    for m in models:
        counts = {f: done(f"{m}_f{f}") for f in (20, 40, 60, 80)}
        total = TOTAL[m] * 4; got = sum(counts.values()); grand += got
        quarter = (got * 4) // total
        if quarter > state.get(m, {}).get("quarter", 0):
            state.setdefault(m, {})["quarter"] = quarter
            print(f"PROGRESS {m}: {got}/{total} cases scored over four fractions ({100*got//total}%), per set {counts}", flush=True)
        for f, c in counts.items():
            key = f"{m}_f{f}_complete"
            if c >= TOTAL[m] and not state.get(key):
                state[key] = True
                print(f"SET COMPLETE: {m}_f{f} ({c}/{TOTAL[m]})", flush=True)
    if grand != last_total:
        last_total, last_change = grand, time.time()
        state["stalled"] = False
    elif alive() and time.time() - last_change > 1200 and not state.get("stalled"):
        state["stalled"] = True
        print(f"STALL: no new scored case for 20 min while a scoring process is alive (total {grand})", flush=True)
    state_file.write_text(json.dumps(state))
    if all(state.get(f"{m}_f{f}_complete") for m in models for f in (20, 40, 60, 80)):
        print("ALL SETS COMPLETE", flush=True)
        break
    time.sleep(120)
