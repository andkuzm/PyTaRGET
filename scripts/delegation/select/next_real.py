"""Next batch of the real-task runs: reads ~/delegation_exp/real_prep.json (verified tasks) and batch_results.json and prints the next (task, rep) whose three arms are not all scored/created,
in the order: repetition 1 of every verified task, then repetition 2. usage: next_real.py -> JSON {"task":..., "rep":..., "command":...} or DONE"""
import json
from pathlib import Path

E = Path.home() / "delegation_exp"
prep = json.loads((E / "real_prep.json").read_text()) if (E / "real_prep.json").exists() else {}
res = json.loads((E / "batch_results.json").read_text()) if (E / "batch_results.json").exists() else {}
runs = {p.name for p in (E / "runs").iterdir()} if (E / "runs").exists() else set()
tasks = [t for t, s in prep.items() if s.get("ok")]
for rep in (1, 2):
    for t in tasks:
        names = [f"{t}__solo__r{rep}", f"{t}__deleg-S4-auto-review__r{rep}", f"{t}__deleg-S1-auto-review__r{rep}"]
        if all(n in res or n in runs for n in names): continue
        print(json.dumps({"task": t, "rep": rep, "command": f"python scripts/delegation/make_batch.py {t} {rep}"})); raise SystemExit
print("DONE")
