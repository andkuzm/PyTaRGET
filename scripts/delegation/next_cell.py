"""Print the next pending pilot cell of ~/delegation_exp/plan_queue.json and the launcher command for it (or DONE)."""
import json
from pathlib import Path
EXP = Path.home() / "delegation_exp"
q = json.loads((EXP / "plan_queue.json").read_text()); st = json.loads((EXP / "state.json").read_text()).get("runs", {})
done = {r.get("cell") for r in st.values() if r.get("status") == "scored"}
for c in q["cells"]:
    if c["id"] in done: continue
    cmd = f"python scripts/delegation/launcher.py new-run {c['task']} {c['arm']} {c['rep']}" + (f" --strategy {c['strategy']} --context {c['context']} --verify {c['verify']}" if c["arm"] == "delegate" else "") + f" --cell {c['id']}"
    print(json.dumps({"cell": c["id"], "command": cmd})); break
else:
    print("DONE")
