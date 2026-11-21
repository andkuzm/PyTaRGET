"""Create the three runs (solo, S4+review, S1+review) of one repo and repetition and print the agent pointer prompts as JSON lines.
usage: make_batch.py <task> <rep> [arms=solo,S4,S1]"""
import json
import subprocess
import sys
from pathlib import Path

H = Path.home(); LAUNCH = [str(H / "venv3.12/bin/python"), str(H / "PyTaRGET/scripts/delegation/launcher.py")]
task, rep = sys.argv[1], sys.argv[2]; arms = (sys.argv[3] if len(sys.argv) > 3 else "solo,S4,S1").split(",")
hand = H / "delegation_exp/synth" / task / "handoff.txt"
if task.startswith("ledger"): hand = H / "delegation_exp/ledger_handoff.txt"
for arm in arms:
    cmd = LAUNCH + ["new-run", task, "solo" if arm == "solo" else "delegate", rep, "--cell", f"R-{task}-{arm}-{rep}"]
    if arm != "solo": cmd += ["--strategy", arm, "--context", "auto", "--verify", "review"]
    if hand.exists(): cmd += ["--handoff", str(hand)]
    rid = subprocess.run(cmd, capture_output=True, text=True).stdout.split("\n")[0].strip()
    print(json.dumps({"arm": arm, "run": rid}))
