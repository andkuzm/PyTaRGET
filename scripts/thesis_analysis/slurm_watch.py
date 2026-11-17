"""Event stream for Slurm jobs: one line whenever a watched job changes state (PENDING -> RUNNING -> COMPLETED/FAILED/...), and
'ALL DONE' when every job is in a final state. For use as a Claude Code Monitor source (stdout lines are events).
  python -u slurm_watch.py <state file> <jobid> [<jobid> ...]
The state file remembers what was already reported, so re-arming the watch does not repeat old events. A final state other than
COMPLETED is tagged ATTENTION."""
import json, subprocess, sys, time
from pathlib import Path
FINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED")
state_file, ids = Path(sys.argv[1]), sys.argv[2:]
seen = json.loads(state_file.read_text()) if state_file.exists() else {}
while True:
    out = subprocess.run(["sacct", "-j", ",".join(ids), "-X", "-n", "-P", "--format=JobID,JobName,State,Elapsed,ExitCode"],
                         capture_output=True, text=True).stdout
    current = {}
    for line in out.strip().splitlines():
        jid, name, st, elapsed, code = (line.split("|") + [""] * 5)[:5]
        current[jid] = (name, st.split()[0], elapsed, code)
    for jid, (name, st, elapsed, code) in current.items():
        if seen.get(jid) != st:
            seen[jid] = st
            tag = " ATTENTION" if st in FINAL and st != "COMPLETED" else ""
            print(f"{name} ({jid}): {st}, elapsed {elapsed}, exit {code}{tag}", flush=True)
    state_file.write_text(json.dumps(seen))
    if current and len(current) == len(ids) and all(v[1] in FINAL for v in current.values()):
        print("ALL DONE", flush=True)
        break
    time.sleep(60)
