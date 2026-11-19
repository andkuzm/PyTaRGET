"""Reset the Java agent workspaces to their snapshot commit (tracked files restored, untracked scratch removed; ignored build output stays): the start state of a rerun.
usage: python reset_workspaces_java.py <root>"""
import json, subprocess, sys
from pathlib import Path
m = json.load(open(Path(sys.argv[1]) / "manifest.json"))
n = dirty = 0
for e in m.values():
    if e["status"] != "ok":
        continue
    subprocess.run(["git", "checkout", "--quiet", "--", "."], cwd=e["ws"])
    subprocess.run(["git", "clean", "-fdq"], cwd=e["ws"])
    n += 1
    dirty += bool(subprocess.run(["git", "status", "--porcelain"], cwd=e["ws"], capture_output=True, text=True).stdout.strip())
print(f"reset {n} workspaces, still dirty: {dirty}", flush=True)
