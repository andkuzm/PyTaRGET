"""Check a task: (1) workspace + REFERENCE tests (the repaired state) must compile and pass - run ONLINE, which also fills the shared offline Maven repository; (2) the task workspace (old tests) must
fail (compile errors or failing tests), run offline. Prints both compact results. usage: verify_task.py <task id>"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtools import common  # noqa: E402

task = Path.home() / "delegation_exp/tasks" / sys.argv[1]
tmp = Path(tempfile.mkdtemp(prefix="verify_", dir=Path.home() / "delegation_exp/scratch"))
ws = tmp / "ws"; shutil.copytree(task / "ws", ws, symlinks=True)
for f in (task / "ref").rglob("*.java"):
    dst = ws / f.relative_to(task / "ref"); dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(f.read_text())
rc, log = common.mvn(ws, offline=False, timeout=3000); kind, data = common.result_of(ws, rc, log)
print("REFERENCE state (online):", common.compact(ws, kind, data, rc, 6).split("\n")[0], "| passes" if kind == "tests" and all(r[2] in ("pass", "skipped") for r in data) else "| PROBLEM")
bad = [r for r in data if r[2] in ("fail", "error")] if kind == "tests" else []
for c, m, s, msg in bad[:8]: print("   still failing in the reference state:", c.split(".")[-1], m, msg[:100])
shutil.rmtree(tmp)
ws2 = tmp.parent / ("verify_task_" + sys.argv[1]); shutil.rmtree(ws2, ignore_errors=True); shutil.copytree(task / "ws", ws2, symlinks=True)
rc, log = common.mvn(ws2, offline=True, timeout=3000); kind, data = common.result_of(ws2, rc, log)
print("TASK state (old tests, offline):"); print(common.compact(ws2, kind, data, rc, 10)); shutil.rmtree(ws2)
