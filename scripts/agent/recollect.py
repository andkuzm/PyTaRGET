"""Recompute the collect_result part of existing agent results from the (still existing) workspaces (after a change of collect_result).
Usage: PYTHONPATH=. python scripts/agent/recollect.py <split dir> <agent work root> <agent output dir>"""
import json, sys
from pathlib import Path
from data_processing import agent_workspace as aw
rows = json.load(open(Path(sys.argv[1]) / "splits" / "test.json"))
manifest = json.load(open(Path(sys.argv[2]) / "manifest.json"))
for result_file in sorted(Path(sys.argv[3]).glob("*/result.json")):
    i = result_file.parent.name
    result = json.loads(result_file.read_text())
    result.update(aw.collect_result(rows[int(i)], manifest[i]["ws"]))
    result_file.write_text(json.dumps(result))
print("recollected", len(list(Path(sys.argv[3]).glob("*/result.json"))))
