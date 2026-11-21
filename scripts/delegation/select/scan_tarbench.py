"""Scan TaRBench (~/references/TaRBench_download/TaRBench/projects/*/*/dataset.json) and list commit pairs (aCommit, bCommit) as candidates for real delegation tasks: many changed test methods in several test
files, heterogeneous repairs (different AST-action signatures), a moderate source diff. Output: CSV ~/delegation_exp/tarbench_pairs.csv sorted by a heterogeneity score. usage: scan_tarbench.py"""
import ast
import csv
import glob
import json
import os
from collections import Counter, defaultdict

root = os.path.expanduser("~/references/TaRBench_download/TaRBench/projects")
rows = []
for f in sorted(glob.glob(root + "/*/*/dataset.json")):
    proj = "/".join(f.split("/")[-3:-1])
    try: d = json.load(open(f))
    except Exception: continue
    pairs = defaultdict(list)
    for x in d: pairs[(x["aCommit"], x["bCommit"])].append(x)
    for (a, b), xs in pairs.items():
        files = {x["aPath"] for x in xs}
        sigs = set(); statuses = Counter(); src_lines = 0; src_files = set()
        for x in xs:
            try:
                acts = ast.literal_eval(x["astActions"]) if isinstance(x["astActions"], str) else x["astActions"]
                sigs.add(tuple(sorted({(c.get("type"), c.get("nodeType")) for c in acts})))
            except Exception: pass
            try: statuses[ast.literal_eval(x["verdict"]).get("status") if isinstance(x["verdict"], str) else x["verdict"].get("status")] += 1
            except Exception: pass
            try:
                h = ast.literal_eval(x["hunk"]) if isinstance(x["hunk"], str) else x["hunk"]
                for c in h.get("sourceChanges", []): src_lines += 1; src_files.add(c.get("path") or c.get("file") or "")
            except Exception: pass
        rows.append({"project": proj, "aCommit": a, "bCommit": b, "tests": len(xs), "test_files": len(files), "distinct_repairs": len(sigs), "compile_error": statuses.get("compile_error", 0),
                     "failure": statuses.get("failure", 0), "source_change_lines": src_lines, "aTime": min(int(x["aCommitTime"]) for x in xs)})
rows.sort(key=lambda r: (-r["distinct_repairs"], -r["test_files"], -r["tests"]))
out = os.path.expanduser("~/delegation_exp/tarbench_pairs.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "commit pairs ->", out)
