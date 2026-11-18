"""Java span control: run each case's test with the repaired span REMOVED (empty candidate). If it still passes, the test does not need the span
(not discriminating: any 'success' of a candidate is then meaningless); if it fails to compile or fails, the span is needed.
Usage (from ~/PyTaRGET, PYTHONPATH=.): python -u scripts/thesis_analysis/java_span_control.py <out.json> <workdir> <workers> [union.json: only cases with reference_fails excluded]
Resumable. Runs third-party build code: not on a login node."""
import json, shutil, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from data_processing.java_plausibility import JavaPlausibility, split_method

H = Path.home() / "eval_results_java_llm"
out, workdir, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
union = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
rows = json.load(open(H / "contrast300" / "splits" / "test.json"))
res = json.loads(out.read_text()) if out.exists() else {}
runner, lock, by_repo = JavaPlausibility(workdir, log_dir=out.with_suffix(".logs")), threading.Lock(), {}
for i, row in enumerate(rows):
    if str(i) in res or union.get(str(i), {}).get("status") != "evaluated":
        continue
    by_repo.setdefault(row["ID"].split(":")[0], []).append((i, row))

def work(item):
    repo, group = item
    for i, row in group:
        try:
            project = runner.prepare(repo, row["aCommit"])
            rel = Path(row["aPath"]); original = (project / rel).read_text(); cls, meth = split_method(row["name"])
            status, java, _ = runner.execute(project, row, original, "", rel, cls, meth)
            rec = {"id": row["ID"], "span_removed": status}
        except Exception as e:
            rec = {"id": row["ID"], "span_removed": "error", "detail": repr(e)[:200]}
        with lock:
            res[str(i)] = rec; tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.replace(out)
        print(repo, i, rec["span_removed"], flush=True)
    shutil.rmtree(Path(workdir) / "repos" / repo.replace("/", "__"), ignore_errors=True)

print(len(by_repo), "repos", sum(len(g) for g in by_repo.values()), "cases", flush=True)
with ThreadPoolExecutor(workers) as pool:
    list(pool.map(work, by_repo.items()))
