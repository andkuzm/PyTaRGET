"""Table for the reasoning-effort sweep on 150 validation cases (slurm/reasoning_sweep.sbatch): exact match, match ignoring whitespace, empty answers, answers cut off at the output limit, mean output tokens, wall time.
usage: reasoning_sweep_report.py <sweep log> [sweep dir]"""
import ast
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_processing import prediction_files  # noqa: E402

log = Path(sys.argv[1]).read_text(errors="ignore"); root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "eval_inkling/sweep"
stats, secs, cur = {}, {}, None
for line in log.splitlines():
    m = re.match(r"=== (\w+) \w{3} ", line)
    if m: cur = m.group(1)
    m = re.search(r"saved \d+ cases; this run: (\{.*\})", line)
    if m and cur: stats[cur] = ast.literal_eval(m.group(1))
    m = re.match(r"=== (\w+) exit (\d+) seconds (\d+)", line)
    if m: secs[m.group(1)] = (int(m.group(2)), int(m.group(3)))
norm = lambda s: "".join(s.split())
base = None
print("| run | cases | exact | ignoring whitespace | empty | cut off | mean output tokens | seconds |\n|---|---|---|---|---|---|---|---|")
res = {}
for name in secs:
    f = root / name / "test_predictions.json"
    if not f.exists(): print(f"| {name} | exit {secs[name][0]} | | | | | | |"); continue
    r = prediction_files.load(f); n = len(r); st = stats.get(name, {})
    em = [x["preds"][0] == x["target"] for x in r]; nm = [norm(x["preds"][0]) == norm(x["target"]) for x in r]
    res[name] = {x["ID"]: v for x, v in zip(r, nm)}
    print(f"| {name} | {n} | {sum(em)} ({100*sum(em)/n:.1f}%) | {sum(nm)} ({100*sum(nm)/n:.1f}%) | {sum(not x['preds'][0].strip() for x in r)} | {st.get('cut_off', '?')} | {st.get('output_tokens', 0)/max(st.get('candidates', 1), 1):.0f} | {secs[name][1]} |")
ref = res.get("qwen3coder")
if ref:
    print("\nPaired against qwen3coder (ignoring whitespace): only-this / only-qwen")
    for k, v in res.items():
        if k != "qwen3coder": print(f"  {k}: {sum(v[i] and not ref[i] for i in v)} / {sum(ref[i] and not v[i] for i in v)}")
