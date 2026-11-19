"""Python sibling-safe agent: 40-call run (agent_work2 / qwen3coder_sibling_safe) against the 80-call run (agent_work3 / qwen3coder_sibling_safe_80), paired on the cases
judged in BOTH runs (and in the Qwen A/B/BC and particle-1 B/BC columns for context). Checked plausible@1, weighted with the population sizes of the contrast subset.
Run: PYTHONPATH=. python scripts/thesis_analysis/agent_python_40_vs_80.py"""
import json, re
from pathlib import Path
from data_processing import prediction_files
R = Path.home()
meta = json.load(open(R / "eval_results_perproject_llm/contrast300/subset_meta.json")); cases, pop = meta["cases"], meta["population"]
ASSERT = re.compile(r"\bassert\b|\.assert\w*\(|\braises\b|pytest\.fail|\bexpect")
def valid(r): return bool(r) and r["status"] == "evaluated" and bool(r["verdicts"])
def counted(r):
    if not valid(r) or r["verdicts"][0] != "pass": return False
    return r.get("span_removed") not in ("pass", "unparseable") or bool(r.get("assertion_kept", [False])[0])
qwenA = json.load(open(R / "eval_ablation_plaus/qwen3coder_A/0.8/plausibility_v3.json"))["results"]
def agent(root, name):
    manifest = json.load(open(R / root / "manifest.json")); ids = [i for i, e in manifest.items() if e["status"] == "ok"]
    layout = R / "eval_agent_plaus" / name / "0.8"
    cm = json.load(open(layout / "case_map.json")); skipped = cm["skipped"]; inv = {i: int(p) for p, i in cm["case_map"].items()}
    derived = json.load(open(layout / "splits/test.json")); res = json.load(open(layout / "plausibility_v3.json"))["results"]
    preds = {p["ID"]: p for p in prediction_files.load(layout / "test_predictions.json")}
    out = {}
    for i in ids:
        if i in skipped: out[i] = (True, False, False); continue
        pos = inv.get(i); r = res.get(str(pos)) if pos is not None else None
        if not valid(r): out[i] = (False, False, False); continue
        plain = r["verdicts"][0] == "pass"
        needs = valid(qwenA.get(i)) and qwenA[i].get("span_removed") not in ("pass", "unparseable")
        out[i] = (True, plain, plain and (needs or len(ASSERT.findall(preds[pos]["preds"][0])) >= len(ASSERT.findall(derived[pos]["output"]))))
    return out, skipped
a40, s40 = agent("agent_work2", "qwen3coder_sibling_safe"); a80, s80 = agent("agent_work3", "qwen3coder_sibling_safe_80")
common = [i for i in a40 if i in a80 and a40[i][0] and a80[i][0] and valid(qwenA.get(i))]
def weighted(f, sel):
    r = {}
    for g in ("solved", "unsolved"):
        s = [i for i in sel if cases[i]["group"] == g]; r[g] = sum(f(i) for i in s) / len(s) if s else float("nan")
    return 100 * r["solved"], 100 * r["unsolved"], 100 * (pop["solved"] * r["solved"] + pop["unsolved"] * r["unsolved"]) / (pop["solved"] + pop["unsolved"])
print(f"cases in both runs {len(a40.keys() & a80.keys())}; judged in both and baseline {len(common)}; failures by rule: 40-call {sum(1 for i in common if i in s40)}, 80-call {sum(1 for i in common if i in s80)}")
for name, a in (("AGENT 40 calls", a40), ("AGENT 80 calls", a80)):
    s, u, w = weighted(lambda i: a[i][2], common); print(f"{name:16s} solved {s:5.1f} unsolved {u:5.1f} weighted {w:5.1f}  total {sum(a[i][2] for i in common)}/{len(common)}  | plain {sum(a[i][1] for i in common)}")
plus = sum(1 for i in common if a80[i][2] and not a40[i][2]); minus = sum(1 for i in common if a40[i][2] and not a80[i][2])
print(f"paired 80 vs 40: +{plus} -{minus}")
print("by rule violation, 80-call failures:", {k: sum(1 for i in common if s80.get(i) == k) for k in set(s80.values())}, "| 40-call:", {k: sum(1 for i in common if s40.get(i) == k) for k in set(s40.values())})
