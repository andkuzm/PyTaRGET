"""Agent (mini-SWE-agent + Qwen3-Coder) against the context-ablation runs on the SAME cases (the agent's prepared cases of the nested 100).
Agent verdict: a case without a usable test function (missing function / other tracked files changed) is a failure; a case whose reference does not pass in
the scoring environment is excluded from every column. 'checked' for the agent: passes AND (the case needs the repaired span OR the function has at least as
many assertions as the repaired function). Qwen/particle columns use the usual checked rule. Run: PYTHONPATH=. python scripts/thesis_analysis/agent_vs_ablation.py"""
import json, re
from pathlib import Path
R = Path.home()
meta = json.load(open(R / "eval_results_perproject_llm/contrast300/subset_meta.json")); cases, pop = meta["cases"], meta["population"]
import sys
ROOT = R / (sys.argv[1] if len(sys.argv) > 1 else "agent_work")                         # preparation root (manifest.json)
LAYOUT = R / "eval_agent_plaus" / (sys.argv[2] if len(sys.argv) > 2 else "qwen3coder_agent")   # scoring layout of the agent run
LABEL = sys.argv[3] if len(sys.argv) > 3 else "AGENT (qwen)"
manifest = json.load(open(ROOT / "manifest.json"))
ids = sorted((i for i, e in manifest.items() if e["status"] == "ok"), key=int)
layout = LAYOUT / "0.8"
case_map = json.load(open(layout / "case_map.json")); skipped = case_map["skipped"]
derived = json.load(open(layout / "splits" / "test.json")); agent_res = json.load(open(layout / "plausibility_v3.json"))["results"]
from data_processing import prediction_files
preds = {p["ID"]: p for p in prediction_files.load(layout / "test_predictions.json")}
ASSERT = re.compile(r"\bassert\b|\.assert\w*\(|\braises\b|pytest\.fail|\bexpect")
def load(p): return json.load(open(p))["results"]
def valid(r): return bool(r) and r["status"] == "evaluated" and bool(r["verdicts"])
def counted(r):
    if not valid(r) or r["verdicts"][0] != "pass": return False
    return r.get("span_removed") not in ("pass", "unparseable") or bool(r.get("assertion_kept", [False])[0])
qwen = {n: load(R / f"eval_ablation_plaus/qwen3coder_{n}/0.8/plausibility_v3.json") for n in ("A", "B", "C", "BC")}
part = {"A": load(R / "eval_openrouter_plaus/particle_1/0.8/plausibility_v3.json")}
part.update({n: load(R / f"eval_ablation_plaus/particle1_{n}/0.8/plausibility_v3.json") for n in ("B", "C", "BC")})
# agent verdicts: (valid, plain pass, checked pass)
agent = {}
inv = {i: int(pos) for pos, i in case_map["case_map"].items()}
for i in ids:
    if i in skipped: agent[i] = (True, False, False); continue
    pos = inv.get(i)
    r = agent_res.get(str(pos)) if pos is not None else None
    if not valid(r): agent[i] = (False, False, False); continue            # reference does not pass here / setup failed: excluded
    plain = r["verdicts"][0] == "pass"
    ref_fn, cand = derived[pos]["output"], preds[pos]["preds"][0]
    base = qwen["A"].get(i)
    needs = valid(base) and base.get("span_removed") not in ("pass", "unparseable")      # span control of the same case from the normal runs
    checked = plain and (needs or len(ASSERT.findall(cand)) >= len(ASSERT.findall(ref_fn)))
    agent[i] = (True, plain, checked)
common = [i for i in ids if agent[i][0] and all(valid(qwen[n].get(i)) for n in qwen) and all(valid(part[n].get(i)) for n in part)]
print(f"agent cases {len(ids)}; judged for the agent {sum(agent[i][0] for i in ids)}; common to ALL columns {len(common)}; agent failures by rule (no usable function etc.): {sum(1 for i in ids if i in skipped)}")
def rate(flags):
    out = {}
    for g in ("solved", "unsolved"):
        sel = [i for i in common if cases[i]["group"] == g]
        out[g] = (sum(flags(i) for i in sel), len(sel))
    s, u = (out[g][0] / out[g][1] if out[g][1] else float("nan") for g in out)
    return out, 100 * s, 100 * u, 100 * (pop["solved"] * s + pop["unsolved"] * u) / (pop["solved"] + pop["unsolved"])
cols = {"qwen A": lambda i: counted(qwen["A"][i]), "qwen B": lambda i: counted(qwen["B"][i]), "qwen C": lambda i: counted(qwen["C"][i]), "qwen BC": lambda i: counted(qwen["BC"][i]),
        "particle A": lambda i: counted(part["A"][i]), "particle B": lambda i: counted(part["B"][i]), "particle C": lambda i: counted(part["C"][i]), "particle BC": lambda i: counted(part["BC"][i]),
        LABEL: lambda i: agent[i][2], "agent plain": lambda i: agent[i][1]}
print(f"{'column':14s} solved   unsolved  weighted  total pass | vs qwen B (paired +/-)")
for name, f in cols.items():
    o, s, u, w = rate(f)
    plus = sum(1 for i in common if f(i) and not cols["qwen B"](i)); minus = sum(1 for i in common if cols["qwen B"](i) and not f(i))
    print(f"{name:14s} {s:6.1f}  {u:8.1f}  {w:8.1f}  {sum(f(i) for i in common):4d}/{len(common)}    | +{plus} -{minus}")
print("groups in common:", {g: sum(1 for i in common if cases[i]["group"] == g) for g in ("solved", "unsolved")})

# ---- confound: sibling tests ---------------------------------------------------------------------------------------------------------
# The agent's workspace is the repaired commit: other tests fixed in the same commit are already repaired there and can be copied. The prompt-only runs never see them.
rows = json.load(open(R / "eval_results_perproject_llm/contrast300/splits/test.json"))
from collections import Counter
key = lambda r: (r["repository_name"], r["repaired_hash"])
count = Counter(key(r) for r in rows)                      # cases of the same repository + repaired commit INSIDE the 300-case subset (a lower bound for siblings)
print("\nSplit by siblings (other cases of the same repository and repaired commit in the 300-case subset; lower bound):")
for name, sel in (("with siblings", [i for i in common if count[key(rows[int(i)])] > 1]), ("no sibling", [i for i in common if count[key(rows[int(i)])] == 1])):
    print(f"  {name:14s} n={len(sel):3d} | " + " | ".join(f"{c} {sum(cols[c](i) for i in sel)}/{len(sel)}" for c in (LABEL, "qwen B", "qwen BC", "particle BC")))
