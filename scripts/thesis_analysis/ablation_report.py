"""Context-ablation table: plausible@1 (plain and checked) per run, group (solved / unsolved by the five earlier models) and weighted with the population
sizes of the contrast subset; paired gains/losses against the baseline A; extra prompt tokens of the added sections.
Run (python -I): scripts/thesis_analysis/ablation_report.py"""
import json, math, re
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home()
meta = json.load(open(R / "eval_results_perproject_llm/contrast300/subset_meta.json")); cases, pop = meta["cases"], meta["population"]
def load(path):
    return json.load(open(path))["results"]
def valid(r):
    return bool(r) and r["status"] == "evaluated" and bool(r["verdicts"])
def counted(r):
    if not valid(r) or r["verdicts"][0] != "pass": return False
    return r.get("span_removed") not in ("pass", "unparseable") or bool(r.get("assertion_kept", [False])[0])
def rates(res, ids):
    out = {}
    for g in ("solved", "unsolved"):
        sel = [i for i in ids if cases[i]["group"] == g and valid(res.get(i))]
        out[g] = (sum(res[i]["verdicts"][0] == "pass" for i in sel), sum(counted(res[i]) for i in sel), len(sel))
    return out
def weighted(o, k):
    s, u = (o[g][k] / o[g][2] if o[g][2] else float("nan") for g in ("solved", "unsolved"))
    return 100 * (pop["solved"] * s + pop["unsolved"] * u) / (pop["solved"] + pop["unsolved"]), s, u
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ctx = json.load(open(R / "eval_ablation/context.json"))
rows = json.load(open(R / "eval_results_perproject_llm/contrast300/splits/test.json"))
from data_processing.context_ablation import case_key
extra = {"B": [], "C": [], "BC": []}
for r in rows:
    e = ctx[case_key(r)]
    b = len(tok.encode("### Test Failure Output:\n" + e["failure"] + "\n\n")) if e["failure"] else 0
    c = len(tok.encode("### Definitions of Changed Symbols:\n" + e["definitions"] + "\n\n")) if e["definitions"] else 0
    extra["B"].append(b); extra["C"].append(c); extra["BC"].append(b + c)
mean = lambda x: sum(x) / len(x)
def table(title, runs, ids, base):
    print(f"\n== {title} ({len(ids)} cases)")
    print(f"{'run':14s} {'extra tok':>9s} {'empty':>5s} | plain: solved unsolved weighted | CHECKED: solved unsolved weighted | gain vs A (checked, pts) | paired +/-")
    for name, (res, variant) in runs.items():
        o = rates(res, ids); wp, sp, up = weighted(o, 0); wc, sc, uc = weighted(o, 1)
        empty = sum(1 for i in ids if valid(res.get(i)) and res[i]["verdicts"][0] == "empty")
        gain = wc - weighted(rates(base, ids), 1)[0]
        both = [i for i in ids if valid(res.get(i)) and valid(base.get(i))]          # paired: cases judged in both runs
        plus = sum(1 for i in both if counted(res[i]) and not counted(base[i]))
        minus = sum(1 for i in both if counted(base[i]) and not counted(res[i]))
        ex = mean(extra[variant]) if variant else 0
        print(f"{name:14s} {ex:9.0f} {empty:5d} | {100*sp:6.1f} {100*up:8.1f} {wp:8.1f} | {100*sc:6.1f} {100*uc:8.1f} {wc:8.1f} | {gain:+6.1f} | +{plus} -{minus}")
q = {n: load(R / f"eval_ablation_plaus/qwen3coder_{n}/0.8/plausibility_v3.json") for n in ("A", "B", "C", "BC")}
table("Qwen3-Coder-30B-A3B (local), one candidate", {f"qwen A": (q["A"], None), "qwen B": (q["B"], "B"), "qwen C": (q["C"], "C"), "qwen BC": (q["BC"], "BC")}, list(cases), q["A"])
ids100 = sorted(json.load(open(R / "eval_ablation/particle1_B/test_predictions.json"))["ID"].values())
ids100 = [str(i) for i in ids100]
pA = load(R / "eval_openrouter_plaus/particle_1/0.8/plausibility_v3.json")
pr = {n: load(R / f"eval_ablation_plaus/particle1_{n}/0.8/plausibility_v3.json") for n in ("B", "C", "BC") if (R / f"eval_ablation_plaus/particle1_{n}/0.8/plausibility_v3.json").exists()}
if pr:
    table("particle-1 (free, remote), one candidate, A = first pass without retry", {"particle A": (pA, None), **{f"particle {n}": (r, n) for n, r in pr.items()}}, ids100, pA)
