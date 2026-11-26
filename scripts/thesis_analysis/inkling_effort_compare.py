"""Python plausibility of Inkling low vs medium vs the earlier zero-shot runs, on the cases evaluated for all of them (checked rule of plausibility_test.summarize).  usage: PYTHONPATH=. python scripts/thesis_analysis/inkling_effort_compare.py"""
import json
from pathlib import Path
from data_processing.plausibility_test import summarize
H = Path.home() / "eval_llm_plaus"
M = {"inkling low": "full_inkling_low", "inkling medium": "full_inkling_medium", "qwen3coder": "full_qwen3coder", "gptoss20b low": "full_gptoss20b"}
R = {k: (lambda d: d.get("results", d))(json.load(open(H / v / "0.8/plausibility_v3.json"))) for k, v in M.items()}
ev = [k for k in R["inkling low"] if all(k in r and r[k]["status"] == "evaluated" for r in R.values())]
print(f"cases evaluated for all four: {len(ev)}\n| model | plain @1 | plain @5 | checked @1 | checked @5 | discriminating @1 (n) |\n|---|---|---|---|---|---|")
for m, r in R.items():
    s = summarize({k: r[k] for k in ev})
    print(f"| {m} | {s['plausible@1']} | {s['plausible@k']} | {s['plausible@1_checked']} | {s['plausible@k_checked']} | {s.get('plausible@1_discriminating')} ({s.get('discriminating_cases')}) |")


def ok(r, i):
    if r["verdicts"][i] != "pass": return False
    if r.get("span_removed") not in ("pass", "unparseable"): return True
    k = r.get("assertion_kept"); return bool(k and k[i])


for rank, name in ((1, "@1"), (5, "@5")):
    f = lambda m, k: any(ok(R[m][k], i) for i in range(min(rank, len(R[m][k]["verdicts"]))))
    a = sum(f("inkling low", k) and not f("inkling medium", k) for k in ev); b = sum(f("inkling medium", k) and not f("inkling low", k) for k in ev)
    print(f"paired checked{name}: only low {a}, only medium {b}")
