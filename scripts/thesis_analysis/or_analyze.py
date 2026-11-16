"""2x2 + weighted estimate for one OpenRouter model on the contrast300 subset, from its plausibility_v3.json (+ optional retry file)."""
import json, sys, collections
from pathlib import Path
R = Path.home()
name = sys.argv[1]
meta = json.load(open(R / "eval_results_perproject_llm/contrast300/subset_meta.json"))
cases, pop = meta["cases"], meta["population"]
files = [R / f"eval_openrouter_plaus/{name}/0.8/plausibility_v3.json"]
for extra in sys.argv[2:]:
    files.append(Path(extra))
res = {}
for f in files:                      # later files (the retry of empty answers) replace earlier verdicts of the same case
    if f.exists():
        res.update(json.load(open(f))["results"])

def counted(r):                      # checked rule
    if r["verdicts"][0] != "pass": return False
    settled = r.get("span_removed") not in ("pass", "unparseable")
    return settled or bool(r.get("assertion_kept", [False])[0])

tab = {g: collections.Counter() for g in ("solved", "unsolved")}
for i, c in cases.items():
    g = c["group"]; r = res.get(i)
    if r is None: tab[g]["not scored"] += 1; continue
    if r["status"] != "evaluated": tab[g][r["status"]] += 1; continue
    v = r["verdicts"][0]
    tab[g]["evaluated"] += 1
    if v == "empty": tab[g]["empty answer"] += 1
    if v == "pass": tab[g]["pass (plain)"] += 1
    if counted(r): tab[g]["pass (checked)"] += 1
for g in tab: print(g, dict(tab[g]))
def rate(g, k): 
    e = tab[g]["evaluated"]; return tab[g][k] / e if e else float("nan")
for k in ("pass (plain)", "pass (checked)"):
    w = (pop["solved"] * rate("solved", k) + pop["unsolved"] * rate("unsolved", k)) / (pop["solved"] + pop["unsolved"])
    print(f"{k}: solved-group {100*rate('solved',k):.1f}%  unsolved-group {100*rate('unsolved',k):.1f}%  weighted overall estimate {100*w:.1f}%")
# which cases does it solve that no earlier model did, and what does it lose among cases solved by exactly one other model
new = [i for i, c in cases.items() if c["group"] == "unsolved" and i in res and res[i]["status"] == "evaluated" and counted(res[i])]
kept = [i for i, c in cases.items() if c["group"] == "solved" and i in res and res[i]["status"] == "evaluated" and counted(res[i])]
only1 = [i for i, c in cases.items() if c["group"] == "solved" and len(c["solved_by"]) == 1 and i in res and res[i]["status"] == "evaluated"]
print(f"solves {len(new)} cases that none of the five solved; keeps {len(kept)} of the solved group; of {len(only1)} cases solved by exactly one other model it solves {sum(counted(res[i]) for i in only1)}")
print("new solves:", [(i, cases[i]['repo']) for i in new][:10])
