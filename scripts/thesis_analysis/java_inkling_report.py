"""Inkling-2-Mini on the Java contrast subset, scored like scripts/thesis_analysis/java_report.py (same 257 cases whose reference passes in the earlier runs, raw and checked rule, weighted with the population sizes).
Usage: python -I scripts/thesis_analysis/java_inkling_report.py low [medium]"""
import json, re, sys
from pathlib import Path
H = Path.home(); R = H / "eval_results_java_llm"
union, span = json.load(open(H / "eval_java_plaus/union.json")), json.load(open(H / "eval_java_plaus/span.json"))
cases273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]; meta = json.load(open(R / "contrast300/subset_meta.json")); pop = meta["population"]
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b")
idx300 = {int(v["index300"]): k for k, v in cases273.items()}; group = {int(i): v["group"] for i, v in meta["cases"].items()}
valid = [i for i in idx300 if union.get(str(i), {}).get("status") == "evaluated"]
ps, pu = pop.get("solved_at_1", pop.get("solved")), pop.get("not_solved_at_1", pop.get("unsolved"))
for effort in sys.argv[1:]:
    d = H / "eval_inkling" / f"java273_{effort}"; preds = json.load(open(d / "test_predictions.json")); res = json.load(open(d / "java_plausibility.json"))
    recs = {int(preds["ID"][k]): preds["preds"][k][0] for k in preds["ID"]} if isinstance(preds, dict) else {int(r["ID"]): r["preds"][0] for r in preds}
    for checked in (False, True):
        won, notjudged = {}, 0
        for i in valid:
            k = int(idx300[i]); r = res.get(str(k), {})
            if r.get("status") != "evaluated": notjudged += 1; won[i] = False; continue
            ok = any(v.get("status") == "success" for v in r["verdicts"][:1]); c = recs[k].strip()
            if ok and checked: ok = span.get(str(i), {}).get("span_removed") != "success" or bool(ASSERT.search(c))
            won[i] = ok
        g = {gr: [sum(won[i] for i in valid if group[i] == gr), sum(1 for i in valid if group[i] == gr)] for gr in ("solved_at_1", "not_solved_at_1")}
        s, u = (g[x][0] / g[x][1] for x in g); w = (ps * s + pu * u) / (ps + pu)
        print(f"inkling {effort:6s} {'CHECKED' if checked else 'RAW    '} solved {g['solved_at_1'][0]}/{g['solved_at_1'][1]} {100*s:5.1f}%  unsolved {g['not_solved_at_1'][0]}/{g['not_solved_at_1'][1]} {100*u:5.1f}%  weighted {100*w:5.1f}%  (cases the reference did not pass in this run, counted as failed: {notjudged})")
    ex = sum(1 for i in valid if "".join(recs[int(idx300[i])].split()) == "".join((preds["target"][idx300[i]] if isinstance(preds, dict) else "").split()))
    print(f"  verdicts: " + str(__import__('collections').Counter((res[k]['verdicts'][0]['status'] if res[k].get('verdicts') else res[k]['status']) for k in res)))
