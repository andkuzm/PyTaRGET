"""Table of the Java contrast-subset results (WP2): plausible@1 per model and group, raw and 'checked' (test needs the span or the candidate still
asserts), weighted overall estimate with the population sizes, on the common 273 cases whose reference passes here.
Usage: python -I scripts/thesis_analysis/java_report.py   (reads ~/eval_java_plaus/{union,span}.json and the prediction files)"""
import json, re, sys
from pathlib import Path
H = Path.home()
R = H / "eval_results_java_llm"
union, span = json.load(open(H / "eval_java_plaus/union.json")), json.load(open(H / "eval_java_plaus/span.json"))
meta273 = json.load(open(R / "contrast273/subset_meta.json")); cases273 = meta273["cases"]
pop = json.load(open(R / "contrast300/subset_meta.json"))["population"]
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b")
SRC = {"qwen3coder": (R / "local/qwen3coder/test_predictions.json", 300), "gpt-oss-20b": (R / "local/gptoss20b/test_predictions.json", 300)}
for name, d in (("ernie-5.5-flash", "java273_baidu_ernie-5.5-flash-preview_free"), ("particle-1", "java273_particle-ai_particle-1_free"),
                ("nemotron-3-ultra", "java273_nvidia_nemotron-3-ultra-550b-a55b_free")):
    SRC[name] = (H / "eval_openrouter" / d / "test_predictions.json", 273)
idx300 = {int(v["index300"]): k for k, v in cases273.items()}
group = {int(i): v["group"] for i, v in json.load(open(R / "contrast300/subset_meta.json"))["cases"].items()}
valid = [i for i in idx300 if union.get(str(i), {}).get("status") == "evaluated"]
print(f"common cases {len(idx300)}, reference passes here: {len(valid)}; groups:", {g: sum(1 for i in valid if group[i] == g) for g in set(group.values())})
print("population:", pop)
def table(checked):
    out = {}
    for name, (path, space) in SRC.items():
        d = json.load(open(path)); won = {}
        for k in d["ID"]:
            i = int(d["ID"][k]) if space == 300 else int(cases273[str(d["ID"][k])]["index300"])
            if i not in valid: continue
            c = d["preds"][k][0].strip(); s = union[str(i)]["cache"].get(c, [None])[0] if c else "empty"
            ok = s == "success"
            if ok and checked:
                ok = span.get(str(i), {}).get("span_removed") != "success" or bool(ASSERT.search(c))
            won[i] = ok
        out[name] = won
    return out
for checked in (False, True):
    print("\n== " + ("CHECKED (test needs the span or the candidate asserts)" if checked else "RAW (test passes)"))
    tab = table(checked)
    for name, won in tab.items():
        g = {gr: [sum(won.get(i, False) for i in valid if group[i] == gr), sum(1 for i in valid if group[i] == gr)] for gr in ("solved_at_1", "not_solved_at_1")}
        s, u = (g[x][0] / g[x][1] for x in g)
        ps, pu = pop.get("solved_at_1", pop.get("solved")), pop.get("not_solved_at_1", pop.get("unsolved"))
        w = (ps * s + pu * u) / (ps + pu)
        print(f"{name:18s} solved {g['solved_at_1'][0]}/{g['solved_at_1'][1]} {100*s:5.1f}%  unsolved {g['not_solved_at_1'][0]}/{g['not_solved_at_1'][1]} {100*u:5.1f}%  weighted {100*w:5.1f}%")
print("\nCodeT5+ published @1 over TaRBench: 50.29% (solved group 100% / unsolved 0% by construction)")
