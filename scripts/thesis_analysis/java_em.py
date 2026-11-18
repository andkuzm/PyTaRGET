"""Exact match of the Java candidates (one per case) per model and group: whitespace-insensitive equality with the reference (raw code) and with the
tokenised target; over all 273 cases and over the 257 whose reference passes here. Weighted with the population sizes."""
import json, re
from pathlib import Path
H = Path.home(); R = H / "eval_results_java_llm"
raw = json.load(open(R / "contrast300/reference_raw.json")); m300 = json.load(open(R / "contrast300/subset_meta.json")); m273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]
union = json.load(open(H / "eval_java_plaus/union.json")); pop = m300["population"]; group = {int(i): v["group"] for i, v in m300["cases"].items()}
n = lambda s: re.sub(r"\s+", "", s)
SRC = {"qwen3coder": (R / "local/qwen3coder", 300), "gpt-oss-20b": (R / "local/gptoss20b", 300),
       "ernie-5.5-flash": (H / "eval_openrouter/java273_baidu_ernie-5.5-flash-preview_free", 273),
       "particle-1": (H / "eval_openrouter/java273_particle-ai_particle-1_free", 273),
       "nemotron-3-ultra": (H / "eval_openrouter/java273_nvidia_nemotron-3-ultra-550b-a55b_free", 273)}
cases = {int(v["index300"]) for v in m273.values()}
def run(subset, label):
    print(f"\n== EM (whitespace-insensitive) vs reference raw code, {label} ({len(subset)} cases)")
    for name, (p, sp) in SRC.items():
        d = json.load(open(p / "test_predictions.json")); hit = {}
        for k in d["ID"]:
            i = int(d["ID"][k]) if sp == 300 else int(m273[str(d["ID"][k])]["index300"])
            if i in subset: hit[i] = n(d["preds"][k][0]) == n(raw[str(i)])
        g = {x: [sum(hit.get(i, False) for i in subset if group[i] == x), sum(1 for i in subset if group[i] == x)] for x in ("solved_at_1", "not_solved_at_1")}
        s, u = g["solved_at_1"][0] / g["solved_at_1"][1], g["not_solved_at_1"][0] / g["not_solved_at_1"][1]
        w = (pop["solved_at_1"] * s + pop["not_solved_at_1"] * u) / (pop["solved_at_1"] + pop["not_solved_at_1"])
        print(f"{name:18s} solved {g['solved_at_1'][0]}/{g['solved_at_1'][1]} {100*s:5.1f}%  unsolved {g['not_solved_at_1'][0]}/{g['not_solved_at_1'][1]} {100*u:5.1f}%  weighted {100*w:5.1f}%  all {sum(hit.values())}/{len(hit)} {100*sum(hit.values())/len(hit):.1f}%")
run(cases, "all 273")
run({i for i in cases if union.get(str(i), {}).get("status") == "evaluated"}, "reference passes here")
