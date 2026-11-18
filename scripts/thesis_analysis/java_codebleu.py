"""BLEU and CodeBLEU (Java, the jCodeBLEU of the TaRGet pipeline) per model and group on the 273 contrast cases, plus CodeT5+ (authors' published rank-1
predictions) on the same cases as a reference. Run in the TaRGET-compatible environment (tree-sitter 0.20.1): from ~/PyTaRGET with
  PYTHONPATH=. ~/tarVenv/bin/python -I scripts/thesis_analysis/java_codebleu.py   (-I ignores PYTHONPATH: use plain `python` with PYTHONPATH set)
Two variants: 'as is' (whitespace tokens, like the pipeline) and 'retok' (both sides split into identifiers/numbers/single symbols, so the spacing
style of an answer does not matter)."""
import json, re, sys
from pathlib import Path
from data_processing.jCodeBLEU.bleu import corpus_bleu
from data_processing.jCodeBLEU.code_bleu import calc_code_bleu
H = Path.home(); R = H / "eval_results_java_llm"
m300 = json.load(open(R / "contrast300/subset_meta.json")); m273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]
rows = json.load(open(R / "contrast300/splits/test.json")); pop = m300["population"]
group = {int(i): v["group"] for i, v in m300["cases"].items()}; cases = {int(v["index300"]) for v in m273.values()}
retok = lambda s: " ".join(re.findall(r"\w+|[^\w\s]", s))
SRC = {"qwen3coder": (R / "local/qwen3coder", 300), "gpt-oss-20b": (R / "local/gptoss20b", 300),
       "ernie-5.5-flash": (H / "eval_openrouter/java273_baidu_ernie-5.5-flash-preview_free", 273),
       "particle-1": (H / "eval_openrouter/java273_particle-ai_particle-1_free", 273),
       "nemotron-3-ultra": (H / "eval_openrouter/java273_nvidia_nemotron-3-ultra-550b-a55b_free", 273)}
def score(targets, preds):
    preds = [p if p.strip() else "<empty>" for p in preds]
    b = corpus_bleu([[t.split()] for t in targets], [p.split() for p in preds])
    return 100 * b, 100 * calc_code_bleu([targets], preds, lang="java")
def report(name, pairs):          # pairs: {index300: (target, pred)}
    out = {}
    for variant, f in (("as is", lambda s: s), ("retok", retok)):
        res = {}
        for g in ("solved_at_1", "not_solved_at_1", "all"):
            sel = [i for i in sorted(pairs) if g == "all" or group[i] == g]
            res[g] = score([f(pairs[i][0]) for i in sel], [f(pairs[i][1]) for i in sel])
        w = lambda k: (pop["solved_at_1"] * res["solved_at_1"][k] + pop["not_solved_at_1"] * res["not_solved_at_1"][k]) / (pop["solved_at_1"] + pop["not_solved_at_1"])
        print(f"{name:22s} {variant:6s} BLEU solved {res['solved_at_1'][0]:5.1f} unsolved {res['not_solved_at_1'][0]:5.1f} weighted {w(0):5.1f} | "
              f"CodeBLEU solved {res['solved_at_1'][1]:5.1f} unsolved {res['not_solved_at_1'][1]:5.1f} weighted {w(1):5.1f}", flush=True)
for name, (p, sp) in SRC.items():
    d = json.load(open(p / "test_predictions.json")); pairs = {}
    for k in d["ID"]:
        i = int(d["ID"][k]) if sp == 300 else int(m273[str(d["ID"][k])]["index300"])
        if i in cases: pairs[i] = (d["target"][k], d["preds"][k][0])
    report(name, pairs)
# reference: CodeT5+ published predictions on the same cases (authors' files; ID = TaRBench id)
B = H / "references/TaRGet_Results_download/TaRGet_Results/Best_on_TaRBench"
pub = {p["ID"]: p for p in json.load(open(B / "test_predictions.json"))}
pairs = {i: (rows[i]["output"] if "output" in rows[i] else pub[rows[i]["ID"]]["target"], pub[rows[i]["ID"]]["preds"][0]) for i in cases if rows[i]["ID"] in pub}
report("CodeT5+ (published)", pairs)
