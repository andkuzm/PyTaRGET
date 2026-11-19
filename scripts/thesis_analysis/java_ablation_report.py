"""Java context ablation: baseline prompt (A) against + failure output of the broken test (B), + definitions of the changed symbols (C) and both (BC) for the local Qwen3-Coder and
the free particle-1, on (a) all 257 cases of the Java contrast subset whose reference passes in our environment and (b) the 59 cases of the agent comparison.
Checked plausible@1 (test passes AND (test needs the repaired lines OR the candidate still asserts)), per group (solved / not solved at rank 1 by CodeT5+) and weighted with the
population sizes; paired gains/losses against A; mean extra prompt tokens of the added sections.
Run: PYTHONPATH=. python scripts/thesis_analysis/java_ablation_report.py"""
import json, re
from pathlib import Path
from transformers import AutoTokenizer
from data_processing import prediction_files
H = Path.home(); R = H / "eval_results_java_llm"
union = json.load(open(H / "eval_java_plaus/union.json")); span = json.load(open(H / "eval_java_plaus/span.json"))
m300 = json.load(open(R / "contrast300/subset_meta.json")); m273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]
group = {int(i): v["group"] for i, v in m300["cases"].items()}; pop = m300["population"]
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b")
ctx = json.load(open(H / "java_ctx/context.json")); rows = json.load(open(R / "contrast300/splits/test.json"))
id2idx = {r["ID"]: i for i, r in enumerate(rows)}
def column(path, space):
    if not (path / "test_predictions.json").exists(): return None
    out = {}
    for p in prediction_files.load(path / "test_predictions.json"):
        pos = int(p["ID"]); i = pos if space == 300 else int(m273[str(pos)]["index300"])
        c = p["preds"][0].strip(); rec = union.get(str(i), {})
        if rec.get("status") != "evaluated": continue
        plain = c != "" and rec["cache"].get(c, [None])[0] == "success"
        out[i] = (c != "", plain, plain and (span.get(str(i), {}).get("span_removed") != "success" or bool(ASSERT.search(c))))
    return out
E = H / "eval_ablation_java"
cols = {"qwen A": column(R / "local/qwen3coder", 300), "qwen B": column(E / "qwen3coder_B", 273), "qwen C": column(E / "qwen3coder_C", 273), "qwen BC": column(E / "qwen3coder_BC", 273),
        "particle A": column(H / "eval_openrouter/java273_particle-ai_particle-1_free", 273), "particle BC": column(E / "particle1_BC", 273)}
cols = {k: v for k, v in cols.items() if v is not None}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
extra = {"B": [], "C": [], "BC": []}
for e in ctx.values():
    b = len(tok.encode("### Test Failure Output:\n" + e["failure"] + "\n\n")) if e["failure"] else 0
    c = len(tok.encode("### Definitions of Changed Symbols:\n" + e["definitions"] + "\n\n")) if e["definitions"] else 0
    extra["B"].append(b); extra["C"].append(c); extra["BC"].append(b + c)
mean = lambda x: sum(x) / len(x)
def rates(sel, f):
    out = {}
    for g in ("solved_at_1", "not_solved_at_1"):
        s = [i for i in sel if group[i] == g]; out[g] = sum(f(i) for i in s) / len(s) if s else float("nan")
    return 100 * out["solved_at_1"], 100 * out["not_solved_at_1"], 100 * (pop["solved_at_1"] * out["solved_at_1"] + pop["not_solved_at_1"] * out["not_solved_at_1"]) / (pop["solved_at_1"] + pop["not_solved_at_1"])
def table(title, ids, base_of):
    print(f"\n== {title}")
    print(f"{'run':12s} {'n':>4s} {'extra tok':>9s} {'empty':>5s} | solved unsolved weighted | gain vs A (pts) | paired +/-")
    for name, c in cols.items():
        sel = [i for i in ids if i in c and i in cols[base_of(name)]]
        if not sel: continue
        v = name.split()[1]; w = rates(sel, lambda i: c[i][2]); wa = rates(sel, lambda i, b=cols[base_of(name)]: b[i][2])
        plus = sum(1 for i in sel if c[i][2] and not cols[base_of(name)][i][2]); minus = sum(1 for i in sel if cols[base_of(name)][i][2] and not c[i][2])
        print(f"{name:12s} {len(sel):4d} {mean(extra[v]) if v in extra else 0:9.0f} {sum(1 for i in sel if not c[i][0]):5d} | {w[0]:6.1f} {w[1]:8.1f} {w[2]:8.1f} | {w[2]-wa[2]:+6.1f} | +{plus} -{minus}")
base_of = lambda n: "qwen A" if n.startswith("qwen") else "particle A"
valid = sorted(i for i in (int(v["index300"]) for v in m273.values()) if union.get(str(i), {}).get("status") == "evaluated")
manifest = json.load(open(H / "agent_java/manifest.json")); warm = json.load(open(H / "agent_java/warm.json"))
agent_ids = [int(i) for i, e in manifest.items() if e["status"] == "ok" and warm.get(i) == "success"]
table("all cases of the Java contrast subset whose reference passes here", valid, base_of)
table("the 59 cases of the agent comparison", agent_ids, base_of)
broken = {}
for k, v in ctx.items(): broken[v["broken_status"]] = broken.get(v["broken_status"], 0) + 1
print("\nbroken-run statuses of the context build:", broken, "| cases without failure text:", sum(1 for v in ctx.values() if not v["failure"]), "| without definitions:", sum(1 for v in ctx.values() if not v["definitions"]))
