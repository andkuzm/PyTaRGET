"""Paid OpenRouter models (qwen3-coder-next, qwen3-coder, devstral-2512, glm-5.3, kimi-k2.5) against the free/local models: variants A (baseline prompt) and B (+ failure output),
one candidate per case, checked plausible@1 per group and weighted with the population sizes; paired gains/losses of B against A; real cost (sum of the reported cost per run).
Java: (a) the 59 cases of the agent comparison, (b) all cases covered by the run. Python: the 61 common cases.
Run: PYTHONPATH=. python scripts/thesis_analysis/paid_models_report.py"""
import json, re
from pathlib import Path
from data_processing import prediction_files
H = Path.home(); R = H / "eval_results_java_llm"; P = H / "eval_paid"
PAID = [("qwen3-coder-next", "qn"), ("qwen3-coder", "q3c"), ("devstral-2512", "dev"), ("glm-5.3", "glm"), ("kimi-k2.5", "kimi")]
def cost(run):
    f = P / run / "requests.jsonl"
    return sum(json.loads(l).get("cost_reported", 0) or 0 for l in open(f)) if f.exists() else 0.0
def weights(meta):
    pop = meta["population"]; return pop[next(k for k in pop if k.startswith("solved") and "not" not in k and "un" not in k)], pop[next(k for k in pop if k.startswith("not") or k.startswith("un"))]
def rates(sel, f, group, w):
    """(solved-group %, unsolved-group %, weighted %): the groups are found by name ('solved'/'solved_at_1' vs 'unsolved'/'not_solved_at_1'), not by sort order."""
    names = set(group.values())
    solved = next(g for g in names if g.startswith("solved")); unsolved = next(g for g in names if g != solved)
    r = {}
    for g in (solved, unsolved):
        s_ = [i for i in sel if group[i] == g]; r[g] = sum(f(i) for i in s_) / len(s_) if s_ else float("nan")
    return 100 * r[solved], 100 * r[unsolved], 100 * (w[0] * r[solved] + w[1] * r[unsolved]) / (w[0] + w[1])
# ---------------- Java ----------------
union = json.load(open(H / "eval_java_plaus/union.json")); span = json.load(open(H / "eval_java_plaus/span.json"))
m300 = json.load(open(R / "contrast300/subset_meta.json")); m273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]
jgroup = {int(i): v["group"] for i, v in m300["cases"].items()}; jw = weights(m300)
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b")
def jcol(path, space=273):
    f = Path(path) / "test_predictions.json"
    if not f.exists(): return {}
    out = {}
    for p in prediction_files.load(f):
        pos = int(p["ID"]); i = pos if space == 300 else int(m273[str(pos)]["index300"])
        c = p["preds"][0].strip(); rec = union.get(str(i), {})
        if rec.get("status") != "evaluated": continue
        plain = c != "" and rec["cache"].get(c, [None])[0] == "success"
        out[i] = (c != "", plain and (span.get(str(i), {}).get("span_removed") != "success" or bool(ASSERT.search(c))))
    return out
jcols = {"qwen3-coder-30b (local) A": jcol(R / "local/qwen3coder", 300), "qwen3-coder-30b (local) B": jcol(H / "eval_ablation_java/qwen3coder_B"),
         "particle-1 (free) A": jcol(H / "eval_openrouter/java273_particle-ai_particle-1_free"), "particle-1 (free) B": jcol(H / "eval_ablation_java/particle1_B")}
for name, short in PAID:
    for v in "AB": jcols[f"{name} {v}"] = jcol(P / f"{short}_java{v}")
manifest = json.load(open(H / "agent_java/manifest.json")); warm = json.load(open(H / "agent_java/warm.json"))
ids59 = [int(i) for i, e in manifest.items() if e["status"] == "ok" and warm.get(i) == "success"]
valid257 = sorted(int(v["index300"]) for v in m273.values() if union.get(str(v["index300"]), {}).get("status") == "evaluated")
def jtable(title, ids):
    print(f"\n== JAVA: {title}")
    print(f"{'model / variant':32s} {'n':>4s} {'empty':>5s} | solved unsolved weighted | B-A pts (paired +/-)   cost($)")
    for name in list(dict.fromkeys(k.rsplit(" ", 1)[0] for k in jcols)):
        a, b = jcols.get(name + " A", {}), jcols.get(name + " B", {})
        for v, c in (("A", a), ("B", b)):
            sel = [i for i in ids if i in c]
            if not sel: continue
            w = rates(sel, lambda i: c[i][1], jgroup, jw)
            extra = ""
            if v == "B" and a:
                both = [i for i in sel if i in a]; wa = rates(both, lambda i: a[i][1], jgroup, jw); wb = rates(both, lambda i: c[i][1], jgroup, jw)
                extra = f"| {wb[2]-wa[2]:+6.1f} (+{sum(1 for i in both if c[i][1] and not a[i][1])}/-{sum(1 for i in both if a[i][1] and not c[i][1])})"
            short = dict((n, s) for n, s in PAID).get(name)
            cst = f"{cost(f'{short}_java{v}'):.3f}" if short else ""
            print(f"{(name + ' ' + v):32s} {len(sel):4d} {sum(1 for i in sel if not c[i][0]):5d} | {w[0]:6.1f} {w[1]:8.1f} {w[2]:8.1f} {extra:24s} {cst:>8s}")
jtable("the 59 cases of the agent comparison", ids59)
jtable("all cases covered by each run (n differs; Kimi only the 59 cases)", valid257)
# ---------------- Python ----------------
pm = json.load(open(H / "eval_results_perproject_llm/contrast300/subset_meta.json")); pgroup = {int(i): v["group"] for i, v in pm["cases"].items()}; pw = weights(pm)
ids61 = json.load(open(P / "ids_python61.json"))
def pcounted(r):
    if not r or r.get("status") != "evaluated" or not r.get("verdicts") or r["verdicts"][0] != "pass": return None if (not r or r.get("status") != "evaluated" or not r.get("verdicts")) else False
    return r.get("span_removed") not in ("pass", "unparseable") or bool(r.get("assertion_kept", [False])[0])
def pcol(path, empty_from=None):
    f = Path(path)
    if not f.exists(): return {}
    res = json.load(open(f))["results"]; out = {}
    for k, r in res.items():
        v = pcounted(r)
        if v is not None: out[int(k)] = (r["verdicts"][0] != "empty", v)
    return out
pcols = {"qwen3-coder-30b (local) A": pcol(H / "eval_ablation_plaus/qwen3coder_A/0.8/plausibility_v3.json"), "qwen3-coder-30b (local) B": pcol(H / "eval_ablation_plaus/qwen3coder_B/0.8/plausibility_v3.json"),
         "particle-1 (free) A": pcol(H / "eval_openrouter_plaus/particle_1/0.8/plausibility_v3.json"), "particle-1 (free) B": pcol(H / "eval_ablation_plaus/particle1_B/0.8/plausibility_v3.json")}
for name, short in PAID:
    for v in "AB": pcols[f"{name} {v}"] = pcol(H / f"eval_paid_plaus/{short}_py{v}/0.8/plausibility_v3.json")
print("\n== PYTHON: the 61 common cases (judged in every shown run is not enforced: n shown)")
print(f"{'model / variant':32s} {'n':>4s} {'empty':>5s} | solved unsolved weighted | B-A pts (paired +/-)   cost($)")
for name in list(dict.fromkeys(k.rsplit(" ", 1)[0] for k in pcols)):
    a, b = pcols.get(name + " A", {}), pcols.get(name + " B", {})
    for v, c in (("A", a), ("B", b)):
        sel = [i for i in ids61 if i in c]
        if not sel: continue
        w = rates(sel, lambda i: c[i][1], pgroup, pw)
        extra = ""
        if v == "B" and a:
            both = [i for i in sel if i in a]; wa = rates(both, lambda i: a[i][1], pgroup, pw); wb = rates(both, lambda i: c[i][1], pgroup, pw)
            extra = f"| {wb[2]-wa[2]:+6.1f} (+{sum(1 for i in both if c[i][1] and not a[i][1])}/-{sum(1 for i in both if a[i][1] and not c[i][1])})"
        short = dict((n, s) for n, s in PAID).get(name)
        cst = f"{cost(f'{short}_py{v}'):.3f}" if short else ""
        print(f"{(name + ' ' + v):32s} {len(sel):4d} {sum(1 for i in sel if not c[i][0]):5d} | {w[0]:6.1f} {w[1]:8.1f} {w[2]:8.1f} {extra:24s} {cst:>8s}")
print("\nFor reference (agent, 80 calls, same cases): Java 59 strict 63.4 / relaxed 70.3; Python 61: 48.3.")
