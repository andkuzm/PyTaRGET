"""Tokens and cost of the paid OpenRouter runs, retroactively, from each run's requests.jsonl (one line per request: case position, prompt/completion tokens, provider-reported cost;
retries of empty answers are separate lines of the same case). Per model, language and variant: the 59 Java / 61 Python cases of the agent comparison (stage 1), all cases of the run,
and per-case means. Run: python scripts/thesis_analysis/paid_cost_report.py"""
import json
from pathlib import Path
H = Path.home(); P = H / "eval_paid"
PAID = [("qwen3-coder-next", "qn"), ("qwen3-coder", "q3c"), ("devstral-2512", "dev"), ("glm-5.3", "glm"), ("kimi-k2.5", "kimi")]
ids = {"java": set(json.load(open(P / "ids_java59.json"))), "py": set(json.load(open(P / "ids_python61.json")))}
def load(run):
    f = P / run / "requests.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []
def agg(reqs, subset=None):
    sel = [r for r in reqs if subset is None or r["id"] in subset]
    return {"cases": len({r["id"] for r in sel}), "requests": len(sel), "tin": sum(r.get("tokens_in", 0) or 0 for r in sel), "tout": sum(r.get("completion_tokens", 0) or 0 for r in sel), "cost": sum(r.get("cost_reported", 0) or 0 for r in sel)}
tot = {"cases": 0, "requests": 0, "tin": 0, "tout": 0, "cost": 0.0}
print(f"{'model':18s} {'run':7s} | subset (59 Java / 61 Python): cases  req  tokens in   out   cost$  $/case | all cases of the run: cases  req  tokens in    out    cost$  $/case")
for name, short in PAID:
    for lang in ("java", "py"):
        for v in "AB":
            reqs = load(f"{short}_{lang}{v}")
            if not reqs: continue
            a, b = agg(reqs, ids[lang]), agg(reqs)
            for k in tot: tot[k] += b[k]
            print(f"{name:18s} {lang+v:7s} | {a['cases']:5d} {a['requests']:4d} {a['tin']:10d} {a['tout']:9d} {a['cost']:7.3f} {a['cost']/max(a['cases'],1):7.4f} | {b['cases']:5d} {b['requests']:4d} {b['tin']:10d} {b['tout']:9d} {b['cost']:7.3f} {b['cost']/max(b['cases'],1):7.4f}")
    # model totals
    allr = [r for lang in ("java", "py") for v in "AB" for r in load(f"{short}_{lang}{v}")]
    s = {lang: agg([r for v in "AB" for r in load(f"{short}_{lang}{v}")], ids[lang]) for lang in ("java", "py")}
    t = agg(allr)
    print(f"{name:18s} TOTAL   | stage-1 subset (A+B, Java 59 + Python 61): cost {s['java']['cost']+s['py']['cost']:.3f}  tokens in {s['java']['tin']+s['py']['tin']}  out {s['java']['tout']+s['py']['tout']} | whole model: cost {t['cost']:.3f}  in {t['tin']}  out {t['tout']}\n")
trial = load("trial_glm53_javaB"); tt = agg(trial)
print(f"all paid runs: cost {tot['cost']:.3f} USD, prompt tokens {tot['tin']}, completion tokens {tot['tout']}, requests {tot['requests']} (+ the 20-case GLM trial: {tt['cost']:.3f} USD)")
