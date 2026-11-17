"""Learning curve of the training-size experiment (WP1) -> docs/results/trainfrac_curve.json and a printed table.
Per model and fraction: training cases, EM/BLEU/CodeBLEU (metrics.json), plausibility summary when ALL cases are scored (else only the
progress, a partial summary would depend on the repository order). The 100% point comes from ~/eval_results_perproject."""
import json
from pathlib import Path
H = Path.home()
OUT = H / "eval_results_perproject_trainfrac"
LAY = H / "eval_trainfrac_plaus"
FULL = H / "eval_results_perproject"
curve = {}
def plaus(path, total):
    if not path.exists():
        return {"done": 0, "total": total}
    d = json.load(open(path)); n = len(d["results"])
    if n < total:
        return {"done": n, "total": total}
    keep = ("evaluated", "plausible@1", "plausible@k", "plausible@1_checked", "plausible@k_checked", "not_discriminating", "span_unparseable", "candidate_timeouts")
    return {"done": n, "total": total} | {k: d["summary"][k] for k in keep if k in d["summary"]}
for model in ("codet5p", "plbart", "codegen"):
    rows = {}
    for frac in (20, 40, 60, 80):
        d = OUT / str(frac) / model / "0.8"
        if not (d / "metrics.json").exists():
            continue
        m = json.load(open(d / "metrics.json")); info = json.load(open(d / "trainfrac.json"))
        rows[f"{frac}%"] = {"train_cases": info["cases_kept"], "train_share": info["share_kept"], "test_cases": m["cases"], "EM": m["EM"], "BLEU": m["BLEU"],
                            "CodeBLEU": m["CodeBLEU"], "plausibility": plaus(LAY / f"{model}_f{frac}" / "0.8" / "plausibility_v3.json", m["cases"])}
    m = json.load(open(FULL / model / "0.8" / "metrics.json"))
    st = json.load(open(FULL / model / "0.8" / "training_stats.json"))
    rows["100%"] = {"train_cases": st["train_set_size"], "train_share": 1.0, "test_cases": m["cases"], "EM": m["EM"], "BLEU": m["BLEU"], "CodeBLEU": m["CodeBLEU"],
                    "plausibility": plaus(FULL / model / "0.8" / "plausibility_v3.json", m["cases"])}
    curve[model] = rows
json.dump(curve, open("docs/results/trainfrac_curve.json", "w"), indent=1)
print(f"{'model':8s} {'frac':5s} {'train':>6s} {'EM':>6s} {'BLEU':>6s} {'CodeBLEU':>8s}  plausibility (plain @1/@5 | checked @1/@5)")
for model, rows in curve.items():
    for frac, r in rows.items():
        p = r["plausibility"]
        ptxt = (f"{p['plausible@1']}/{p['plausible@k']} | {p.get('plausible@1_checked')}/{p.get('plausible@k_checked')}" if "plausible@1" in p
                else f"scored {p['done']}/{p['total']}")
        print(f"{model:8s} {frac:5s} {r['train_cases']:6d} {r['EM']:6.2f} {r['BLEU']:6.2f} {r['CodeBLEU']:8.2f}  {ptxt}")
