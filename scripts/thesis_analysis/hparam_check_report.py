"""Compare the hyper-parameter checks (CodeT5+, full training set) with the baseline run; rule: a variant is 'clearly better' when its
exact match is at least 2.0 points higher and its CodeBLEU is not lower (1293 test cases, standard error of EM about 1.1 points)."""
import json
from pathlib import Path
H = Path.home()
base = json.load(open(H / "eval_results_perproject/codet5p/0.8/metrics.json"))
runs = {"a: epoch-5 checkpoint (baseline training)": "epoch5_last", "b: paper settings, batch 1": "paper_settings_bs1",
        "c: paper lr/epochs, batch 8": "paper_lr_bs8"}
print(f"baseline (lr 5e-5, batch 8, best epoch 2): cases {base['cases']} EM {base['EM']} BLEU {base['BLEU']} CodeBLEU {base['CodeBLEU']}")
for label, d in runs.items():
    f = H / "eval_hparam_check" / d / "codet5p/0.8/metrics.json"
    stats = H / "eval_hparam_check" / d / "codet5p/0.8/training_stats.json"
    extra = ""
    if stats.exists():
        ep = json.load(open(stats))["epochs"]
        if ep:
            best = min(ep, key=lambda e: e["valid_loss"])
            extra = f" | epochs so far {len(ep)}, best epoch {best['epoch']}, valid loss {best['valid_loss']}"
        else:
            extra = " | training: no epoch finished yet"
    if not f.exists():
        print(f"{label}: not finished{extra}"); continue
    m = json.load(open(f))
    better = (m["EM"] - base["EM"] >= 2.0) and (m["CodeBLEU"] >= base["CodeBLEU"])
    print(f"{label}: cases {m['cases']} EM {m['EM']} ({m['EM'] - base['EM']:+.2f}) BLEU {m['BLEU']} ({m['BLEU'] - base['BLEU']:+.2f}) "
          f"CodeBLEU {m['CodeBLEU']} ({m['CodeBLEU'] - base['CodeBLEU']:+.2f}){extra} => {'CLEARLY BETTER' if better else 'not clearly better'}")
