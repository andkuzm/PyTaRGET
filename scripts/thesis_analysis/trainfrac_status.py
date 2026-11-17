"""Status of the training-size experiment (WP1) and the plausibility layout for finished test runs.
  python -I scripts/thesis_analysis/trainfrac_status.py [--link]   (--link builds ~/eval_trainfrac_plaus/<model>_f<N>/0.8 for every finished test run)
Prints, per (fraction, model): train/test state from the result files, metrics, the plausibility progress, and the model names ready to score."""
import json, os, sys
from pathlib import Path
H = Path.home()
OUT = H / "eval_results_perproject_trainfrac"
LAY = H / "eval_trainfrac_plaus"
TOTAL = {"codet5p": 1293, "plbart": 1342, "codegen": 1585}
link = "--link" in sys.argv
ready, scored = [], []
for frac in (20, 40, 60, 80):
    for model in ("codet5p", "plbart", "codegen"):
        d = OUT / str(frac) / model / "0.8"
        metrics = d / "metrics.json"
        state = "no checkpoint yet"
        if (d / "checkpoint-best").exists(): state = "trained"
        if (d / "training_stats.json").exists(): state = "trained (stats)"
        m = json.load(open(metrics)) if metrics.exists() else None
        if m: state = f"tested: cases {m['cases']} BLEU {m['BLEU']} CodeBLEU {m['CodeBLEU']} EM {m['EM']}"
        name = f"{model}_f{frac}"
        plaus = ""
        pf = LAY / name / "0.8" / "plausibility_v3.json"
        if m:
            if link:
                t = LAY / name / "0.8"
                (t / "splits").mkdir(parents=True, exist_ok=True)
                for src, dst in ((d / "splits" / "test.json", t / "splits" / "test.json"), (d / "test_predictions.json", t / "test_predictions.json")):
                    if dst.is_symlink() or dst.exists(): dst.unlink()
                    dst.symlink_to(src.resolve())
            if pf.exists():
                r = json.load(open(pf))["results"]; s = json.load(open(pf))["summary"]
                plaus = f" | plausibility {len(r)}/{m['cases']} done" + (f", @1 {s.get('plausible@1')} @5 {s.get('plausible@k')}" if len(r) >= m['cases'] else "")
                (scored if len(r) >= m['cases'] else ready).append(name)
            else:
                ready.append(name)
        print(f"{frac:3d}% {model:8s} {state}{plaus}")
print("READY TO SCORE (tested, plausibility missing or incomplete):", " ".join(ready) or "-")
print("SCORED:", " ".join(scored) or "-")
