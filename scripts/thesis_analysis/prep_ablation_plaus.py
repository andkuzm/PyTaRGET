"""Plausibility layout for the context-ablation runs: ~/eval_ablation_plaus/<run>/0.8/{splits/test.json (link to contrast300), test_predictions.json (fences stripped)}.
Usage (from ~/PyTaRGET, PYTHONPATH=.): python scripts/thesis_analysis/prep_ablation_plaus.py <run dir name under ~/eval_ablation>..."""
import sys
from pathlib import Path
from data_processing.openrouter_run import strip_code_fence
from data_processing import prediction_files
for name in sys.argv[1:]:
    run_dir = Path.home() / "eval_ablation" / name
    out = Path.home() / "eval_ablation_plaus" / name / "0.8"
    (out / "splits").mkdir(parents=True, exist_ok=True)
    link = out / "splits" / "test.json"
    if not link.exists():
        link.symlink_to(Path.home() / "eval_results_perproject_llm" / "contrast300" / "splits" / "test.json")
    recs = prediction_files.load(run_dir / "test_predictions.json")
    for r in recs:
        r["preds"] = [strip_code_fence(x) for x in r["preds"]]
    prediction_files.save(recs, out / "test_predictions.json")
    print(f"{name}: {len(recs)} cases -> {out}")
