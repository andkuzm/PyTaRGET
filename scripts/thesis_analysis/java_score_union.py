"""Score the Java LLM outputs (local and OpenRouter) with the Java plausibility runner, incrementally (union of candidates per case).
Usage (from ~/PyTaRGET, PYTHONPATH=.): python -u scripts/thesis_analysis/java_score_union.py <out.json> <workdir> <workers> <name=predfile[:space]>...
space = 300 (positions in contrast300, default) or 273 (positions in contrast273). Runs third-party build code: not on a login node."""
import json, sys
from pathlib import Path
from data_processing.java_plausibility import JavaPlausibility

H = Path.home() / "eval_results_java_llm"
out, workdir, workers, specs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
rows = json.load(open(H / "contrast300" / "splits" / "test.json"))
meta273 = json.load(open(H / "contrast273" / "subset_meta.json"))["cases"]
cands = {}
for spec in specs:
    name, rest = spec.split("=", 1)
    path, _, space = rest.partition(":")
    data = json.load(open(path)); n = len(data["ID"])
    for k in data["ID"]:
        pos = int(data["ID"][k])
        i300 = pos if space in ("", "300") else meta273[str(pos)]["index300"]
        cands.setdefault(i300, {"target": data["target"][k], "c": []})["c"].append(data["preds"][k][0])
cases = [{"row": rows[i], "key": str(i), "target": v["target"], "candidates": v["c"]} for i, v in sorted(cands.items())]
print(len(cases), "cases", sum(len(c["candidates"]) for c in cases), "candidates", flush=True)
JavaPlausibility(workdir, log_dir=Path(out).with_suffix(".logs")).run_union(cases, out, workers)
