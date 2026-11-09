"""
PR@k of the fine-tuned models (our Python data) next to the TaRGet paper's best model (Java), from the plausibility
results: the first run (candidates 1-5, `plausibility.json`) and the extension (candidates 6-20,
`plausibility_ranks6_20.json`), and the paper's published predictions and verdicts.

    python -m data_processing.plausibility_report [--results ~/eval_results] [--tarbench-results .../Best_on_TaRBench]
"""
import argparse
import json
from pathlib import Path

from data_processing.metrics import normalize_indentation

KS = (1, 5, 10, 20)
INVALID = {"timeout", "test_not_executed", "unrelated_compile_error", "expected_exception_failure", "test_match_failure",
           "unrelated_failure", "dependency_error", "pom_not_found", "invalid_edit_sequence", "unknown"}


def pct(count, total):
    return round(100 * count / total, 2) if total else None


def ours(results_dir, model, train_size="0.8"):
    """{k: {"cases", "pr", "em"}} over the cases that were evaluated and have verdicts for all candidates up to k."""
    base = Path(results_dir) / model / train_size
    predictions = json.loads((base / "test_predictions.json").read_text(encoding="utf-8"))
    first = json.loads((base / "plausibility.json").read_text(encoding="utf-8"))["results"]
    later_file = base / "plausibility_ranks6_20.json"
    later = json.loads(later_file.read_text(encoding="utf-8"))["results"] if later_file.exists() else {}
    out = {k: {"cases": 0, "pr": 0, "em": 0} for k in KS}
    for case_id, entry in first.items():
        if entry["status"] != "evaluated":
            continue
        verdicts = list(entry["verdicts"])
        extra = later.get(case_id)
        if extra and extra["status"] == "evaluated":
            verdicts += extra["verdicts"]
        target = normalize_indentation(predictions["target"][case_id])
        candidates = predictions["preds"][case_id]
        for k in KS:
            if len(verdicts) < min(k, len(candidates)):
                continue                                      # candidates up to k were not all judged for this case
            out[k]["cases"] += 1
            out[k]["pr"] += "pass" in verdicts[:k]
            out[k]["em"] += any(normalize_indentation(c) == target for c in candidates[:k])
    return {k: {"cases": v["cases"], "PR": pct(v["pr"], v["cases"]), "EM": pct(v["em"], v["cases"])} for k, v in out.items()}


def paper(directory):
    """The paper's best model from its published predictions and verdicts (rule of test_run.py: cases with >= 75%
    invalid verdicts are dropped for PR)."""
    directory = Path(directory)
    predictions = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    verdicts = {}
    for row in json.loads((directory / "test_verdicts.json").read_text(encoding="utf-8")):
        verdicts.setdefault(row["ID"], {})[row["rank"]] = row["verdict"]["status"]
    kept = [c for c in predictions if sum(v in INVALID for v in verdicts[c["ID"]].values()) / len(verdicts[c["ID"]]) < 0.75
            or len(verdicts[c["ID"]]) == 1]
    out = {}
    for k in KS + (40,):
        out[k] = {"cases": len(kept),
                  "PR": pct(sum(any(verdicts[c["ID"]][r] == "success" for r in range(k)) for c in kept), len(kept)),
                  "EM": pct(sum(any(p.strip() == c["target"].strip() for p in c["preds"][:k]) for c in predictions), len(predictions))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path.home() / "eval_results"))
    parser.add_argument("--models", nargs="+", default=["codet5p", "plbart", "codegen"])
    parser.add_argument("--tarbench-results", default=str(Path.home() / "references/TaRGet_Results_download/TaRGet_Results/Best_on_TaRBench"))
    parser.add_argument("--out", default=None, help="json file for the numbers (default: <results>/plausibility_report.json)")
    args = parser.parse_args(argv)
    report = {model: ours(args.results, model) for model in args.models}
    report["paper_codet5p_java"] = paper(args.tarbench_results)
    out = Path(args.out or Path(args.results) / "plausibility_report.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{'k':>3} " + " | ".join(f"{name:>22}" for name in report))
    for k in KS:
        print(f"{k:>3} " + " | ".join(f"PR {str(report[n][k]['PR']):>5} EM {str(report[n][k]['EM']):>5} ({report[n][k]['cases']:>4})".rjust(22) for n in report))
    print(f"(PR / EM in %, in brackets: number of cases with verdicts up to k; wrote {out})")


if __name__ == "__main__":
    main()
