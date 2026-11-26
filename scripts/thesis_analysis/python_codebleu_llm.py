"""BLEU / CodeBLEU of the zero-shot LLM Python runs on the per-project test set. 'pipeline' = the definition of Tester_llm.compute_scores (the exact-match candidate if one of the 5 equals the reference,
else the first), 'first' = always the first answer; 'retok' = both sides split into identifiers/numbers/symbols (spacing of an answer does not matter). Medium is compared with low on the cases it has.
usage (from ~/PyTaRGET): PYTHONPATH=. ~/venv_inkling/bin/python scripts/thesis_analysis/python_codebleu_llm.py"""
import re
from pathlib import Path
from data_processing import prediction_files
from data_processing.CodeBLEU.bleu import corpus_bleu
from data_processing.CodeBLEU.code_bleu import calc_code_bleu
H = Path.home()
RUNS = {"inkling low": H / "eval_inkling/python_full_low", "inkling medium": H / "eval_inkling/python_full_medium", "qwen3coder": H / "eval_results_perproject_llm/full/qwen3coder", "gpt-oss-20b low": H / "eval_results_perproject_llm/full/gptoss20b"}
D = {k: {r["ID"]: r for r in prediction_files.load(v / "test_predictions.json")} for k, v in RUNS.items() if (v / "test_predictions.json").exists()}
retok = lambda s: " ".join(re.findall(r"\w+|[^\w\s]", s))


def score(targets, preds):
    return 100 * corpus_bleu([[t.split()] for t in targets], [p.split() for p in preds]), 100 * calc_code_bleu([targets], preds, lang="python")


def table(names, ids):
    print(f"\n{len(ids)} cases\n| model | scoring | BLEU | CodeBLEU | BLEU retok | CodeBLEU retok |\n|---|---|---|---|---|---|")
    for k in names:
        for mode in ("pipeline", "first"):
            t = [D[k][i]["target"] for i in ids]
            p = [(D[k][i]["target"] if mode == "pipeline" and D[k][i]["target"] in D[k][i]["preds"] else D[k][i]["preds"][0]) for i in ids]
            a = score(t, p); b = score([retok(x) for x in t], [retok(x) for x in p])
            print(f"| {k} | {mode} | {a[0]:.2f} | {a[1]:.2f} | {b[0]:.2f} | {b[1]:.2f} |", flush=True)


full = [k for k in D if k != "inkling medium"]
table(full, sorted(set.intersection(*(set(D[k]) for k in full))))
if "inkling medium" in D:
    ids = sorted(set(D["inkling medium"]) & set(D["inkling low"]) & set(D["qwen3coder"]))
    table(list(D), ids)
