"""Step 0 of the delegation plan: one bundle file per Python case for the strong sub-agent (C-strong). A bundle has the failing test WITHOUT the oracle breakage markers, the failure output of the old test,
and ALL changed source hunks of the commit (the automatic prioritisation is not applied), so that the strong agent has to pick the context itself. No reference repair is in a bundle.
usage: PYTHONPATH=. python scripts/delegation/make_step0_bundles.py <n first cases of ids_python61 | all> [--out dir]"""
import json, re, sys
from pathlib import Path
H = Path.home()
n = sys.argv[1]
ids = json.load(open(H / "eval_paid/ids_python61.json"))
ids = ids if n == "all" else ids[:int(n)]
rows = json.load(open(H / "eval_results_perproject_llm/contrast300/splits/test.json"))
ctx = json.load(open(H / "eval_ablation/context.json"))
from data_processing.context_ablation import case_key
out = H / "delegation_exp/step0/bundles"
out.mkdir(parents=True, exist_ok=True)
for i in ids:
    r = rows[i]
    m = re.search(r"\[<TESTCONTEXT>](.*?)\[<REPAIRCONTEXT>]", r["input"], re.DOTALL)
    test = re.sub(r"\[</?BREAKAGE>]", "", m.group(1)).strip() if m else ""
    hunks = [h["annotated_doc"].replace("[<HUNK>]", "").replace("[</HUNK>]", "").strip() for h in r["prioritized_changes"]]
    bundle = {"case": i, "repository": r["repository_name"], "test_file": r["relative_path"], "failing_test_code": test,
              "failure_output_of_the_old_test": ctx[case_key(r)]["failure"], "changed_source_hunks": [{"id": k, "text": t[:6000]} for k, t in enumerate(hunks)]}
    (out / f"case_{i}.json").write_text(json.dumps(bundle, indent=1))
    print(i, r["repository_name"], "hunks", len(hunks), "bytes", len(json.dumps(bundle)))
