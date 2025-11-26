import os
import json
import subprocess
import pandas as pd
from git import Repo   # GitPython recommended

def run_plausibility_test(cases_csv, predictions_json, workdir):
    df = pd.read_csv(cases_csv)
    with open(predictions_json) as f:
        predictions = json.load(f)["preds"]

    results = {}

    for idx, row in df.iterrows():
        repo_name = row["repository_name"]
        repaired_hash = row["repaired_hash"]
        rel_path = row["relative_path"]

        # === 1. Clone repo (if not exists) ===
        repo_dir = os.path.join(workdir, repo_name.replace("/", "_"))
        if not os.path.exists(repo_dir):
            Repo.clone_from(f"https://github.com/{repo_name}.git", repo_dir)

        repo = Repo(repo_dir)
        repo.git.checkout(repaired_hash, force=True)

        # === 2. Apply prediction tests ===
        preds = predictions.get(str(idx), [])
        sample_results = []

        for p_id, test_code in enumerate(preds):
            pred_fname = f"test_pred_{idx}_{p_id}.py"
            pred_path = os.path.join(repo_dir, os.path.dirname(rel_path), pred_fname)
            with open(pred_path, "w", encoding="utf-8") as f:
                f.write(test_code)

            # === 3. Run tests ===
            try:
                # pytest inside repo, timeout recommended
                subprocess.check_call(
                    ["pytest", "-q"],
                    cwd=repo_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60
                )
                sample_results.append(1)
            except:
                sample_results.append(0)

        results[idx] = sample_results

    return results