"""
Saving and loading of the prediction files written by Tester / Tester_llm (`test_predictions.json`,
`<model>_llm_test_predictions.json`): one record per test case, {"ID": position in test.json, "target": ..., "preds": [...]}.

The format on disk is the one DataFrame.to_json produces (the consumers read it), but
  - it is written atomically, so a killed job never leaves a truncated file;
  - it is read without pandas' type guessing, which turns numeric looking strings into numbers;
  - an unreadable file is an error, never silently replaced by an empty result.
"""
import json
from pathlib import Path

import pandas as pd

from data_processing.atomic_io import atomic_dataframe_to_json


def save(predictions, path):
    """Write the records (any order) to `path`, sorted by ID."""
    frame = pd.DataFrame(sorted(predictions, key=lambda record: int(record["ID"])),
                         columns=["ID", "target", "preds"])
    atomic_dataframe_to_json(frame, path, indent=2)


def load(path):
    """The records of a saved predictions file: [{"ID": int, "target": str, "preds": [str]}, ...]."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Cannot read the saved predictions {path}: {e}. The file is damaged (it was probably "
                           f"written by a version that did not write atomically). Fix or delete it to start over; "
                           f"it was not touched.") from e
    if isinstance(data, list):
        records = data
    else:                                       # column oriented: {"ID": {"0": 0, ...}, "target": {...}, "preds": {...}}
        records = [{"ID": data["ID"][key], "target": data["target"][key], "preds": data["preds"][key]}
                   for key in data.get("ID", {})]
    return [{"ID": int(r["ID"]), "target": r["target"], "preds": list(r["preds"])} for r in records]


def check_matches_test_split(records, test_rows):
    """Refuse to continue from predictions that belong to a different test split."""
    for record in records:
        i = record["ID"]
        if not 0 <= i < len(test_rows) or record["target"] != test_rows[i]["output"]:
            raise ValueError(
                f"The saved predictions do not belong to this test split (case {i} differs or is out of range). "
                f"The data was encoded again after they were made; delete the predictions file to start over.")


def describe(done, total):
    """'over 12/12 cases' and a warning text when the results are partial (None when complete)."""
    text = f"over {done}/{total} cases"
    warning = None
    if done < total:
        warning = (f"WARNING: only {done} of {total} test cases have a prediction, so the scores are over "
                   f"an incomplete run and are not comparable with other results.")
    return text, warning
