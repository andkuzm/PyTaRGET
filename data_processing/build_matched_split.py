"""
Build the "matched" test set: the cases that codet5p, plbart and codegen all encoded (identified by repository, path,
test name and both commit hashes), taken with the CodeT5+ input (1024-token context). Writes
<out>/splits/test.json (the rows) and <out>/matched_keys.json (the keys, to filter the fine-tuned models' predictions).
  python -m data_processing.build_matched_split --results ~/eval_results_perproject --out ~/eval_results_perproject_llm/matched
"""
import argparse
import json
from pathlib import Path

MODELS = ("codet5p", "plbart", "codegen")


def key(r):
    return [r["repository_name"], r["relative_path"], r["test_name"], r["broken_hash"], r["repaired_hash"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keys, rows = {}, None
    for m in MODELS:
        data = json.load(open(Path(args.results) / m / "0.8" / "splits" / "test.json"))
        keys[m] = [tuple(key(r)) for r in data]
        print(m, len(data), "unique keys", len(set(keys[m])), flush=True)
        if m == "codet5p":
            rows = data
        else:
            del data
    inter = set(keys["codet5p"]) & set(keys["plbart"]) & set(keys["codegen"])
    out = [r for r, k in zip(rows, keys["codet5p"]) if k in inter]
    d = Path(args.out)
    (d / "splits").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "splits" / "test.json", "w"))
    json.dump({m: [list(k) for k in keys[m]] for m in MODELS} | {"matched": [key(r) for r in out]},
              open(d / "matched_keys.json", "w"))
    print("matched cases:", len(out), flush=True)


if __name__ == "__main__":
    main()
