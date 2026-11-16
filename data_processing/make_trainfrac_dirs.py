"""
Build the experiment directories of the training-size study (the paper's RQ3.1) from ALREADY ENCODED per-project results, so
nothing is tokenised again: for every fraction the training set keeps the newest commits of every project (train_fraction.py),
validation and test are exactly those of the full run (links), and the train/test stages can run on the new directory unchanged:

  python -m data_processing.make_trainfrac_dirs --src ~/eval_results_perproject --out ~/eval_results_perproject_trainfrac
  # then, per fraction (ask the user before submitting):  OUT=<out>/<20|40|60|80> slurm/submit_pipeline.sh --from train <model>
  #   (the experiment folder keeps the name <model>/0.8/ so run_stage.py works with --train-size 0.8 and SPLIT_MODE=per_project)

Reads <src>/<model>/0.8/splits/train.json (one case per line, only three columns are used) and train.pkl; writes
<out>/<pct>/<model>/0.8/train.pkl (atomic), links valid.pkl, test.pkl, splits/valid.json, splits/test.json, and trainfrac.json.
"""
import argparse
import copy
import json
import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd

from data_processing.train_fraction import describe, select_recent_per_project

KEEP_COLUMNS = ("repository_name", "broken_hash", "broken_date")


def read_meta(train_json):
    """The three columns of every case of splits/train.json (one case per line inside [ ... ]), in file order."""
    rows = []
    with open(train_json, encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line in ("", "[", "]"):
                continue
            case = json.loads(line)
            rows.append({k: case.get(k) for k in KEEP_COLUMNS})
    return pd.DataFrame(rows)


def link(target, name):
    name = Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(Path(target).resolve())


def build(src, out, model, fractions, train_size="0.8"):
    base = Path(src) / model / train_size
    meta = read_meta(base / "splits" / "train.json")
    with open(base / "train.pkl", "rb") as f:
        dataset = pickle.load(f)
    if len(dataset.data) != len(meta):
        raise SystemExit(f"{model}: train.pkl has {len(dataset.data)} items but splits/train.json has {len(meta)} cases; "
                         "they must be aligned (re-encode or check the run)")
    report = {}
    for fraction in fractions:
        kept = select_recent_per_project(meta, fraction)
        sub = copy.copy(dataset)
        sub.data = [dataset.data[i] for i in kept]
        target = Path(out) / str(int(round(fraction * 100))) / model / train_size
        target.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=target, suffix=".tmp")
        with os.fdopen(handle, "wb") as f:
            pickle.dump(sub, f)
        os.replace(tmp, target / "train.pkl")
        for name in ("valid.pkl", "test.pkl"):
            link(base / name, target / name)
        for name in ("valid.json", "test.json"):
            link(base / "splits" / name, target / "splits" / name)
        info = describe(meta, kept) | {"fraction": fraction, "kept_positions_first_last": [kept[0], kept[-1]] if kept else None}
        (target / "trainfrac.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        report[str(fraction)] = info
        print(f"{model} {fraction:.0%}: kept {info['cases_kept']} of {info['cases_before']} training cases "
              f"({info['share_kept']:.1%}), {info['projects_kept']}/{info['projects']} projects, "
              f"{info['commits_kept']}/{info['commits_before']} commits -> {target}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.expanduser("~/eval_results_perproject"))
    ap.add_argument("--out", default=os.path.expanduser("~/eval_results_perproject_trainfrac"))
    ap.add_argument("--models", nargs="+", default=["codet5p", "plbart", "codegen"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--train-size", default="0.8", help="name of the experiment folder of the full run")
    args = ap.parse_args()
    for model in args.models:
        build(args.src, args.out, model, args.fractions, args.train_size)


if __name__ == "__main__":
    main()
