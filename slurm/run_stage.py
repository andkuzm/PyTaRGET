"""
One stage of the evaluation pipeline for one fine-tuned model; the sbatch scripts next to this file call it.

    python slurm/run_stage.py encode --model codet5p     # tokenize, split by commit, write train/valid/test
    python slurm/run_stage.py train  --model codet5p     # fine-tune (continues from checkpoint-last if there is one)
    python slurm/run_stage.py test   --model codet5p     # predictions + scores (continues from saved predictions)
    python slurm/run_stage.py smoke  --model codet5p     # all three on a small sample, 1 epoch, 20 test cases

Every stage can be started again after an interruption and continues. When a stage is interrupted (SIGTERM from Slurm
shortly before the time limit, Ctrl-C) it saves what it has and exits with status 75: start the same stage again.
Defaults come from the environment: CSV (the filtered cases), OUT (results directory).
"""
import argparse
import csv
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.atomic_io import atomic_path, atomic_write_json      # noqa: E402

csv.field_size_limit(sys.maxsize)
MODELS = ("codet5p", "plbart", "codegen")
EXIT_INTERRUPTED = 75           # EX_TEMPFAIL: not finished, run the same command again

DEFAULT_CSV = os.path.expanduser("~/annotated_cases_v2_pr59_filtered_cap_clean.csv")
DEFAULT_OUT = os.path.expanduser("~/eval_results")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["encode", "train", "test", "smoke"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--csv", default=os.environ.get("CSV", DEFAULT_CSV), help="annotated cases (default $CSV)")
    parser.add_argument("--out", default=os.environ.get("OUT", DEFAULT_OUT), help="results directory (default $OUT)")
    parser.add_argument("--train-size", type=float, default=0.8, help="share of the cases (oldest commits) for training")
    parser.add_argument("--beam", type=int, default=20, help="beam size = number of candidates per case")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--early-stop", type=int, default=3, help="epochs without improvement before stopping")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=10, help="test: save the predictions every N cases")
    parser.add_argument("--limit", type=int, default=None, help="test: only the first N test cases")
    parser.add_argument("--restart", action="store_true", help="train/test: ignore saved state and start over")
    parser.add_argument("--smoke-csv", default=os.path.expanduser("~/annotated_cases_v2_pr59_smoke.csv"))
    parser.add_argument("--smoke-commits", type=int, default=40)
    parser.add_argument("--smoke-per-commit", type=int, default=6)
    return parser.parse_args(argv)


def make_smoke_csv(source, destination, commits=40, per_commit=6, seed=1):
    """A small sample of whole commits (at most `per_commit` cases each), so the split by commit still works."""
    sizes = Counter()
    with open(source, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        for row in reader:
            sizes[(row[index["repository_name"]], row[index["broken_hash"]])] += 1
    eligible = sorted(key for key, n in sizes.items() if n >= 2)
    chosen = set(random.Random(seed).sample(eligible, min(commits, len(eligible))))
    taken = Counter()
    with open(source, newline="", encoding="utf-8") as f, atomic_path(destination) as temporary:
        reader = csv.reader(f, delimiter="|")
        next(reader)
        with open(temporary, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out, delimiter="|", quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            for row in reader:
                key = (row[index["repository_name"]], row[index["broken_hash"]])
                if key in chosen and taken[key] < per_commit:
                    taken[key] += 1
                    writer.writerow(row)
    return sum(taken.values()), len(chosen)


def make_eftt(args, csv_path, out_path):
    from data_processing.encode_tune_test import Eftt
    return Eftt(annotated_cases_path=Path(csv_path), out_path=Path(out_path), model=args.model,
                train_size=args.train_size, beam_size=args.beam, batch_size=args.batch_size, hftoken=None)


def peak_memory():
    text = f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
    try:
        import torch
        if torch.cuda.is_available():
            text += f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    except ImportError:
        pass
    return text


def train_options(args):
    return {"epochs": args.epochs, "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
            "early_stop": args.early_stop, "learning_rate": args.learning_rate}


def run_test(args, eftt, limit, restart, experiment):
    scores = eftt.validate(save_every=args.save_every, resume=not restart, limit=limit)
    bleu, codebleu, em, norm_em = scores
    predictions = json.loads((experiment / "test_predictions.json").read_text(encoding="utf-8"))
    metrics = {"model": args.model, "train_size": args.train_size, "beam_size": args.beam, "limit": limit,
               "cases": len(predictions["ID"]), "BLEU": bleu, "CodeBLEU": codebleu, "EM": em,
               "EM_indentation_normalized": norm_em}
    atomic_write_json(experiment / "metrics.json", metrics, indent=2)
    print(f"metrics: {metrics}")
    return metrics


def run(args, make=make_eftt):
    experiment = Path(args.out) / args.model / str(args.train_size)
    started = time.time()

    def done(what):
        print(f"[{args.model}] {what} finished in {time.time() - started:.0f} s, {peak_memory()}", flush=True)

    if args.stage == "encode":
        make(args, args.csv, args.out).encode()
        done("encode")
    elif args.stage == "train":
        make(args, args.csv, args.out).train(resume=not args.restart, **train_options(args))
        done("train")
    elif args.stage == "test":
        run_test(args, make(args, args.csv, args.out), args.limit, args.restart, experiment)
        done("test")
    else:                                                   # smoke: everything on a small sample, in a separate directory
        rows, commits = make_smoke_csv(args.csv, args.smoke_csv, args.smoke_commits, args.smoke_per_commit) \
            if not Path(args.smoke_csv).exists() else (None, None)
        if rows is not None:
            print(f"smoke sample: {rows} cases from {commits} commits -> {args.smoke_csv}")
        eftt = make(args, args.smoke_csv, args.out)
        eftt.encode()
        done("smoke encode")
        smoke_args = argparse.Namespace(**{**vars(args), "epochs": 1})
        eftt.train(resume=False, **train_options(smoke_args))
        done("smoke train (1 epoch)")
        run_test(args, eftt, args.limit or 20, True, experiment)
        done("smoke test")


def main(argv=None, make=make_eftt):
    args = parse_args(argv)
    try:
        run(args, make)
    except KeyboardInterrupt:
        print(f"[{args.model}] interrupted after saving its state: run the same command again to continue", flush=True)
        return EXIT_INTERRUPTED
    return 0


if __name__ == "__main__":
    sys.exit(main())
