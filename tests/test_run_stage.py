"""Tests for slurm/run_stage.py (the entry point of the sbatch scripts), with a fake Eftt: no models, no GPU."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "slurm"))

import run_stage                                                              # noqa: E402
from data_processing import prediction_files                                  # noqa: E402

TINY = ROOT / "tests" / "data" / "tiny_cases.csv"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="|"))


class FakeEftt:
    calls = []

    def __init__(self, args, csv_path, out_path):
        self.args, self.csv_path, self.out_path = args, csv_path, out_path
        FakeEftt.calls.append(("init", str(csv_path)))

    def encode(self):
        FakeEftt.calls.append(("encode",))

    def train(self, resume=True, **options):
        FakeEftt.calls.append(("train", resume, options))
        if getattr(FakeEftt, "interrupt_train", False):
            raise KeyboardInterrupt

    def validate(self, **options):
        FakeEftt.calls.append(("validate", options))
        experiment = Path(self.out_path) / self.args.model / str(self.args.train_size)
        prediction_files.save([{"ID": i, "target": "t", "preds": ["p"]} for i in range(options.get("limit") or 3)],
                              experiment / "test_predictions.json")
        return 11.5, 22.5, 3.0, 4.0


@pytest.fixture(autouse=True)
def fresh_fake():
    FakeEftt.calls = []
    FakeEftt.interrupt_train = False


def go(*argv):
    return run_stage.main(list(argv), make=FakeEftt)


class TestArguments:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("CSV", raising=False)
        monkeypatch.delenv("OUT", raising=False)
        args = run_stage.parse_args(["train", "--model", "codet5p"])
        assert (args.train_size, args.beam, args.epochs, args.batch_size, args.early_stop) == (0.8, 20, 10, 8, 3)
        assert args.csv.endswith("annotated_cases_v2_pr59_filtered_cap_clean.csv") and args.restart is False

    def test_environment_sets_the_paths(self, monkeypatch):
        monkeypatch.setenv("CSV", "/data/cases.csv")
        monkeypatch.setenv("OUT", "/data/results")
        args = run_stage.parse_args(["encode", "--model", "plbart"])
        assert (args.csv, args.out) == ("/data/cases.csv", "/data/results")

    def test_unknown_model_or_stage_is_refused(self):
        with pytest.raises(SystemExit):
            run_stage.parse_args(["train", "--model", "gpt-17"])
        with pytest.raises(SystemExit):
            run_stage.parse_args(["deploy", "--model", "codet5p"])


class TestStages:
    def test_encode(self, tmp_path):
        assert go("encode", "--model", "codet5p", "--csv", "x.csv", "--out", str(tmp_path)) == 0
        assert [c[0] for c in FakeEftt.calls] == ["init", "encode"]

    def test_train_passes_the_options_and_resumes_by_default(self, tmp_path):
        go("train", "--model", "plbart", "--out", str(tmp_path), "--epochs", "3", "--batch-size", "4", "--learning-rate", "1e-4")
        assert FakeEftt.calls[-1] == ("train", True, {"epochs": 3, "batch_size": 4, "eval_batch_size": 8,
                                                       "early_stop": 3, "learning_rate": 1e-4})

    def test_restart_starts_training_over(self, tmp_path):
        go("train", "--model", "codegen", "--out", str(tmp_path), "--restart")
        assert FakeEftt.calls[-1][1] is False

    def test_an_interrupted_stage_exits_with_75_so_that_it_can_be_run_again(self, tmp_path, capsys):
        FakeEftt.interrupt_train = True
        assert go("train", "--model", "codet5p", "--out", str(tmp_path)) == run_stage.EXIT_INTERRUPTED == 75
        assert "run the same command again" in capsys.readouterr().out

    def test_test_writes_the_metrics_file(self, tmp_path):
        assert go("test", "--model", "codet5p", "--out", str(tmp_path), "--beam", "5") == 0
        metrics = json.loads((tmp_path / "codet5p" / "0.8" / "metrics.json").read_text())
        assert metrics == {"model": "codet5p", "train_size": 0.8, "beam_size": 5, "limit": None, "cases": 3,
                           "BLEU": 11.5, "CodeBLEU": 22.5, "EM": 3.0, "EM_indentation_normalized": 4.0}
        assert FakeEftt.calls[-1] == ("validate", {"save_every": 10, "resume": True, "limit": None})

    def test_test_restart_and_limit(self, tmp_path):
        go("test", "--model", "codet5p", "--out", str(tmp_path), "--restart", "--limit", "7", "--save-every", "2")
        assert FakeEftt.calls[-1] == ("validate", {"save_every": 2, "resume": False, "limit": 7})


class TestSmoke:
    def test_runs_everything_small_and_from_scratch(self, tmp_path):
        smoke_csv = tmp_path / "smoke.csv"
        smoke_csv.write_text("header\n")
        assert go("smoke", "--model", "codegen", "--out", str(tmp_path / "out"), "--smoke-csv", str(smoke_csv)) == 0
        kinds = [c[0] for c in FakeEftt.calls]
        assert kinds == ["init", "encode", "train", "validate"]
        assert FakeEftt.calls[0][1] == str(smoke_csv)                       # the sample, not the full file
        assert FakeEftt.calls[2][1] is False and FakeEftt.calls[2][2]["epochs"] == 1
        assert FakeEftt.calls[3][1]["limit"] == 20 and FakeEftt.calls[3][1]["resume"] is False

    def test_builds_the_sample_when_it_does_not_exist(self, tmp_path):
        smoke_csv = tmp_path / "smoke.csv"
        go("smoke", "--model", "codet5p", "--out", str(tmp_path / "out"), "--csv", str(TINY), "--smoke-csv", str(smoke_csv),
           "--smoke-commits", "3", "--smoke-per-commit", "2")
        assert len(read(smoke_csv)) - 1 == 6


class TestSmokeSample:
    def test_whole_commits_with_a_cap_per_commit(self, tmp_path):
        destination = tmp_path / "smoke.csv"
        rows, commits = run_stage.make_smoke_csv(TINY, destination, commits=4, per_commit=3, seed=1)
        out = read(destination)
        assert (rows, commits) == (12, 4) and len(out) == 13
        assert out[0] == read(TINY)[0]
        per_commit = {}
        for row in out[1:]:
            per_commit.setdefault((row[0], row[3]), []).append(row)
        assert len(per_commit) == 4 and all(len(v) == 3 for v in per_commit.values())

    def test_deterministic_for_a_seed_and_different_for_another(self, tmp_path):
        a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
        run_stage.make_smoke_csv(TINY, a, commits=3, per_commit=2, seed=5)
        run_stage.make_smoke_csv(TINY, b, commits=3, per_commit=2, seed=5)
        run_stage.make_smoke_csv(TINY, c, commits=3, per_commit=2, seed=6)
        assert a.read_bytes() == b.read_bytes() and a.read_bytes() != c.read_bytes()

    def test_asking_for_more_commits_than_exist_takes_them_all(self, tmp_path):
        rows, commits = run_stage.make_smoke_csv(TINY, tmp_path / "s.csv", commits=100, per_commit=100)
        assert (rows, commits) == (28, 7)

    def test_commits_with_a_single_case_are_not_sampled(self, tmp_path):
        source = tmp_path / "in.csv"
        lines = read(TINY)
        header, body = lines[0], lines[1:]
        single = body[0][:]
        single[3] = "unique-commit"
        with open(source, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_ALL)
            writer.writerows([header] + body + [single])
        run_stage.make_smoke_csv(source, tmp_path / "s.csv", commits=100, per_commit=100)
        assert all(row[3] != "unique-commit" for row in read(tmp_path / "s.csv")[1:])

    def test_the_source_file_is_not_changed(self, tmp_path):
        before = TINY.read_bytes()
        run_stage.make_smoke_csv(TINY, tmp_path / "s.csv", commits=2, per_commit=2)
        assert TINY.read_bytes() == before
