"""Encoder.split: train < validation < test in time, and no commit in two sets."""
import pandas as pd
import pytest

from data_processing.encode import Encoder


def dataset(sizes, start="2020-01-01"):
    """One row per case; commit i has sizes[i] cases and is one day newer than commit i-1 (given in scrambled order)."""
    rows = []
    for i, size in enumerate(sizes):
        for j in range(size):
            rows.append({"broken_hash": f"c{i:02d}", "broken_date": str(pd.Timestamp(start) + pd.Timedelta(days=i)),
                         "case": j})
    return pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)   # file order is not time order


def split(sizes, train_size):
    encoder = Encoder.__new__(Encoder)
    encoder.train_size = train_size
    return encoder.split(dataset(sizes))


def commits(ds):
    return set(ds["broken_hash"])


def check_separated(train, valid, test):
    assert not (commits(train) & commits(valid)) and not (commits(train) & commits(test)) and not (commits(valid) & commits(test))


class TestSplit:
    def test_whole_commits_go_to_one_set_and_time_is_ordered(self):
        train, valid, test = split([5] * 10, 0.6)
        check_separated(train, valid, test)
        assert (len(train), len(valid), len(test)) == (30, 5, 15)
        assert max(train["broken_date"]) < min(valid["broken_date"]) < min(test["broken_date"])

    def test_validation_is_about_a_quarter_of_the_held_out_cases(self):
        _, valid, test = split([4] * 20, 0.5)
        assert len(valid) + len(test) == 40
        assert len(valid) == 8 and 0.2 <= len(valid) / 40 <= 0.25

    def test_nothing_is_lost_or_duplicated(self):
        sizes = [3, 7, 2, 9, 4, 6, 1, 8]
        train, valid, test = split(sizes, 0.5)
        assert len(train) + len(valid) + len(test) == sum(sizes)
        assert sorted(pd.concat([train, valid, test])["broken_hash"]) == sorted(dataset(sizes)["broken_hash"])

    def test_a_large_oldest_held_out_commit_still_gives_a_validation_set(self):
        train, valid, test = split([4, 4, 4, 30, 2, 2], 0.2)       # train: first commit; oldest held-out commit has 4 cases
        check_separated(train, valid, test)
        assert len(valid) > 0 and len(test) > 0
        train, valid, test = split([3, 30, 2, 2, 2], 0.05)          # train empty; first commit (3) < 25% target? no: 39*.25=9
        check_separated(train, valid, test)
        assert len(valid) > 0 and len(test) > 0

    def test_a_single_held_out_commit_has_no_validation_set(self, capsys):
        train, valid, test = split([5, 5, 5], 0.67)                  # two commits do not fit, so one held-out commit
        assert len(valid) == 0 and len(test) > 0
        assert "no validation set" in capsys.readouterr().out

    def test_train_size_zero_is_an_evaluation_only_run(self):
        train, valid, test = split([5] * 4, 0)
        assert len(train) == 0 and len(valid) == 0 and len(test) == 20

    def test_the_result_does_not_depend_on_the_order_of_the_rows(self):
        a = split([5, 4, 6, 3, 7], 0.5)
        encoder = Encoder.__new__(Encoder)
        encoder.train_size = 0.5
        b = encoder.split(dataset([5, 4, 6, 3, 7]).sample(frac=1, random_state=11))
        for x, y in zip(a, b):
            assert commits(x) == commits(y)


class TestPerProjectSplit:
    """The TaRGet paper's split: inside every project the older commits train, the newer ones validate and test."""

    @staticmethod
    def make(projects):
        rows = []
        for repo, sizes in projects.items():
            for i, size in enumerate(sizes):
                for j in range(size):
                    rows.append({"repository_name": repo, "broken_hash": f"{repo}-c{i:02d}",
                                 "broken_date": str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)), "case": j})
        return pd.DataFrame(rows).sample(frac=1, random_state=5).reset_index(drop=True)

    @staticmethod
    def run(ds, train_size=0.8, valid_share=0.05):
        encoder = Encoder.__new__(Encoder)
        encoder.train_size, encoder.valid_share, encoder.split_mode = train_size, valid_share, "per_project"
        return encoder.split(ds)

    def test_every_project_is_cut_in_time_and_no_commit_is_in_two_sets(self):
        ds = self.make({"a/a": [2] * 20, "b/b": [1] * 40})
        train, valid, test = self.run(ds)
        check_separated(train, valid, test)
        for repo in ("a/a", "b/b"):
            t, v, e = (x[x["repository_name"] == repo] for x in (train, valid, test))
            assert max(t["broken_date"]) < min(e["broken_date"]) and max(t["broken_date"]) < min(v["broken_date"])
            assert max(v["broken_date"]) < min(e["broken_date"])
        assert len(train) + len(valid) + len(test) == len(ds)
        assert 0.78 <= len(train) / len(ds) <= 0.82 and 0.13 <= len(test) / len(ds) <= 0.17

    def test_a_project_with_one_commit_only_trains_so_no_test_repository_is_unseen(self):
        ds = self.make({"a/a": [2] * 20, "solo/solo": [7], "two/two": [3, 3]})
        train, valid, test = self.run(ds)
        assert set(train["repository_name"]) >= {"solo/solo", "two/two"}
        assert "solo/solo" not in set(valid["repository_name"]) | set(test["repository_name"])
        assert set(valid["repository_name"]) | set(test["repository_name"]) <= set(train["repository_name"])
