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
