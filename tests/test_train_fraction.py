"""The training-size experiment (paper RQ3.1): the newest commits of every project, validation and test untouched."""
import pickle

import pandas as pd
import pytest

from data_processing.encode import Encoder
from data_processing.make_trainfrac_dirs import build, read_meta
from data_processing.train_fraction import describe, select_recent_per_project


def project_frame(projects):
    """projects: {repo: [cases per commit, oldest commit first]}; the rows are shuffled, dates grow with the commit number."""
    rows = []
    for repo, sizes in projects.items():
        for i, size in enumerate(sizes):
            for j in range(size):
                rows.append({"repository_name": repo, "broken_hash": f"{repo}-c{i:02d}",
                             "broken_date": str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)), "case": j})
    return pd.DataFrame(rows).sample(frac=1, random_state=11).reset_index(drop=True)


def kept_frame(ds, fraction):
    return ds.iloc[select_recent_per_project(ds, fraction)]


class TestSelection:
    PROJECTS = {"a/a": [2] * 20, "b/b": [1] * 40, "c/c": [5, 1, 1, 8], "solo/solo": [7]}

    def test_a_fraction_of_one_keeps_everything(self):
        ds = project_frame(self.PROJECTS)
        assert select_recent_per_project(ds, 1.0) == list(range(len(ds)))

    @pytest.mark.parametrize("fraction", [0.2, 0.4, 0.6, 0.8])
    def test_the_total_share_is_close_to_the_request_and_every_project_keeps_its_newest_commits(self, fraction):
        ds = project_frame(self.PROJECTS)
        kept = kept_frame(ds, fraction)
        assert abs(len(kept) / len(ds) - fraction) <= 0.06
        for repo, group in ds.groupby("repository_name"):
            mine = kept[kept["repository_name"] == repo]
            assert len(mine) >= 1                                           # no project disappears
            # newest commits only: everything kept is newer than everything dropped, and no commit is cut in two
            dropped = group[~group.index.isin(mine.index)]
            if len(dropped):
                assert dropped["broken_date"].max() < mine["broken_date"].min()
            assert all((mine["broken_hash"] == c).sum() == (group["broken_hash"] == c).sum() for c in mine["broken_hash"].unique())

    def test_many_tiny_projects_cannot_push_the_total_above_the_request(self):
        # 60 projects of 2 commits x 2 cases: a fixed 20% per project would keep 50% (one commit each); the calibration keeps the floor
        ds = project_frame({f"p{i}/p{i}": [2, 2] for i in range(60)})
        kept = kept_frame(ds, 0.2)
        assert len(kept) == 60 * 2 and set(kept["repository_name"]) == set(ds["repository_name"])   # the floor: one commit per project
        assert abs(len(kept_frame(ds, 0.75)) / len(ds) - 0.75) <= 0.01 + 1e-9

    def test_the_subsets_are_nested(self):
        ds = project_frame(self.PROJECTS)
        sets = [set(select_recent_per_project(ds, f)) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
        assert all(small <= large for small, large in zip(sets, sets[1:]))
        assert [len(s) for s in sets] == sorted(len(s) for s in sets)

    def test_nested_and_exact_also_when_all_projects_change_at_once(self):
        ds = project_frame({f"p{i}/p{i}": [2, 2] for i in range(60)})          # identical projects: a step function of the share
        sets = [set(select_recent_per_project(ds, f)) for f in (0.55, 0.65, 0.75, 0.85)]
        assert all(small < large for small, large in zip(sets, sets[1:]))
        assert [abs(len(x) / len(ds) - f) <= 0.01 for x, f in zip(sets, (0.55, 0.65, 0.75, 0.85))] == [True] * 4

    def test_a_project_with_a_single_commit_is_kept_whatever_the_fraction(self):
        ds = project_frame({"solo/solo": [7], "a/a": [2] * 10})
        assert set(kept_frame(ds, 0.2)["repository_name"]) == {"solo/solo", "a/a"}
        assert len(kept_frame(ds, 0.2)[lambda d: d["repository_name"] == "solo/solo"]) == 7

    def test_the_closest_cut_wins_and_a_tie_keeps_more_data(self):
        ds = project_frame({"x/x": [4, 4]})              # 8 cases; 0.5 -> 4 kept (exact); 0.75 -> 4 or 8 kept, tie -> 8
        assert len(select_recent_per_project(ds, 0.5)) == 4
        assert len(select_recent_per_project(ds, 0.75)) == 8

    def test_a_request_below_the_floor_returns_the_floor(self):
        ds = project_frame({"a/a": [3, 3, 3], "b/b": [5, 5]})     # the floor is the newest commit of each: 3 + 5 cases
        assert len(select_recent_per_project(ds, 0.05)) == 8

    def test_rows_without_dates_use_the_row_order_and_bad_fractions_are_refused(self):
        ds = project_frame({"a/a": [1] * 10}).drop(columns="broken_date").sort_values("broken_hash").reset_index(drop=True)
        assert select_recent_per_project(ds, 0.5) == list(range(5, 10))
        for bad in (0, -0.1, 1.5):
            with pytest.raises(ValueError):
                select_recent_per_project(ds, bad)

    def test_describe_reports_the_shares(self):
        ds = project_frame({"a/a": [2] * 10, "b/b": [1] * 10})
        info = describe(ds, select_recent_per_project(ds, 0.5))
        assert info["cases_before"] == 30 and info["projects"] == 2 and info["projects_kept"] == 2 and 14 <= info["cases_kept"] <= 16


class TestEncoderOption:
    @staticmethod
    def run(ds, fraction, mode="per_project"):
        encoder = Encoder.__new__(Encoder)
        encoder.train_size, encoder.valid_share, encoder.split_mode, encoder.train_fraction = 0.8, 0.05, mode, fraction
        return encoder.split(ds)

    def test_validation_and_test_stay_exactly_the_same_and_training_is_a_newer_subset(self):
        ds = project_frame({"a/a": [2] * 30, "b/b": [1] * 60, "c/c": [3, 1, 2] * 8})
        full_train, full_valid, full_test = self.run(ds, 1.0)
        for fraction in (0.2, 0.5, 0.8):
            train, valid, test = self.run(ds, fraction)
            assert valid.equals(full_valid) and test.equals(full_test)
            keys = lambda d: set(zip(d["repository_name"], d["broken_hash"], d["case"]))
            assert keys(train) < keys(full_train)
            assert abs(len(train) / len(full_train) - fraction) <= 0.06
            assert set(train["repository_name"]) == set(full_train["repository_name"])

    def test_the_option_is_refused_for_the_global_split(self):
        ds = project_frame({"a/a": [2] * 30})
        with pytest.raises(ValueError):
            self.run(ds, 0.5, mode="global")


class FakeDataset:
    """Stands in for the pickled dataset: only the items matter (one per case of splits/train.json)."""
    def __init__(self, data):
        self.data = data


def test_the_experiment_directories_link_valid_and_test_and_subset_the_pickle(tmp_path):
    ds = project_frame({"a/a": [2] * 10, "b/b": [1] * 20})
    base = tmp_path / "src" / "codet5p" / "0.8"
    (base / "splits").mkdir(parents=True)
    with open(base / "splits" / "train.json", "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(__import__("json").dumps(r) for r in ds.to_dict("records")) + "\n]")
    for name in ("valid.json", "test.json"):
        (base / "splits" / name).write_text("[]")
    (base / "valid.pkl").write_bytes(b"v")
    (base / "test.pkl").write_bytes(b"t")
    pickle.dump(FakeDataset([{"i": i} for i in range(len(ds))]), open(base / "train.pkl", "wb"))
    meta = read_meta(base / "splits" / "train.json")
    assert list(meta["repository_name"]) == list(ds["repository_name"])

    report = build(tmp_path / "src", tmp_path / "out", "codet5p", [0.5, 1.0])
    half = tmp_path / "out" / "50" / "codet5p" / "0.8"
    items = pickle.load(open(half / "train.pkl", "rb")).data
    assert len(items) == report["0.5"]["cases_kept"] and 0.45 <= len(items) / len(ds) <= 0.55
    expected = select_recent_per_project(meta, 0.5)
    assert [item["i"] for item in items] == expected                   # the right cases, in the original order
    assert (half / "test.pkl").is_symlink() and (half / "test.pkl").read_bytes() == b"t"
    assert (half / "splits" / "valid.json").is_symlink()
    full = pickle.load(open(tmp_path / "out" / "100" / "codet5p" / "0.8" / "train.pkl", "rb")).data
    assert len(full) == len(ds)


def test_misaligned_pickle_and_split_file_are_refused(tmp_path):
    ds = project_frame({"a/a": [2] * 5})
    base = tmp_path / "src" / "plbart" / "0.8"
    (base / "splits").mkdir(parents=True)
    (base / "splits" / "train.json").write_text("[\n" + ",\n".join(__import__("json").dumps(r) for r in ds.to_dict("records")) + "\n]")
    pickle.dump(FakeDataset([{"i": 0}]), open(base / "train.pkl", "wb"))
    with pytest.raises(SystemExit):
        build(tmp_path / "src", tmp_path / "out", "plbart", [0.5])
