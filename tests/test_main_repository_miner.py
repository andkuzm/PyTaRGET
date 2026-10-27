"""Tests for repository_search/main_repository_miner.py.

The CSV writer (`save_case`) is tested for real; the orchestration in
`process_repository` is tested by injecting a fake RepositoryActions so we
exercise the control flow (has-tests gate, annotation skip, save) without git
or the network.
"""

import csv

import pytest

import main_repository_miner
from main_repository_miner import Main
from data_types.Broken_to_repaired import Broken_to_repaired


# --------------------------------------------------------------------------- #
#  save_case                                                                  #
# --------------------------------------------------------------------------- #
class TestSaveCase:
    def _read(self, path):
        with path.open(encoding="utf-8") as fh:
            return list(csv.reader(fh, delimiter="|"))

    def test_writes_header_then_row(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        m = Main("owner/proj", str(tmp_path), str(csv_path))
        m.save_case("owner/proj", "ANNOT", "tests/test_a.py", "br0ken", "f1xed", "LOG")

        rows = self._read(csv_path)
        assert rows[0] == ["repository_name", "annotated_code", "relative_path",
                           "broken_hash", "repaired_hash", "outdated_test_log",
                           "test_name", "broken_date", "repaired_date"]
        assert rows[1] == ["owner/proj", "ANNOT", "tests/test_a.py",
                           "br0ken", "f1xed", "LOG", "", "", ""]

    def test_writes_test_name_and_commit_dates(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        m = Main("owner/proj", str(tmp_path), str(csv_path))
        m.save_case("owner/proj", "ANNOT", "tests/test_a.py", "br0ken", "f1xed", "LOG",
                    test_name="TestA.test_x", broken_date="2020-01-01T10:00:00+00:00",
                    repaired_date="2020-02-01T10:00:00+00:00")

        rows = self._read(csv_path)
        assert rows[0] == main_repository_miner.CSV_HEADER
        assert rows[1][6:] == ["TestA.test_x", "2020-01-01T10:00:00+00:00", "2020-02-01T10:00:00+00:00"]

    def test_appends_without_duplicate_header(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        m = Main("owner/proj", str(tmp_path), str(csv_path))
        m.save_case("owner/proj", "A1", "p1", "b1", "r1", "l1")
        m.save_case("owner/proj", "A2", "p2", "b2", "r2", "l2")

        rows = self._read(csv_path)
        assert len(rows) == 3  # 1 header + 2 data rows
        assert rows[1][1] == "A1"
        assert rows[2][1] == "A2"

    def test_preserves_embedded_pipes_and_newlines(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        m = Main("owner/proj", str(tmp_path), str(csv_path))
        tricky = "line1|with pipe\nline2"
        m.save_case("owner/proj", tricky, "p", "b", "r", "l")

        rows = self._read(csv_path)
        assert rows[1][1] == tricky


# --------------------------------------------------------------------------- #
#  process_repository orchestration                                           #
# --------------------------------------------------------------------------- #
class FakeRA:
    """Configurable stand-in for RepositoryActions."""

    instances = []

    def __init__(self, repository_name, repository_path):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.cloned = False
        FakeRA.instances.append(self)

    # behaviour knobs (class-level defaults, overridden per test)
    has_tests_value = True
    repaired_cases = ()
    annotate_map = {}

    def clone_repository_last(self):
        self.cloned = True
        return "/fake/dest"

    def has_tests(self):
        return self.has_tests_value

    def find_repaired_test_cases(self):
        return set(self.repaired_cases)

    def extract_and_annotate_code(self, case):
        return self.annotate_map.get(case.test_name, "ANNOTATED")

    def get_commit_date(self, commit_hash):
        return f"date-of-{commit_hash}"

    def funnel_rows(self):
        return [{"repository_name": self.repository_name, "bucket": 9, "stage": "", "commits": 5, "cases": 1,
                 "details": '{"parent_failure": 2}'}]


@pytest.fixture
def patch_ra(monkeypatch):
    FakeRA.instances = []
    monkeypatch.setattr(main_repository_miner.repository_actions, "RepositoryActions", FakeRA)
    # Neutralise the pip-uninstall subprocess call.
    monkeypatch.setattr(main_repository_miner.subprocess, "run",
                        lambda *a, **k: None)
    return FakeRA


class TestProcessRepository:
    def _case(self, name="TestC.test_x"):
        return Broken_to_repaired("br0ken", "f1xed", name, "tests/test_x.py", "LOG")

    def test_saves_found_case(self, patch_ra, tmp_path):
        case = self._case()
        patch_ra.has_tests_value = True
        patch_ra.repaired_cases = (case,)
        patch_ra.annotate_map = {case.test_name: "GOOD-ANNOTATION"}

        csv_path = tmp_path / "out.csv"
        Main("owner/proj", str(tmp_path), str(csv_path)).process_repository()

        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh, delimiter="|"))
        assert rows[1][1] == "GOOD-ANNOTATION"
        assert rows[1][3] == "br0ken"
        assert rows[1][4] == "f1xed"
        # test name and commit dates are recorded next to the case
        assert rows[1][6:] == [case.test_name, "date-of-br0ken", "date-of-f1xed"]

    def _funnel(self, tmp_path):
        with (tmp_path / "out_funnel.csv").open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter="|"))

    def test_writes_started_then_the_funnel_and_the_stage(self, patch_ra, tmp_path):
        patch_ra.has_tests_value = True
        patch_ra.repaired_cases = ()

        Main("owner/proj", str(tmp_path), str(tmp_path / "out.csv")).process_repository()

        rows = self._funnel(tmp_path)
        assert [(r["bucket"], r["stage"]) for r in rows] == [("repo", "started"), ("9", ""), ("repo", "searched")]
        assert rows[1]["commits"] == "5" and rows[1]["cases"] == "1" and rows[1]["details"] == '{"parent_failure": 2}'
        assert set(rows[0]) == set(main_repository_miner.FUNNEL_COLUMNS)

    def test_funnel_records_why_a_repository_has_no_search(self, patch_ra, tmp_path):
        patch_ra.has_tests_value = False
        Main("owner/proj", str(tmp_path), str(tmp_path / "out.csv")).process_repository()
        assert [r["stage"] for r in self._funnel(tmp_path)] == ["started", "no_tests"]

    def test_skips_when_no_tests(self, patch_ra, tmp_path):
        patch_ra.has_tests_value = False
        patch_ra.repaired_cases = (self._case(),)

        csv_path = tmp_path / "out.csv"
        Main("owner/proj", str(tmp_path), str(csv_path)).process_repository()
        assert not csv_path.exists()  # save_case never called -> file never created

    def test_skips_error_annotation(self, patch_ra, tmp_path):
        case = self._case()
        patch_ra.has_tests_value = True
        patch_ra.repaired_cases = (case,)
        patch_ra.annotate_map = {case.test_name: "Error"}

        csv_path = tmp_path / "out.csv"
        Main("owner/proj", str(tmp_path), str(csv_path)).process_repository()
        assert not csv_path.exists()

    def test_clone_failure_is_swallowed(self, patch_ra, tmp_path, monkeypatch):
        def boom(self):
            raise RuntimeError("clone exploded")
        monkeypatch.setattr(FakeRA, "clone_repository_last", boom)

        csv_path = tmp_path / "out.csv"
        # Should not raise: process_repository wraps everything in try/except.
        Main("owner/proj", str(tmp_path), str(csv_path)).process_repository()
        assert not csv_path.exists()
        with (tmp_path / "out_funnel.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="|"))
        assert [r["stage"] for r in rows] == ["started", "failed"]
        assert "clone exploded" in rows[1]["details"]
