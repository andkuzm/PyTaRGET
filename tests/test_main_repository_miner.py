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
    crash_after = None      # raise after this many cases (and their progress reports)
    missing_commits = ()    # commits that are no longer in the cloned history
    searches = []           # (start commit, resume_after) of every search

    def clone_repository_last(self):
        self.cloned = True
        self.current_hash = "fresh-head"
        self.visited_commits = {"fresh-head"}
        return "/fake/dest"

    def is_ancestor(self, ancestor, descendant):
        return ancestor not in self.missing_commits

    def has_tests(self):
        return self.has_tests_value

    def find_repaired_test_cases(self, on_case=None, on_progress=None, resume_after=None):
        FakeRA.searches.append((self.current_hash, resume_after))
        for done, case in enumerate(self.repaired_cases, 1):
            if on_case is not None:
                on_case(case)
            if on_progress is not None:
                on_progress(case.broken)
            if self.crash_after == done:
                raise RuntimeError("killed")
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
    # the knobs are class attributes that tests assign to: start every test from the defaults
    FakeRA.has_tests_value, FakeRA.repaired_cases, FakeRA.annotate_map = True, (), {}
    FakeRA.crash_after, FakeRA.missing_commits, FakeRA.searches = None, (), []
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


# --------------------------------------------------------------------------- #
#  progress in the processed-repositories file, saving per case, resuming     #
# --------------------------------------------------------------------------- #
class TestProgressAndResume:
    def _case(self, n):
        return Broken_to_repaired(f"parent{n}", f"child{n}", f"test_{n}", "tests/test_x.py", "LOG")

    def _run(self, tmp_path, processed_text=None):
        processed = tmp_path / "processed.txt"
        if processed_text is not None:
            processed.write_text(processed_text, encoding="utf-8")
        Main("owner/proj", str(tmp_path), str(tmp_path / "out.csv"), str(processed)).process_repository()
        return processed.read_text(encoding="utf-8").splitlines()

    def _rows(self, tmp_path):
        with (tmp_path / "out.csv").open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter="|"))

    def test_finished_repository_is_marked_done_with_its_start(self, patch_ra, tmp_path):
        patch_ra.repaired_cases = (self._case(1), self._case(2))
        assert self._run(tmp_path) == ["owner/proj|fresh-head"]
        assert [r["test_name"] for r in self._rows(tmp_path)] == ["test_1", "test_2"]

    def test_repository_without_tests_is_done_too(self, patch_ra, tmp_path):
        patch_ra.has_tests_value = False
        assert self._run(tmp_path) == ["owner/proj|fresh-head"]

    def test_crash_keeps_what_was_saved_and_the_last_finished_commit(self, patch_ra, tmp_path):
        patch_ra.repaired_cases = (self._case(1), self._case(2), self._case(3))
        patch_ra.crash_after = 2
        lines = self._run(tmp_path)
        # two cases are in the dataset and the repository is in progress after the second one
        assert [r["test_name"] for r in self._rows(tmp_path)] == ["test_1", "test_2"]
        assert lines == ["owner/proj|fresh-head|parent2"]

    def test_crash_without_progress_gives_the_repository_up(self, patch_ra, tmp_path, monkeypatch):
        def boom(self):
            raise RuntimeError("search exploded")
        monkeypatch.setattr(FakeRA, "has_tests", boom)
        # the start was recorded and nothing was finished: retrying would crash the same way, so it is closed
        assert self._run(tmp_path) == ["owner/proj|fresh-head"]

    def test_clone_failure_leaves_a_plain_done_line(self, patch_ra, tmp_path, monkeypatch):
        def boom(self):
            raise RuntimeError("clone exploded")
        monkeypatch.setattr(FakeRA, "clone_repository_last", boom)
        assert self._run(tmp_path) == ["owner/proj"]

    def test_resumes_from_the_recorded_start_and_marker(self, patch_ra, tmp_path):
        patch_ra.repaired_cases = (self._case(3),)
        lines = self._run(tmp_path, "other/repo|abc\nowner/proj|pinned-head|parent2\n")
        assert patch_ra.searches == [("pinned-head", "parent2")]
        assert lines == ["other/repo|abc", "owner/proj|pinned-head"]

    def test_resume_does_not_save_a_case_twice(self, patch_ra, tmp_path):
        patch_ra.repaired_cases = (self._case(2), self._case(3))
        main = Main("owner/proj", str(tmp_path), str(tmp_path / "out.csv"))
        main.save_case("owner/proj", "OLD", "tests/test_x.py", "parent2", "child2", "LOG", test_name="test_2")
        main.save_case("someone/else", "OTHER", "tests/test_x.py", "parent3", "child3", "LOG", test_name="test_3")

        self._run(tmp_path, "owner/proj|pinned-head|parent1\n")
        rows = [(r["repository_name"], r["test_name"], r["annotated_code"]) for r in self._rows(tmp_path)]
        assert rows == [("owner/proj", "test_2", "OLD"), ("someone/else", "test_3", "OTHER"),
                        ("owner/proj", "test_3", "ANNOTATED")]

    def test_marker_missing_from_the_history_starts_over(self, patch_ra, tmp_path):
        patch_ra.missing_commits = ("stale-marker",)
        lines = self._run(tmp_path, "owner/proj|pinned-head|stale-marker\n")
        assert patch_ra.searches == [("pinned-head", None)]
        assert lines == ["owner/proj|pinned-head"]

    def test_start_missing_from_the_history_starts_over_from_the_fresh_head(self, patch_ra, tmp_path):
        patch_ra.missing_commits = ("pinned-head",)
        lines = self._run(tmp_path, "owner/proj|pinned-head|parent2\n")
        assert patch_ra.searches == [("fresh-head", None)]
        assert lines == ["owner/proj|fresh-head"]

    def test_a_case_that_fails_to_annotate_does_not_stop_the_repository(self, patch_ra, tmp_path, monkeypatch):
        patch_ra.repaired_cases = (self._case(1), self._case(2))

        def flaky(self, case):
            if case.test_name == "test_1":
                raise RuntimeError("coverage run died")
            return "ANNOTATED"
        monkeypatch.setattr(FakeRA, "extract_and_annotate_code", flaky)
        assert self._run(tmp_path) == ["owner/proj|fresh-head"]
        assert [r["test_name"] for r in self._rows(tmp_path)] == ["test_2"]
