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
                           "broken_hash", "repaired_hash", "outdated_test_log"]
        assert rows[1] == ["owner/proj", "ANNOT", "tests/test_a.py",
                           "br0ken", "f1xed", "LOG"]

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
