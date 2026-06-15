"""End-to-end test of RepositoryActions.find_repaired_test_cases().

This is the heart of the miner and the most likely place for "no cases found"
to originate, so it is exercised against a *real* (but tiny, local) git repo
that contains a genuine broken->repaired scenario:

    parent commit:  f() returns 1,  test asserts f() == 1   (passes)
    child  commit:  f() returns 2,  test asserts f() == 2   (passes)
    override:       child's f() (==2) + parent's test (==1)  -> FAILS

That triple is exactly what find_repaired_test_cases looks for, so it must
surface one repaired case.  The test file is self-contained (the function
under test lives in the same file) so no ``pip install`` of the repo is needed.
"""

import subprocess
from pathlib import Path

import pytest

import repository_actions


PARENT_FILE = (
    "def f():\n"
    "    return 1\n"
    "\n"
    "def test_f():\n"
    "    assert f() == 1\n"
)

CHILD_FILE = (
    "def f():\n"
    "    return 2\n"
    "\n"
    "def test_f():\n"
    "    assert f() == 2\n"
)


def _git(args, cwd, **kw):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, check=True, **kw)


@pytest.fixture
def broken_repaired_repo(git_available, tmp_path):
    """Build <tmp>/proj as a git repo with a parent->child repair scenario."""
    base = tmp_path
    work = base / "proj"
    tests_dir = work / "tests"
    tests_dir.mkdir(parents=True)

    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)

    test_file = tests_dir / "test_mod.py"

    # --- parent commit ---
    test_file.write_text(PARENT_FILE, encoding="utf-8")
    _git(["add", "-A"], cwd=work)
    _git(["commit", "-q", "-m", "parent"], cwd=work)
    parent_sha = _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()

    # --- child commit (modifies the test file) ---
    test_file.write_text(CHILD_FILE, encoding="utf-8")
    _git(["add", "-A"], cwd=work)
    _git(["commit", "-q", "-m", "child"], cwd=work)
    child_sha = _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()

    return str(base), parent_sha, child_sha


def test_find_repaired_test_cases_surfaces_the_repair(broken_repaired_repo):
    repository_path, parent_sha, child_sha = broken_repaired_repo

    a = repository_actions.RepositoryActions("local/proj", repository_path)
    a.current_hash = child_sha
    a.visited_commits = set()

    cases = a.find_repaired_test_cases()

    assert len(cases) == 1, f"expected exactly one repaired case, got {cases}"
    case = next(iter(cases))
    assert case.test_name == "test_f"
    assert case.rel_path.replace("\\", "/") == "tests/test_mod.py"
    assert case.broken == parent_sha
    assert case.repaired == child_sha


def test_no_case_when_test_unchanged(git_available, tmp_path):
    """If the test method does not change between commits, nothing is found.

    This guards the upstream gate: only *changed* test methods are candidates,
    so a source-only change must not produce a spurious repaired case.
    """
    work = tmp_path / "proj"
    tests_dir = work / "tests"
    tests_dir.mkdir(parents=True)
    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)

    test_file = tests_dir / "test_mod.py"
    # test method identical in both commits; only a comment differs
    v1 = "def test_f():\n    assert 1 == 1\n"
    v2 = "# changed comment\ndef test_f():\n    assert 1 == 1\n"

    test_file.write_text(v1, encoding="utf-8")
    _git(["add", "-A"], cwd=work)
    _git(["commit", "-q", "-m", "p"], cwd=work)

    test_file.write_text(v2, encoding="utf-8")
    _git(["add", "-A"], cwd=work)
    _git(["commit", "-q", "-m", "c"], cwd=work)
    child_sha = _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()

    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    a.current_hash = child_sha
    a.visited_commits = set()

    assert a.find_repaired_test_cases() == set()
