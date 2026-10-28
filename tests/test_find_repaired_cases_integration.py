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


CLASS_PARENT_FILE = (
    "import unittest\n"
    "\n"
    "def f():\n"
    "    return 1\n"
    "\n"
    "class TestF(unittest.TestCase):\n"
    "    def test_f(self):\n"
    "        self.assertEqual(f(), 1)\n"
)

CLASS_CHILD_FILE = (
    "import unittest\n"
    "\n"
    "def f():\n"
    "    return 2\n"
    "\n"
    "class TestF(unittest.TestCase):\n"
    "    def test_f(self):\n"
    "        self.assertEqual(f(), 2)\n"
)


def test_find_repaired_test_cases_surfaces_a_class_based_repair(git_available, tmp_path):
    """Same scenario with a unittest class: the old method pasted into the class used to be an IndentationError,
    which counted as an unknown verdict, so no case was ever found for tests inside classes."""
    work = tmp_path / "proj"
    (work / "tests").mkdir(parents=True)
    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)
    test_file = work / "tests" / "test_mod.py"
    for content in (CLASS_PARENT_FILE, CLASS_CHILD_FILE):
        test_file.write_text(content, encoding="utf-8")
        _git(["add", "-A"], cwd=work)
        _git(["commit", "-q", "-m", "c"], cwd=work)
    child_sha = _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()

    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    a.current_hash = child_sha
    a.visited_commits = set()

    cases = a.find_repaired_test_cases()

    assert [case.test_name for case in cases] == ["TestF.test_f"]
    assert test_file.read_text(encoding="utf-8") == CLASS_CHILD_FILE


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


def _totals(rows):
    """Sum of every counter over all history buckets, details included."""
    import json
    totals = {}
    for row in rows:
        for key, value in row.items():
            if key not in ("repository_name", "bucket", "stage", "details"):
                totals[key] = totals.get(key, 0) + value
        for key, value in json.loads(row["details"] or "{}").items():
            totals[key] = totals.get(key, 0) + value
    return totals


def test_funnel_counts_every_step_of_a_found_case(broken_repaired_repo):
    repository_path, parent_sha, child_sha = broken_repaired_repo
    a = repository_actions.RepositoryActions("local/proj", repository_path)
    a.current_hash = child_sha
    a.visited_commits = set()
    assert len(a.find_repaired_test_cases()) == 1

    rows = a.funnel_rows()
    totals = _totals(rows)
    assert totals["commits"] == 2 and totals["relevant_commits"] == 1
    assert totals["visited_pairs"] == 1 and totals["relevant_pairs"] == 1
    assert totals["pairs_with_changed_test"] == 1 and totals["changed_tests"] == 1
    assert (totals["parent_success"], totals["child_success"], totals["cases"]) == (1, 1, 1)
    # bucket = index * 10 // number of commits: the oldest commit is in bucket 0, the second of two in bucket 5
    assert [row["bucket"] for row in rows] == [0, 5]
    assert rows[-1]["cases"] == 1 and rows[0]["cases"] == 0


def test_funnel_says_where_a_pair_is_lost(git_available, tmp_path):
    # the test already fails at the parent commit, so the pair must be counted as lost at "parent must pass"
    work = tmp_path / "proj"
    (work / "tests").mkdir(parents=True)
    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)
    test_file = work / "tests" / "test_mod.py"
    test_file.write_text("def test_f():\n    assert 1 == 2\n", encoding="utf-8")
    _git(["add", "-A"], cwd=work)
    _git(["commit", "-q", "-m", "parent"], cwd=work)
    test_file.write_text("def test_f():\n    assert 2 == 2\n", encoding="utf-8")
    _git(["add", "-A"], cwd=work)
    _git(["commit", "-q", "-m", "child"], cwd=work)

    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    a.current_hash = _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()
    a.visited_commits = set()
    assert a.find_repaired_test_cases() == set()

    totals = _totals(a.funnel_rows())
    assert totals["changed_tests"] == 1 and totals["parent_success"] == 0 and totals["cases"] == 0
    assert totals["parent_failure"] == 1


def test_funnel_is_the_same_on_a_rerun_of_the_same_repository(broken_repaired_repo):
    repository_path, parent_sha, child_sha = broken_repaired_repo
    results = []
    for _ in range(2):
        a = repository_actions.RepositoryActions("local/proj", repository_path)
        a.current_hash = child_sha
        a.visited_commits = set()
        a.find_repaired_test_cases()
        results.append(a.funnel_rows())
    assert results[0] == results[1]


@pytest.fixture
def three_repairs_repo(git_available, tmp_path):
    """Four commits; each of the last three changes f() and its test, so three cases can be found."""
    work = tmp_path / "proj"
    (work / "tests").mkdir(parents=True)
    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)
    test_file = work / "tests" / "test_mod.py"
    shas = []
    for value in (1, 2, 3, 4):
        test_file.write_text(f"def f():\n    return {value}\n\ndef test_f():\n    assert f() == {value}\n",
                             encoding="utf-8")
        _git(["add", "-A"], cwd=work)
        _git(["commit", "-q", "-m", f"v{value}"], cwd=work)
        shas.append(_git(["rev-parse", "HEAD"], cwd=work).stdout.strip())
    return str(tmp_path), shas


def _search(repository_path, head, **kwargs):
    a = repository_actions.RepositoryActions("local/proj", repository_path)
    a.current_hash = head
    a.visited_commits = {head}
    events = []
    a.find_repaired_test_cases(on_case=lambda case: events.append(("case", case.repaired)),
                               on_progress=lambda commit: events.append(("done", commit)), **kwargs)
    return events


def test_cases_are_reported_as_they_are_found_and_progress_follows_each_pair(three_repairs_repo):
    repository_path, shas = three_repairs_repo
    events = _search(repository_path, shas[-1])
    # newest pair first; a case is always reported before the pair is declared done; a pair is "child:parent"
    assert events == [("case", shas[3]), ("done", f"{shas[3]}:{shas[2]}"),
                      ("case", shas[2]), ("done", f"{shas[2]}:{shas[1]}"),
                      ("case", shas[1]), ("done", f"{shas[1]}:{shas[0]}")]


def test_resuming_after_a_pair_does_only_the_remaining_work(three_repairs_repo):
    repository_path, shas = three_repairs_repo
    events = _search(repository_path, shas[-1], resume_after=f"{shas[3]}:{shas[2]}")
    assert events == [("case", shas[2]), ("done", f"{shas[2]}:{shas[1]}"),
                      ("case", shas[1]), ("done", f"{shas[1]}:{shas[0]}")]

    events = _search(repository_path, shas[-1], resume_after=f"{shas[1]}:{shas[0]}")  # everything was done
    assert events == []

    # a bare parent hash (the marker format of the first version) still means "the first pair with this parent"
    assert _search(repository_path, shas[-1], resume_after=shas[2]) == _search(
        repository_path, shas[-1], resume_after=f"{shas[3]}:{shas[2]}")


def test_resumed_and_uninterrupted_searches_find_the_same_cases(three_repairs_repo):
    repository_path, shas = three_repairs_repo
    full = [e for e in _search(repository_path, shas[-1]) if e[0] == "case"]
    first = [e for e in _search(repository_path, shas[-1]) if e[0] == "case"][:1]  # run killed after one pair
    rest = [e for e in _search(repository_path, shas[-1], resume_after=f"{shas[3]}:{shas[2]}") if e[0] == "case"]
    assert first + rest == full


def test_a_pair_without_work_does_not_move_the_marker(git_available, tmp_path):
    work = tmp_path / "proj"
    (work / "tests").mkdir(parents=True)
    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)
    for n in (1, 2):
        (work / "lib.py").write_text(f"x = {n}\n", encoding="utf-8")
        _git(["add", "-A"], cwd=work)
        _git(["commit", "-q", "-m", f"c{n}"], cwd=work)
    head = _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()
    assert _search(str(tmp_path), head) == []


def test_the_search_does_not_depend_on_what_is_currently_checked_out(three_repairs_repo):
    # Regression: the relevant commits used to come from `git log` of whatever the working tree had checked out,
    # so a second search on the same clone (left at an old commit by the first) skipped the newest pair.
    repository_path, shas = three_repairs_repo
    first = _search(repository_path, shas[-1])
    second = _search(repository_path, shas[-1])
    assert first == second and len([e for e in first if e[0] == "case"]) == 3


# --------------------------------------------------------------------------- #
#  scheduling over the whole history                                          #
# --------------------------------------------------------------------------- #
def _fast_import(work, commits):
    """Create commits quickly: `commits` is a list of {path: text}; every commit sets the files it lists."""
    stream = []
    for i, files in enumerate(commits, 1):
        message = f"c{i}"
        stream.append(f"commit refs/heads/main\ncommitter T <t@example.com> {1700000000 + i} +0000\n"
                      f"data {len(message)}\n{message}\n")
        for path, text in files.items():
            stream.append(f"M 100644 inline {path}\ndata {len(text.encode())}\n{text}\n")
    # bytes, not text: text mode would turn every "\n" into "\r\n" on Windows and git rejects that
    subprocess.run(["git", "fast-import", "--quiet"], cwd=str(work), input="".join(stream).encode("utf-8"), check=True)
    _git(["checkout", "-q", "-f", "main"], cwd=work)
    return _git(["rev-parse", "HEAD"], cwd=work).stdout.strip()


def _init(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    _git(["init", "-q", "-b", "main"], cwd=work)
    _git(["config", "user.email", "t@example.com"], cwd=work)
    _git(["config", "user.name", "Tester"], cwd=work)
    _git(["config", "commit.gpgsign", "false"], cwd=work)
    return work


def test_a_repair_at_the_very_start_of_a_long_history_is_found(git_available, tmp_path):
    # Regression for the old cap of 1000 visited commits, counted from the newest, which never reached it.
    work = _init(tmp_path)
    commits = [{"lib.py": f"def f():\n    return {v}\n",
                "test_mod.py": f"from lib import f\n\n\ndef test_f():\n    assert f() == {v}\n"} for v in (1, 2)]
    commits += [{"notes.txt": f"note {n}\n"} for n in range(1100)]
    head = _fast_import(work, commits)
    assert int(_git(["rev-list", "--count", "HEAD"], cwd=work).stdout) == 1102

    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    a.current_hash = head
    cases = a.find_repaired_test_cases()
    assert [c.test_name for c in cases] == ["test_f"]
    assert _totals(a.funnel_rows())["cases"] == 1


def _burst_repo(tmp_path):
    """21 commits; commit i appends i lines to a test file and i lines to a source file (co-change score i)."""
    work = _init(tmp_path)
    lib, test, commits = "", "from lib import f\n", []
    for i in range(21):
        lib += "".join(f"x{i}_{k} = {i}\n" for k in range(i))
        test += "".join(f"# t{i}_{k}\n" for k in range(i))
        commits.append({"lib.py": lib + "def f():\n    return 1\n", "test_mod.py": test})
    return work, _fast_import(work, commits)


def test_the_plan_reaches_every_part_of_the_history_before_going_deeper(git_available, tmp_path):
    work, head = _burst_repo(tmp_path)
    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    relevant = set(_git(["log", "--format=%H", "--diff-filter=M", head], cwd=work).stdout.split())
    plan, buckets, skipped = a.plan_pairs(head, relevant)

    assert len(plan) == 20 and not skipped
    first_round = [buckets[child] for child, _ in plan[:10]]
    assert first_round == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # one per bucket, the newest part of the history first

    # inside a bucket the biggest co-change goes first: commit i has score i, so a larger index comes earlier
    order = {commit: n for n, commit in enumerate(_git(["rev-list", "--reverse", head], cwd=work).stdout.split())}
    for bucket in range(10):
        members = [order[child] for child, _ in plan if buckets[child] == bucket]
        assert members == sorted(members, reverse=True)


def test_the_plan_is_the_same_every_time(git_available, tmp_path):
    work, head = _burst_repo(tmp_path)
    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    relevant = set(_git(["log", "--format=%H", "--diff-filter=M", head], cwd=work).stdout.split())
    assert a.plan_pairs(head, relevant)[0] == a.plan_pairs(head, relevant)[0]


def test_max_pairs_limits_the_work_and_takes_the_plan_in_order(three_repairs_repo):
    repository_path, shas = three_repairs_repo
    events = _search(repository_path, shas[-1], max_pairs=1)
    assert events == [("case", shas[3]), ("done", f"{shas[3]}:{shas[2]}")]


def test_require_source_change_leaves_out_test_only_pairs(git_available, tmp_path):
    work = _init(tmp_path)
    test = "from lib import f\n\n\ndef test_f():\n    assert f() == {v}\n"
    head = _fast_import(work, [
        {"lib.py": "def f():\n    return 1\n", "test_mod.py": test.format(v=1)},
        {"test_mod.py": test.format(v=1) + "# a comment, no source change\n"},   # test only
        {"lib.py": "def f():\n    return 2\n", "test_mod.py": test.format(v=2)},  # test and source
    ])
    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    relevant = set(_git(["log", "--format=%H", "--diff-filter=M", head], cwd=work).stdout.split())

    everything = a.plan_pairs(head, relevant)[0]
    plan, buckets, skipped = a.plan_pairs(head, relevant, require_source_change=True)
    assert len(everything) == 2 and len(plan) == 1 and sum(skipped.values()) == 1

    a.current_hash = head
    a.find_repaired_test_cases(require_source_change=True)
    assert _totals(a.funnel_rows())["skipped_no_source"] == 1


def test_the_funnel_says_why_a_test_could_not_be_run(git_available, tmp_path):
    work = _init(tmp_path)
    test = "import no_such_package_for_the_pytarget_test\n\n\ndef test_f():\n    assert {v} == {v}\n"
    head = _fast_import(work, [{"test_mod.py": test.format(v=1)}, {"test_mod.py": test.format(v=2)}])

    a = repository_actions.RepositoryActions("local/proj", str(tmp_path))
    a.current_hash = head
    assert a.find_repaired_test_cases() == set()
    totals = _totals(a.funnel_rows())
    assert totals["parent_unknown"] == 1
    assert totals["why:parent:missing_module:no_such_package_for_the_pytarget_test"] == 1
