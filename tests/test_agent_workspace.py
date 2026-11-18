"""Agent workspaces (WP5 level 3): the broken state, no leaking history, result collection and the audit. Uses a local git repository, no network."""
import subprocess
from pathlib import Path

import pytest

from data_processing import agent_workspace as aw

REPAIRED = "import pytest\n\n\ndef test_add():\n    value = calc.add(1, 2)\n    assert value == 3\n\n\ndef test_other():\n    assert True\n"
BROKEN_LINES = "    value = calc.plus(1, 2)"
ROW = {"repository_name": "o/r", "repaired_hash": "x", "relative_path": "tests/test_calc.py", "test_name": "test_add",
       "output": "    value = calc.add(1, 2)", "input": f"[<TESTCONTEXT>]def test_add():\n[<BREAKAGE>]\n{BROKEN_LINES}\n[</BREAKAGE>]\n    assert value == 3[<REPAIRCONTEXT>]"}


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def origin(tmp_path):
    src = tmp_path / "origin"
    (src / "tests").mkdir(parents=True)
    git(src, "init", "-q")
    (src / "tests" / "test_calc.py").write_text("def test_add():\n    value = calc.plus(1, 2)\n    assert value == 3\n")
    git(src, "add", "-A"); git(src, "commit", "-qm", "broken")
    (src / "tests" / "test_calc.py").write_text(REPAIRED)
    git(src, "add", "-A"); git(src, "commit", "-qm", "repaired SECRET-REPAIR-COMMIT")
    return src


def local_clone(origin):
    def clone(repo, commit, dest):
        subprocess.run(["git", "clone", "-q", str(origin), str(dest)], check=True)
    return clone


class TestWorkspace:
    def test_the_test_is_broken_again_and_history_is_gone(self, origin, tmp_path):
        dest = tmp_path / "ws"
        info = aw.prepare_workspace(ROW, dest, clone=local_clone(origin))
        text = (dest / "tests" / "test_calc.py").read_text()
        assert "calc.plus(1, 2)" in text and "calc.add(1, 2)" not in text and "def test_other" in text
        assert info["test_name"] == "test_add"
        log = git(dest, "log", "--format=%s")
        assert log.strip() == "snapshot: failing test" and "SECRET" not in log
        assert subprocess.run(["git", "remote"], cwd=dest, capture_output=True, text=True).stdout.strip() == ""
        assert subprocess.run(["git", "status", "--porcelain"], cwd=dest, capture_output=True, text=True).stdout == ""

    def test_the_repaired_text_is_not_reachable_in_the_object_store(self, origin, tmp_path):
        dest = tmp_path / "ws"
        aw.prepare_workspace(ROW, dest, clone=local_clone(origin))
        blobs = git(dest, "rev-list", "--all", "--objects")
        assert git(dest, "count-objects", "-v")  # store exists
        for line in blobs.splitlines():
            sha = line.split()[0]
            kind = git(dest, "cat-file", "-t", sha).strip()
            if kind == "blob":
                assert "calc.add(1, 2)" not in git(dest, "cat-file", "-p", sha)

    def test_existing_directory_and_unfindable_site_are_refused(self, origin, tmp_path):
        (tmp_path / "there").mkdir()
        with pytest.raises(FileExistsError):
            aw.prepare_workspace(ROW, tmp_path / "there", clone=local_clone(origin))
        with pytest.raises(ValueError):
            aw.prepare_workspace(dict(ROW, output="    nothing like it"), tmp_path / "ws2", clone=local_clone(origin))
        assert not (tmp_path / "ws2").exists()


class TestResult:
    @pytest.fixture
    def ws(self, origin, tmp_path):
        dest = tmp_path / "ws"
        aw.prepare_workspace(ROW, dest, clone=local_clone(origin))
        return dest

    def test_a_clean_fix_inside_the_function(self, ws):
        f = ws / "tests" / "test_calc.py"
        f.write_text(f.read_text().replace("calc.plus", "calc.add"))
        r = aw.collect_result(ROW, ws)
        assert r["function"] == "def test_add():\n    value = calc.add(1, 2)\n    assert value == 3"
        assert r["touched_files"] == ["tests/test_calc.py"] and r["other_files_touched"] == [] and r["new_files"] == [] and not r["test_file_changed_outside_function"]

    def test_edits_outside_the_function_and_in_other_files_are_reported(self, ws):
        f = ws / "tests" / "test_calc.py"
        f.write_text("import os\n" + f.read_text().replace("calc.plus", "calc.add"))
        (ws / "extra.py").write_text("x = 1\n")                               # a new file is scratch, not a change of the project
        r = aw.collect_result(ROW, ws)
        assert r["test_file_changed_outside_function"] and r["new_files"] == ["extra.py"] and r["other_files_touched"] == []

    def test_a_changed_tracked_file_other_than_the_test_is_reported(self, ws):
        (ws / "tests" / "test_calc.py").write_text("def test_add():\n    assert True\n")
        (ws / "tests" / "conftest.py").write_text("x = 1\n")
        git(ws, "add", "-N", "tests/conftest.py")                              # known to git but not committed: shows as a modification
        assert aw.collect_result(ROW, ws)["other_files_touched"] == ["tests/conftest.py"]

    def test_a_deleted_function_is_reported(self, ws):
        (ws / "tests" / "test_calc.py").write_text("import os\n")
        assert aw.collect_result(ROW, ws)["function_found"] is False


def test_audit_flags_history_network_and_installs():
    flags = aw.audit_trajectory(["python -m pytest tests/test_calc.py", "git log --oneline", "pip install requests", "cat tests/test_calc.py",
                                 "curl https://github.com/o/r", "git status", "git diff"])
    assert [f["command"].split()[0] for f in flags] == ["git", "pip", "curl"]


def test_task_prompt_names_the_test_and_forbids_history():
    text = aw.task_prompt(ROW)
    assert "test_add" in text and "tests/test_calc.py" in text and "git history" in text and "weaken" in text


def test_caches_are_not_counted_as_touched_files(origin, tmp_path):
    ws = tmp_path / "ws"
    aw.prepare_workspace(ROW, ws, clone=local_clone(origin))
    (ws / "tests" / "__pycache__").mkdir()
    (ws / "tests" / "__pycache__" / "x.pyc").write_text("x")
    (ws / ".pytest_cache").mkdir()
    (ws / ".pytest_cache" / "v").write_text("x")
    assert aw.collect_result(ROW, ws)["touched_files"] == []


def test_commands_from_a_trajectory():
    traj = {"messages": [{"role": "system"}, {"role": "assistant", "extra": {"actions": [{"command": "ls"}, {"command": "git log"}]}},
                         {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"command": "pytest -x"}'}}]}]}
    assert aw.commands_from_trajectory(traj) == ["ls", "git log", "pytest -x"]
