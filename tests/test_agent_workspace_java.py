"""Java agent workspaces: before-versions of changed tests, method extraction and the derived scoring row (local git repository, no network)."""
import subprocess
from pathlib import Path

import pytest

from data_processing import agent_workspace_java as aj
from data_processing import java_plausibility as jp

BEFORE = "package p;\n\npublic class FooTest {\n    @Test\n    public void testA() {\n        int v = Foo.plus(1, 2);\n        assertEquals(3, v);\n    }\n\n    @Test\n    public void testB() {\n        assertEquals(1, Foo.old());\n    }\n}\n"
AFTER = "package p;\n\npublic class FooTest {\n    @Test\n    public void testA() {\n        int v = Foo.add(1, 2);\n        assertEquals(3, v);\n    }\n\n    @Test\n    public void testB() {\n        assertEquals(1, Foo.fresh());\n    }\n}\n"


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def origin(tmp_path):
    src = tmp_path / "origin"
    (src / "src/test/java/p").mkdir(parents=True)
    (src / "src/main/java/p").mkdir(parents=True)
    git(src, "init", "-q")
    (src / "src/test/java/p/FooTest.java").write_text(BEFORE)
    (src / "src/main/java/p/Foo.java").write_text("class Foo { static int plus(int a, int b){return a+b;} }\n")
    git(src, "add", "-A"); git(src, "commit", "-qm", "before")
    b = git(src, "rev-parse", "HEAD").strip()
    (src / "src/test/java/p/FooTest.java").write_text(AFTER)
    (src / "src/test/java/p/NewTest.java").write_text("class NewTest {}\n")
    (src / "src/main/java/p/Foo.java").write_text("class Foo { static int add(int a, int b){return a+b;} }\n")
    git(src, "add", "-A"); git(src, "commit", "-qm", "after")
    return src, b, git(src, "rev-parse", "HEAD").strip()


def row(b, a):
    return {"ID": "o/r:1", "name": "p.FooTest.testA()", "bCommit": b, "aCommit": a, "bPath": "src/test/java/p/FooTest.java", "aPath": "src/test/java/p/FooTest.java",
            "bSource": {"startLine": 4, "code": "@Test\n    public void testA() {\n        int v = Foo.plus(1, 2);\n        assertEquals(3, v);\n    }"},
            "aSource": {"startLine": 4, "code": "@Test\n    public void testA() {\n        int v = Foo.add(1, 2);\n        assertEquals(3, v);\n    }"},
            "hunk": {"targetChanges": [{"lineNo": 6, "type": "ADD", "line": "x"}]}}


def clone_from(origin_path):
    return lambda repo, commit, dest: subprocess.run(["git", "clone", "-q", str(origin_path), str(dest)], check=True)


def test_default_workspace_breaks_only_the_target_method_and_counts_other_test_changes(origin, tmp_path):
    src, b, a = origin
    dest = tmp_path / "ws"
    info = aj.prepare_workspace(row(b, a), dest, clone=clone_from(src))
    test = (dest / "src/test/java/p/FooTest.java").read_text()
    assert "Foo.plus(1, 2)" in test and "Foo.add" not in test                 # the target method is broken again
    assert "Foo.fresh()" in test                                              # the sibling testB stays repaired (Java needs all tests to compile)
    assert (dest / "src/test/java/p/NewTest.java").exists()
    assert "add(" in (dest / "src/main/java/p/Foo.java").read_text()          # source keeps the repaired state
    assert git(dest, "log", "--format=%s").strip() == "snapshot: failing test"
    assert info["class"] == "FooTest" and info["method"] == "testA" and info["method_lines"] == [4, 8]
    assert info["other_test_lines_changed"] == 3                              # testB: 1 added + 1 deleted, NewTest.java: 1 added line


def test_revert_all_tests_mode_restores_every_changed_test_file(origin, tmp_path):
    src, b, a = origin
    dest = tmp_path / "ws"
    aj.prepare_workspace(row(b, a), dest, clone=clone_from(src), mode="revert_all_tests")
    assert (dest / "src/test/java/p/FooTest.java").read_text() == BEFORE
    assert not (dest / "src/test/java/p/NewTest.java").exists()


def test_moved_files_and_misplaced_methods_are_refused(origin, tmp_path):
    src, b, a = origin
    with pytest.raises(ValueError):
        aj.prepare_workspace(dict(row(b, a), aPath="elsewhere/FooTest.java"), tmp_path / "w1", clone=clone_from(src))
    bad = dict(row(b, a), aSource={"startLine": 6, "code": row(b, a)["aSource"]["code"]})
    with pytest.raises(ValueError):
        aj.prepare_workspace(bad, tmp_path / "w2", clone=clone_from(src))
    assert not (tmp_path / "w2").exists()


def test_collect_extracts_the_final_method_and_flags_outside_edits(origin, tmp_path):
    src, b, a = origin
    dest = tmp_path / "ws"
    info = aj.prepare_workspace(row(b, a), dest, clone=clone_from(src))
    f = dest / "src/test/java/p/FooTest.java"
    f.write_text(f.read_text().replace("Foo.plus", "Foo.add"))
    r = aj.collect_result(row(b, a), dest, info["method_lines"])
    assert r["method_found"] and "Foo.add(1, 2)" in r["method"] and r["method"].startswith("    @Test") and r["other_files_touched"] == []
    f.write_text(f.read_text().replace("testB", "testBB"))                                   # a sibling method changed too: outside the method
    r = aj.collect_result(row(b, a), dest, info["method_lines"])
    assert r["test_file_changed_outside_method"] and r["method"] is None
    (dest / "scratch.txt").write_text("x")
    assert aj.collect_result(row(b, a), dest, info["method_lines"])["new_files"] == ["scratch.txt"]


def test_the_derived_row_replaces_the_whole_after_method():
    r = row("b", "a")
    d = aj.derived_row(r)
    assert [c["lineNo"] for c in d["hunk"]["targetChanges"]] == [4, 5, 6, 7, 8]
    patched = jp.apply_candidate(AFTER, d, "    @Test\n    public void testA() {\n        assertTrue(true);\n    }")
    assert "assertTrue(true)" in patched and "Foo.fresh()" in patched and "Foo.add" not in patched      # only testA replaced, testB untouched


def test_prompt_names_the_method_and_command():
    text = aj.task_prompt(row("b", "a"), "mvn test -Dtest=FooTest#testA")
    assert "testA" in text and "FooTest.java" in text and "mvn test -Dtest=FooTest#testA" in text and "weaken" in text
