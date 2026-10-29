"""Internal tests for repository_search/repository_actions.py.

These cover the AST parsing, diffing, annotation and filesystem-walk helpers
without touching git or the network.  They use the ``make_actions`` factory
(see conftest.py) to materialise a fake repo on disk.
"""

import os
import textwrap

import pytest


def dedent(s):
    return textwrap.dedent(s).lstrip("\n")


# --------------------------------------------------------------------------- #
#  find_test_methods                                                          #
# --------------------------------------------------------------------------- #
class TestFindTestMethods:
    SRC = dedent(
        """
        import pytest

        def test_top_level():
            assert True

        @pytest.mark.parametrize("x", [1, 2])
        def test_parametrized(x):
            assert x

        class TestThing:
            def test_in_class(self):
                assert True
            def helper(self):
                return 1

        import unittest
        class MyCase(unittest.TestCase):
            def test_case_method(self):
                self.assertTrue(True)
        """
    )

    def _names(self, actions):
        return {tuple(x) for x in actions.find_test_methods("tests/test_x.py")}

    def test_finds_plain_test_function(self, make_actions):
        a = make_actions({"tests/test_x.py": self.SRC})
        assert ("tests/test_x.py", "test_top_level") in self._names(a)

    def test_skips_parametrized_top_level(self, make_actions):
        a = make_actions({"tests/test_x.py": self.SRC})
        assert ("tests/test_x.py", "test_parametrized") not in self._names(a)

    def test_finds_class_qualified_methods(self, make_actions):
        a = make_actions({"tests/test_x.py": self.SRC})
        names = self._names(a)
        assert ("tests/test_x.py", "TestThing.test_in_class") in names
        assert ("tests/test_x.py", "MyCase.test_case_method") in names

    def test_ignores_non_test_helpers(self, make_actions):
        a = make_actions({"tests/test_x.py": self.SRC})
        assert ("tests/test_x.py", "TestThing.helper") not in self._names(a)

    def test_no_bare_duplicate_for_class_methods(self, make_actions):
        # ast.walk() previously also descended into class bodies, re-adding
        # each class method a second time under its bare (unqualified) name.
        # That bare entry can never be collected by pytest (there is no
        # module-level "test_in_class" function), so it must not appear.
        a = make_actions({"tests/test_x.py": self.SRC})
        names = self._names(a)
        assert ("tests/test_x.py", "test_in_class") not in names
        assert ("tests/test_x.py", "test_case_method") not in names

    def test_unreadable_file_returns_empty(self, make_actions):
        a = make_actions({})  # file not created
        assert a.find_test_methods("tests/missing.py") == []

    def test_syntax_error_returns_empty(self, make_actions):
        a = make_actions({"tests/bad.py": "def test_x(:\n  pass\n"})
        assert a.find_test_methods("tests/bad.py") == []


# --------------------------------------------------------------------------- #
#  extract_method_code / _get_method_line_range                               #
# --------------------------------------------------------------------------- #
class TestExtractMethodCode:
    SRC = dedent(
        """
        import pytest

        @staticmethod
        def test_decorated():
            x = 1
            assert x == 1

        @pytest.mark.parametrize("y", [1])
        def test_param(y):
            assert y

        class TestC:
            def test_method(self):
                return 7
        """
    )

    def test_extracts_with_decorator(self, make_actions):
        a = make_actions({"t.py": self.SRC})
        code = a.extract_method_code("t.py", "test_decorated")
        assert code.startswith("@staticmethod")
        assert "def test_decorated():" in code
        assert "assert x == 1" in code

    def test_parametrized_returns_empty(self, make_actions):
        a = make_actions({"t.py": self.SRC})
        assert a.extract_method_code("t.py", "test_param") == ""

    def test_extracts_class_method(self, make_actions):
        a = make_actions({"t.py": self.SRC})
        code = a.extract_method_code("t.py", "TestC.test_method")
        assert "def test_method(self):" in code
        assert "return 7" in code

    def test_missing_file_returns_empty(self, make_actions):
        a = make_actions({})
        assert a.extract_method_code("nope.py", "test_x") == ""

    def test_unknown_method_returns_empty(self, make_actions):
        a = make_actions({"t.py": self.SRC})
        assert a.extract_method_code("t.py", "test_absent") == ""

    def test_line_range_includes_decorator(self, make_actions):
        a = make_actions({"t.py": self.SRC})
        rng = a._get_method_line_range("t.py", "test_decorated")
        assert rng is not None
        start, end = rng
        lines = self.SRC.splitlines()
        chunk = "\n".join(lines[start:end])
        assert chunk.startswith("@staticmethod")
        assert "assert x == 1" in chunk

    def test_line_range_missing_returns_none(self, make_actions):
        a = make_actions({"t.py": self.SRC})
        assert a._get_method_line_range("t.py", "test_absent") is None


# --------------------------------------------------------------------------- #
#  extract_test_imports                                                       #
# --------------------------------------------------------------------------- #
class TestExtractImports:
    def test_collects_import_statements(self, make_actions):
        src = dedent(
            """
            import os
            from pathlib import Path
            x = 1
            import sys
            """
        )
        a = make_actions({"t.py": src})
        imports = a.extract_test_imports("t.py")
        assert "import os" in imports
        assert "from pathlib import Path" in imports
        assert "import sys" in imports
        assert "x = 1" not in imports

    def test_unparseable_returns_empty(self, make_actions):
        a = make_actions({"t.py": "def f(:\n"})
        assert a.extract_test_imports("t.py") == ""


# --------------------------------------------------------------------------- #
#  is_test_method_changed                                                     #
# --------------------------------------------------------------------------- #
class TestIsTestMethodChanged:
    def test_identical_is_unchanged(self, make_actions):
        a = make_actions({})
        code = "def test_x():\n    assert True\n"
        assert a.is_test_method_changed(code, code) is False

    def test_comment_only_change_is_unchanged(self, make_actions):
        a = make_actions({})
        before = "def test_x():\n    assert True\n"
        after = "def test_x():\n    # a new comment\n    assert True\n"
        assert a.is_test_method_changed(before, after) is False

    def test_real_change_is_detected(self, make_actions):
        a = make_actions({})
        before = "def test_x():\n    assert foo() == 1\n"
        after = "def test_x():\n    assert foo() == 2\n"
        assert a.is_test_method_changed(before, after) is True

    def test_whitespace_only_change_is_unchanged(self, make_actions):
        a = make_actions({})
        before = "def test_x():\n    assert True\n"
        after = "def test_x():\n    assert True   \n"
        assert a.is_test_method_changed(before, after) is False


# --------------------------------------------------------------------------- #
#  filter_diff_lines                                                          #
# --------------------------------------------------------------------------- #
class TestFilterDiffLines:
    def test_drops_headers(self, make_actions):
        a = make_actions({})
        lines = ["--- a", "+++ b", "@@ -1 +1 @@", "-old", "+new", " ctx"]
        out = a.filter_diff_lines(lines)
        assert out == ["-old", "+new", " ctx"]

    def test_strip_markers(self, make_actions):
        a = make_actions({})
        lines = ["@@ -1 +1 @@", "-old", "+new", " ctx"]
        out = a.filter_diff_lines(lines, strip_markers=True)
        assert out == ["old", "new", " ctx"]


# --------------------------------------------------------------------------- #
#  annotate_code                                                              #
# --------------------------------------------------------------------------- #
class TestAnnotateCode:
    def test_builds_annotated_blocks(self, make_actions):
        a = make_actions({"t.py": "import os\n"})
        broken = "def test_x():\n    assert foo() == 1"
        repaired = "def test_x():\n    assert foo() == 2"
        source = "class Helper:\n[<HUNK>] Helper.foo\n...\n[</HUNK>]"

        out = a.annotate_code(broken, repaired, source, "t.py")

        assert "[<TESTCONTEXT>]" in out and "[</TESTCONTEXT>]" in out
        assert "[<BREAKAGE>]" in out and "[</BREAKAGE>]" in out
        assert "[<REPAIREDTEST>]" in out and "[</REPAIREDTEST>]" in out
        assert "[<REPAIRCONTEXT>]" in out and "[</REPAIRCONTEXT>]" in out
        # imports pulled from the on-disk test file
        assert "import os" in out
        # the changed lines land in the right blocks
        breakage = out.split("[<BREAKAGE>]")[1].split("[</BREAKAGE>]")[0]
        repaired_block = out.split("[<REPAIREDTEST>]")[1].split("[</REPAIREDTEST>]")[0]
        assert "assert foo() == 1" in breakage
        assert "assert foo() == 2" in repaired_block
        assert source in out

    def test_no_diff_returns_empty(self, make_actions):
        a = make_actions({"t.py": "import os\n"})
        same = "def test_x():\n    assert True"
        assert a.annotate_code(same, same, "src", "t.py") == ""

    BROKEN = ("def test_x(self):\n        a = make(1)\n        b = make(2)\n        c = make(3)\n"
              "        d = make(4)\n        assert a == 1\n        assert d == 4")

    @staticmethod
    def _rebuild(out):
        """(broken test, repaired test) as the annotation encodes them."""
        context = out.split("[<TESTCONTEXT>]")[1].split("[</TESTCONTEXT>]")[0].split("\n")
        start, end = context.index("[<BREAKAGE>]"), context.index("[</BREAKAGE>]")
        head = next(i for i, line in enumerate(context) if line.startswith("def "))
        before, breakage, after = context[head:start], context[start + 1:end], context[end + 1:]
        repair = out.split("[<REPAIREDTEST>]\n")[1].split("\n[</REPAIREDTEST>]")[0].split("\n")
        return ("\n".join(before + breakage + after).rstrip("\n"), "\n".join(before + repair + after).rstrip("\n"))

    def test_unchanged_lines_keep_their_exact_indentation(self, make_actions):
        # Regression: unified-diff context lines carry a leading " " that used to stay in the annotation, so every
        # unchanged line was indented one space too deep (" def test_x(self):") and a repair spanning an unchanged
        # line mixed 4 and 5 spaces.
        a = make_actions({"t.py": "import os\n"})
        repaired = (self.BROKEN.replace("make(1)", "make(1, strict=True)")
                    .replace("assert d == 4", "assert d == 40"))
        out = a.annotate_code(self.BROKEN, repaired, "src", "t.py")

        assert "\ndef test_x(self):\n" in out and "\n def " not in out
        assert self._rebuild(out) == (self.BROKEN, repaired)
        repaired_block = out.split("[<REPAIREDTEST>]\n")[1].split("\n[</REPAIREDTEST>]")[0]
        assert "        b = make(2)" in repaired_block.split("\n")  # the unchanged line in the middle, exact

    def test_changed_def_line_is_handled(self, make_actions):
        a = make_actions({"t.py": "import os\n"})
        broken = "def test_y(tmp):\n    run(tmp)\n    assert ok()"
        repaired = "def test_y(tmp_path):\n    run(tmp_path)\n    assert ok()"
        assert self._rebuild(a.annotate_code(broken, repaired, "src", "t.py")) == (broken, repaired)

    def test_case_that_cannot_round_trip_is_skipped(self, make_actions):
        # A removed line starting with "--" shows up as "---..." in the diff, which filter_diff_lines() treats as
        # a header and drops; the annotation would then lack that line, so the case must be skipped.
        a = make_actions({"t.py": "import os\n"})
        broken = "def test_x():\n    x = 1\n-- bad\n    assert x"
        repaired = "def test_x():\n    x = 1\n    assert x"
        assert a.annotate_code(broken, repaired, "src", "t.py") == ""


# --------------------------------------------------------------------------- #
#  pasting an old test over a new one (run_test_with_overridden_test_code)    #
# --------------------------------------------------------------------------- #
CLASS_TESTS = dedent(
    """
    import unittest


    class TestThing(unittest.TestCase):
        def test_other(self):
            self.assertEqual(1, 1)

        @staticmethod
        def test_value():
            x = 2
            assert x == 2
    """
)


class TestRestoreIndent:
    def test_first_line_gets_the_indentation_back(self):
        import repository_actions
        assert repository_actions._restore_indent("def t(self):\n        pass", "    ") == "    def t(self):\n        pass"

    def test_module_level_code_is_left_alone(self):
        import repository_actions
        assert repository_actions._restore_indent("def t():\n    pass", "") == "def t():\n    pass"

    def test_already_indented_code_is_left_alone(self):
        import repository_actions
        assert repository_actions._restore_indent("    def t(self):\n        pass", "    ") == "    def t(self):\n        pass"

    def test_empty_code(self):
        import repository_actions
        assert repository_actions._restore_indent("", "    ") == ""


class TestOverriddenTestCode:
    """The old test is pasted into the new file; for a method of a class that used to be an IndentationError."""

    def _run(self, a, monkeypatch, rel, name, code):
        import repository_actions
        pasted = {}

        def fake_run(repo_dir, rel_path, test_method, log_path):
            pasted["text"] = (repo_dir / rel_path).read_text(encoding="utf-8")
            return repository_actions.TestVerdict(repository_actions.TestVerdict.FAILURE, set(), "FAILED")

        monkeypatch.setattr(repository_actions, "compile_and_run_test_python", fake_run)
        return a.run_test_with_overridden_test_code(rel, name, code), pasted.get("text")

    def test_class_method_is_pasted_with_its_indentation(self, make_actions, monkeypatch):
        import ast
        a = make_actions({"t.py": CLASS_TESTS})
        old = a.extract_method_code("t.py", "TestThing.test_other").replace("(1, 1)", "(1, 2)")
        verdict, text = self._run(a, monkeypatch, "t.py", "TestThing.test_other", old)
        assert verdict.status == "failure"
        ast.parse(text)  # used to raise IndentationError
        assert "    def test_other(self):\n        self.assertEqual(1, 2)" in text

    def test_decorated_class_method(self, make_actions, monkeypatch):
        import ast
        a = make_actions({"t.py": CLASS_TESTS})
        old = a.extract_method_code("t.py", "TestThing.test_value").replace("== 2", "== 3")
        verdict, text = self._run(a, monkeypatch, "t.py", "TestThing.test_value", old)
        ast.parse(text)
        assert "    @staticmethod\n    def test_value():" in text
        assert "assert x == 3" in text

    def test_module_level_function_unchanged(self, make_actions, monkeypatch):
        a = make_actions({"t.py": "def test_a():\n    assert 1\n"})
        verdict, text = self._run(a, monkeypatch, "t.py", "test_a", "def test_a():\n    assert 0")
        assert text == "def test_a():\n    assert 0\n"

    def test_original_file_is_restored(self, make_actions, monkeypatch):
        a = make_actions({"t.py": CLASS_TESTS})
        old = a.extract_method_code("t.py", "TestThing.test_other")
        self._run(a, monkeypatch, "t.py", "TestThing.test_other", old)
        assert (a.repo_dir / "t.py").read_text(encoding="utf-8") == CLASS_TESTS

    def test_unparsable_paste_is_unknown_and_never_run(self, make_actions, monkeypatch):
        import repository_actions
        a = make_actions({"t.py": CLASS_TESTS})
        verdict, text = self._run(a, monkeypatch, "t.py", "TestThing.test_other",
                                  "def test_other(self):\n  pass\n        pass")  # not parsable
        assert verdict.status == "unknown"
        assert text is None  # pytest was not started
        assert repository_actions._unknown_reason(verdict.log) == "override_not_pasted"
        assert (a.repo_dir / "t.py").read_text(encoding="utf-8") == CLASS_TESTS


# --------------------------------------------------------------------------- #
#  changed_test_methods: same answer as the checkout-and-scan method          #
# --------------------------------------------------------------------------- #
OLD_TESTS = """import pytest
import unittest


def test_changed():
    assert f() == 1


def test_same():
    assert f() == 1


# a comment-only change is not a change
def test_comment():
    assert f() == 1


@pytest.mark.parametrize("x", [1, 2])
def test_param(x):
    assert f() == x


@pytest.mark.slow
def test_decorated():
    assert f() == 1


async def test_async():
    assert await f() == 1


class TestFoo:
    def test_method(self):
        assert f() == 1

    def test_method_same(self):
        assert f() == 1


class Mixed(unittest.TestCase):
    def test_unit(self):
        self.assertEqual(f(), 1)

    class Inner:
        pass


class Helper:
    def test_not_collected(self):
        assert f() == 1
"""

NEW_TESTS = (OLD_TESTS.replace("== 1\n\n\ndef test_same", "== 2\n\n\ndef test_same")
             .replace("# a comment-only change is not a change", "# a comment-only change is still not a change")
             .replace("def test_param(x):\n    assert f() == x", "def test_param(x):\n    assert f() != x")
             .replace("def test_decorated():\n    assert f() == 1", "def test_decorated():\n    assert f() == 3")
             .replace("assert await f() == 1", "assert await f() == 4")
             .replace("def test_method(self):\n        assert f() == 1", "def test_method(self):\n        assert f() == 5")
             .replace("self.assertEqual(f(), 1)", "self.assertEqual(f(), 6)")
             .replace("def test_not_collected(self):\n        assert f() == 1",
                      "def test_not_collected(self):\n        assert f() == 7"))


def _git(cwd, *args):
    import subprocess
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


def _legacy_changed_tests(actions, parent, child):
    """What the loop used to do: check out each side, read every file, parse the file once per test."""
    def collect(commit):
        actions.git_checkout_with_retry(str(actions.repo_dir), commit)
        found = {}
        for file in actions.list_test_files():
            rel = str(file.relative_to(actions.repo_dir))
            for _, name in actions.find_test_methods(rel):
                code = actions.extract_method_code(rel, name)
                if code:
                    found[(rel, name)] = code
        return found
    before, after = collect(parent), collect(child)
    return {k: before[k] for k in before if k in after and actions.is_test_method_changed(before[k], after[k])}


class TestChangedTestMethods:
    def _build_repo(self, make_actions):
        a = make_actions({})
        repo = a.repo_dir
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
        _git(repo, "config", "core.autocrlf", "false")

        def write(rel, content, binary=False):
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if binary:
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8", newline="")

        def commit(message):
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "-m", message)
            return _git(repo, "rev-parse", "HEAD")

        write("tests/test_mod.py", OLD_TESTS)
        write("tests/test_crlf.py", "def test_crlf():\r\n    assert f() == 1\r\n")
        write("tests/test_syntax.py", "def test_s():\n    assert 1 == 1\n")
        write("tests/test_renamed.py", "def test_r():\n    assert 1 == 1\n")
        write("tests/test_latin1.py", "# caf\xe9\ndef test_l():\n    assert 1 == 1\n".encode("latin-1"), binary=True)
        write("tests/test_gone.py", "def test_g():\n    assert 1 == 1\n")
        write("src/lib.py", "def f():\n    return 1\n")
        write("tests/helpers.py", "def helper():\n    return 1\n")
        parent = commit("old")

        write("tests/test_mod.py", NEW_TESTS)
        write("tests/test_crlf.py", "def test_crlf():\r\n    assert f() == 2\r\n")
        write("tests/test_syntax.py", "def test_s(:\n    assert 1 == 2\n")
        (repo / "tests/test_renamed.py").rename(repo / "tests/test_renamed2.py")
        write("tests/test_renamed2.py", "def test_r():\n    assert 1 == 2\n")
        write("tests/test_latin1.py", "# caf\xe9\ndef test_l():\n    assert 1 == 2\n".encode("latin-1"), binary=True)
        (repo / "tests/test_gone.py").unlink()
        write("tests/test_new.py", "def test_n():\n    assert 1 == 1\n")
        write("src/lib.py", "def f():\n    return 2\n")
        write("tests/helpers.py", "def helper():\n    return 2\n")
        return a, parent, commit("new")

    def test_matches_the_checkout_and_scan_method(self, make_actions):
        a, parent, child = self._build_repo(make_actions)
        legacy = _legacy_changed_tests(a, parent, child)
        new = a.changed_test_methods(parent, child)

        assert new == legacy
        names = sorted(name for _, name in new)
        # the changed ones, including a decorated test, an async one, a class method, a TestCase method and a file
        # with CRLF line endings; a method of a class pytest would not collect (Helper) is not among them
        assert names == sorted(["test_changed", "test_decorated", "test_async", "TestFoo.test_method",
                                "Mixed.test_unit", "test_crlf"])
        # and not these: parametrized, unchanged, comment-only, syntax error, rename, deleted, new, no tests
        for skipped in ("test_param", "test_same", "test_comment", "TestFoo.test_method_same", "test_s", "test_r",
                        "test_l", "test_g", "test_n", "helper"):
            assert skipped not in names

    def test_nothing_changed_without_test_edits(self, make_actions):
        a = make_actions({})
        repo = a.repo_dir
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
        (repo / "lib.py").write_text("x = 1\n")
        (repo / "test_a.py").write_text("def test_a():\n    assert True\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "one")
        first = _git(repo, "rev-parse", "HEAD")
        (repo / "lib.py").write_text("x = 2\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "two")
        assert a.changed_test_methods(first, _git(repo, "rev-parse", "HEAD")) == {}

    def test_unknown_commit_gives_none(self, make_actions):
        a = make_actions({})
        _git(a.repo_dir, "init", "-q")
        assert a.changed_test_methods("0" * 40, "1" * 40) is None


# --------------------------------------------------------------------------- #
#  get_commit_date                                                            #
# --------------------------------------------------------------------------- #
class TestGetCommitDate:
    def test_reads_the_committer_date_and_tolerates_unknown_commits(self, make_actions):
        import subprocess
        a = make_actions({"t.py": "x = 1\n"})
        env = dict(os.environ, GIT_COMMITTER_DATE="2021-03-04T05:06:07+00:00",
                   GIT_AUTHOR_DATE="2021-03-04T05:06:07+00:00")

        def run(*args):
            return subprocess.run(["git", *args], cwd=str(a.repo_dir), check=True, capture_output=True,
                                  text=True, env=env).stdout.strip()

        run("init", "-q")
        run("config", "user.email", "t@example.com")
        run("config", "user.name", "t")
        run("add", ".")
        run("commit", "-q", "-m", "init")

        from datetime import datetime
        date = a.get_commit_date(run("rev-parse", "HEAD"))  # git prints UTC as "...Z"
        assert datetime.fromisoformat(date) == datetime.fromisoformat("2021-03-04T05:06:07+00:00")
        assert a.get_commit_date("0" * 40) == ""


# --------------------------------------------------------------------------- #
#  format_inline_diff                                                         #
# --------------------------------------------------------------------------- #
class TestFormatInlineDiff:
    def test_groups_add_and_del_blocks(self, make_actions):
        a = make_actions({})
        diff = ["@@ -1 +1 @@", "-old1", "-old2", "+new1", " ctx"]
        out = a.format_inline_diff("Cls.method", diff)
        assert "[<HUNK>]" in out and "[</HUNK>]" in out
        assert "[<DEL>]" in out and "[</DEL>]" in out
        assert "[<ADD>]" in out and "[</ADD>]" in out
        del_block = out.split("[<DEL>]")[1].split("[</DEL>]")[0]
        assert "old1" in del_block and "old2" in del_block
        add_block = out.split("[<ADD>]")[1].split("[</ADD>]")[0]
        assert "new1" in add_block
        assert "ctx" in out

    def test_empty_diff_returns_empty(self, make_actions):
        a = make_actions({})
        assert a.format_inline_diff("Cls.method", []) == ""


# --------------------------------------------------------------------------- #
#  extract_executed_methods_from_file                                         #
# --------------------------------------------------------------------------- #
class TestExtractExecutedMethods:
    SRC = dedent(
        """
        class Foo:
            def bar(self):
                return 1

        def baz():
            return 2
        """
    )

    def test_class_method_keyed_by_class(self, make_actions, tmp_path):
        a = make_actions({"mod.py": self.SRC})
        path = tmp_path / "proj" / "mod.py"
        methods = a.extract_executed_methods_from_file(str(path), {2, 3})
        assert "Foo.bar" in methods
        assert "def bar" in methods["Foo.bar"]

    def test_global_function_keyed_global(self, make_actions, tmp_path):
        a = make_actions({"mod.py": self.SRC})
        path = tmp_path / "proj" / "mod.py"
        methods = a.extract_executed_methods_from_file(str(path), {6})
        assert "Global.baz" in methods

    def test_uncovered_lines_yield_nothing(self, make_actions, tmp_path):
        a = make_actions({"mod.py": self.SRC})
        path = tmp_path / "proj" / "mod.py"
        assert a.extract_executed_methods_from_file(str(path), {999}) == {}

    def test_unreadable_returns_empty(self, make_actions):
        a = make_actions({})
        assert a.extract_executed_methods_from_file("/no/such/file.py", {1}) == {}


# --------------------------------------------------------------------------- #
#  has_tests / list_test_files                                                #
# --------------------------------------------------------------------------- #
class TestFilesystemWalkers:
    def test_has_tests_true(self, make_actions):
        a = make_actions({"pkg/test_a.py": "def test_a():\n    pass\n",
                          "pkg/util.py": "def helper():\n    pass\n"})
        assert a.has_tests() is True

    def test_has_tests_false(self, make_actions):
        a = make_actions({"pkg/util.py": "def helper():\n    pass\n"})
        assert a.has_tests() is False

    def test_list_test_files_selects_only_test_files(self, make_actions):
        a = make_actions({
            "pkg/test_a.py": "def test_a():\n    pass\n",
            "pkg/util.py": "def helper():\n    pass\n",
            "pkg/sub/test_b.py": "def test_b():\n    pass\n",
        })
        names = {p.name for p in a.list_test_files()}
        assert names == {"test_a.py", "test_b.py"}


# --------------------------------------------------------------------------- #
#  _invalidate_bytecode_cache                                                 #
# --------------------------------------------------------------------------- #
class TestInvalidateBytecodeCache:
    def test_removes_pycache_dirs_and_pyc_files(self, make_actions, tmp_path):
        a = make_actions({"pkg/mod.py": "x = 1\n"})
        work = tmp_path / "proj"
        cache = work / "pkg" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "mod.cpython-311.pyc").write_bytes(b"stale")
        loose = work / "pkg" / "other.pyc"
        loose.write_bytes(b"stale")

        a._invalidate_bytecode_cache()

        assert not cache.exists()
        assert not loose.exists()
        assert (work / "pkg" / "mod.py").exists()  # source untouched


# --------------------------------------------------------------------------- #
#  set_full_permissions                                                       #
# --------------------------------------------------------------------------- #
class TestSetFullPermissions:
    def test_grants_rwx(self, make_actions, tmp_path):
        a = make_actions({"pkg/mod.py": "x = 1\n"})
        a.set_full_permissions()
        f = tmp_path / "proj" / "pkg" / "mod.py"
        mode = os.stat(f).st_mode & 0o777
        assert mode & 0o700 == 0o700  # at least owner rwx

    def test_dangling_links_do_not_stop_it(self, make_actions, tmp_path):
        # psf/black (docs/show_your_style.md) and commaai/openpilot (opendbc) were lost to a link like these
        a = make_actions({"pkg/mod.py": "x = 1\n"})
        work = tmp_path / "proj"
        (work / "docs").mkdir()
        os.symlink(work / "nowhere.md", work / "docs" / "dangling.md")
        os.symlink(work / "no_such_dir", work / "dangling_dir")
        os.chmod(work / "pkg" / "mod.py", 0o600)
        a.set_full_permissions()  # used to raise FileNotFoundError
        assert os.stat(work / "pkg" / "mod.py").st_mode & 0o777 == 0o777

    def test_links_to_files_outside_the_repository_are_left_alone(self, make_actions, tmp_path, tmp_path_factory):
        a = make_actions({"pkg/mod.py": "x = 1\n"})
        outside = tmp_path_factory.mktemp("elsewhere") / "outside.txt"  # not under a.repository_path
        outside.write_text("not part of the repository")
        os.chmod(outside, 0o600)
        os.symlink(outside, tmp_path / "proj" / "link.txt")
        a.set_full_permissions()
        assert os.stat(outside).st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------- #
#  planning helpers                                                           #
# --------------------------------------------------------------------------- #
class TestPlanningHelpers:
    def test_round_robin_interleaves_and_handles_uneven_groups(self):
        import repository_actions
        assert repository_actions._round_robin([[1, 2, 3], [10], [20, 21]]) == [1, 10, 20, 2, 21, 3]
        assert repository_actions._round_robin([]) == []
        assert repository_actions._round_robin([[], []]) == []

    def test_test_paths_follow_the_same_rules_as_the_commit_filter(self):
        import repository_actions
        is_test = repository_actions._is_test_path
        for path in ("test_a.py", "pkg/test_a.py", "a_test.py", "tests/a.py", "pkg/tests/sub/a.py", "test/a.py"):
            assert is_test(path), path
        for path in ("a.py", "pkg/a.py", "latest.py", "pkg/contest/a.py", "testing/a.py", "tests.py"):
            assert not is_test(path), path


# --------------------------------------------------------------------------- #
#  repositories that cannot be installed                                      #
# --------------------------------------------------------------------------- #
class TestCloneOfAnUninstallableRepository:
    def _fake_git_and_pip(self, monkeypatch):
        import subprocess

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "clone" in cmd:
                os.makedirs(cmd[-1], exist_ok=True)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd[0] == "git" and "rev-parse" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "abc123\n", "")
            if "pip" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "ERROR: Directory '.' is not installable.\n")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_the_search_goes_on_without_installing(self, make_actions, monkeypatch, capsys):
        import repository_actions
        a = make_actions({})
        installs = []
        monkeypatch.setattr(a, "_install_test_dependencies", lambda dest, **kw: installs.append(dest))
        monkeypatch.delenv("PYTARGET_REQUIRE_INSTALL", raising=False)
        self._fake_git_and_pip(monkeypatch)

        dest = a.clone_repository_last()

        assert a.current_hash == "abc123" and installs == [dest]  # test requirements are still installed
        assert "going on without installing it" in capsys.readouterr().out

    def test_the_old_strictness_can_be_asked_for(self, make_actions, monkeypatch):
        a = make_actions({})
        monkeypatch.setattr(a, "_install_test_dependencies", lambda dest, **kw: None)
        monkeypatch.setenv("PYTARGET_REQUIRE_INSTALL", "1")
        self._fake_git_and_pip(monkeypatch)
        with pytest.raises(Exception, match="pip install failed"):
            a.clone_repository_last()


class TestDependenciesOfAnUninstallableProject:
    PYPROJECT = (
        '[project]\nname = "demo"\ndependencies = ["numpy>=2", "beautifulsoup4"]\n'
        '[project.optional-dependencies]\ntest = ["pytest-cov", "hypothesis"]\ndocs = ["sphinx"]\n'
    )

    def _pip_calls(self, make_actions, monkeypatch, project_installed):
        import subprocess
        a = make_actions({"pyproject.toml": self.PYPROJECT})
        calls = []

        def fake_run(cmd, **kwargs):
            if "pip" in cmd:
                calls.append(cmd[cmd.index("install") + 1:])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(subprocess, "run", fake_run)
        a._install_test_dependencies(str(a.repo_dir), project_installed=project_installed)
        return calls

    def test_runtime_dependencies_and_extras_are_installed_directly(self, make_actions, monkeypatch):
        calls = self._pip_calls(make_actions, monkeypatch, project_installed=False)
        assert ["numpy>=2", "beautifulsoup4"] in calls          # [project].dependencies
        assert ["pytest-cov", "hypothesis"] in calls            # the "test" extra, as plain requirements
        assert not any(arg.startswith(".[") for call in calls for arg in call)   # `.[extra]` cannot work here
        assert not any("sphinx" in call for call in calls)      # "docs" is not a test extra

    def test_an_installed_project_keeps_using_the_extras_of_pip(self, make_actions, monkeypatch):
        calls = self._pip_calls(make_actions, monkeypatch, project_installed=True)
        assert [".[test]"] in calls
        assert ["numpy>=2", "beautifulsoup4"] not in calls      # pip install . already pulled them in

    def test_one_unresolvable_requirement_does_not_cost_the_others(self, make_actions, monkeypatch):
        import subprocess
        a = make_actions({})
        installs = []

        def fake_run(cmd, **kwargs):
            requirements = cmd[cmd.index("install") + 1:]
            installs.append(requirements)
            return subprocess.CompletedProcess(cmd, 1 if len(requirements) > 1 else 0, "", "")  # batch fails
        monkeypatch.setattr(subprocess, "run", fake_run)
        a._pip_install_list(["good", "also-good"], str(a.repo_dir))
        assert installs == [["good", "also-good"], ["good"], ["also-good"]]


class TestUnknownReasons:
    def test_the_reason_comes_from_pytests_own_output(self):
        import repository_actions
        why = repository_actions._unknown_reason
        assert why("E   ModuleNotFoundError: No module named 'numpy.linalg'") == "missing_module:numpy"
        assert why("pytest: error: unrecognized arguments: --cov=.") == "pytest_usage_error"
        assert why("ERROR: file or directory not found: tests/test_x.py::test_y") == "test_not_found"
        assert why("ERROR at setup of test_a\nE ValueError") == "setup_error"
        assert why("ERROR collecting test_a.py\nE SyntaxError") == "collection_error"
        assert why("ImportError: cannot import name 'x' from 'y'") == "import_error"
        assert why("collected 0 items\n\nno tests ran in 0.01s") == "no_tests_ran"
        assert why("something unexpected") == "other"
        assert why(None) == "other"
