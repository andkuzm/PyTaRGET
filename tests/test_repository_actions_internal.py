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
