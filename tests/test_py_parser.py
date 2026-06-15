"""Tests for repository_search/py_parser.py.

These are fully internal: TestVerdict logic, the diff/verdict parsers, and the
real ``compile_and_run_test_python`` / ``run_cmd`` against tiny temp projects
(pytest is the only external dependency and is already installed).
"""

import time
from pathlib import Path

import pytest

import py_parser
from py_parser import (
    TestVerdict,
    compile_and_run_test_python,
    parse_invalid_execution_py,
    parse_successful_execution_py,
    parse_test_failure_py,
    run_cmd,
)

# TestVerdict is a production class (not a pytest test class); tell pytest to
# skip collecting it now that it lives in this module's namespace.
TestVerdict.__test__ = False


# --------------------------------------------------------------------------- #
#  TestVerdict                                                                 #
# --------------------------------------------------------------------------- #
class TestTestVerdict:
    def test_success_predicates(self):
        v = TestVerdict(TestVerdict.SUCCESS, None)
        assert v.is_valid()
        assert v.succeeded()
        assert not v.is_broken()

    def test_failure_is_valid_and_broken(self):
        v = TestVerdict(TestVerdict.FAILURE, {3})
        assert v.is_valid()
        assert v.is_broken()
        assert not v.succeeded()

    def test_syntax_error_is_valid_and_broken(self):
        v = TestVerdict(TestVerdict.SYNTAX_ERR, None)
        assert v.is_valid()
        assert v.is_broken()

    @pytest.mark.parametrize(
        "status",
        [TestVerdict.TIMEOUT, TestVerdict.UNKNOWN, TestVerdict.TEST_NOT_EXECUTED,
         TestVerdict.UNEXPECTED_FAILURE, TestVerdict.UNCONVENTIONAL],
    )
    def test_invalid_statuses(self, status):
        v = TestVerdict(status, None)
        assert not v.is_valid()
        assert not v.is_broken()
        assert not v.succeeded()

    def test_to_dict_sorts_error_lines(self):
        v = TestVerdict(TestVerdict.FAILURE, {7, 3, 5})
        assert v.to_dict() == {"status": "failure", "error_lines": [3, 5, 7]}

    def test_to_dict_with_none_error_lines(self):
        v = TestVerdict(TestVerdict.SUCCESS, None)
        assert v.to_dict() == {"status": "success", "error_lines": None}

    def test_str(self):
        assert "success" in str(TestVerdict(TestVerdict.SUCCESS, None))


# --------------------------------------------------------------------------- #
#  Log parsers                                                                 #
# --------------------------------------------------------------------------- #
class TestParsers:
    def test_parse_successful(self):
        v = parse_successful_execution_py("anything")
        assert v.status == TestVerdict.SUCCESS

    def test_parse_failure_extracts_line_numbers(self):
        log = 'File "/x/test_mod.py", line 42, in test_thing\n  assert False'
        v = parse_test_failure_py(log, "test_mod", "test_thing")
        assert v.status == TestVerdict.FAILURE
        assert v.error_lines == {42}

    def test_parse_failure_generic_when_only_FAILED(self):
        v = parse_test_failure_py("FAILED test_mod.py::test_thing", "test_mod", "test_thing")
        assert v.status == TestVerdict.FAILURE
        assert v.error_lines == set()

    def test_parse_failure_returns_none_without_signal(self):
        assert parse_test_failure_py("nothing useful", "test_mod", "test_thing") is None

    def test_parse_invalid_syntax_error(self):
        assert parse_invalid_execution_py("E   SyntaxError: bad").status == TestVerdict.SYNTAX_ERR

    def test_parse_invalid_unknown(self):
        assert parse_invalid_execution_py("random noise").status == TestVerdict.UNKNOWN


# --------------------------------------------------------------------------- #
#  run_cmd                                                                     #
# --------------------------------------------------------------------------- #
class TestRunCmd:
    def test_returns_output_and_zero(self):
        rc, out = run_cmd(["python", "-c", "print('hello-out')"], timeout=30, cwd=".", env=None)
        assert rc == 0
        assert "hello-out" in out

    def test_captures_nonzero_exit(self):
        rc, _ = run_cmd(["python", "-c", "import sys; sys.exit(3)"], timeout=30, cwd=".", env=None)
        assert rc == 3

    def test_timeout_returns_124(self):
        start = time.time()
        rc, _ = run_cmd(["python", "-c", "import time; time.sleep(30)"], timeout=1, cwd=".", env=None)
        assert rc == 124
        assert time.time() - start < 15  # was actually killed, not waited out


# --------------------------------------------------------------------------- #
#  compile_and_run_test_python                                                 #
# --------------------------------------------------------------------------- #
class TestCompileAndRun:
    def _project(self, tmp_path, body):
        f = tmp_path / "test_mod.py"
        f.write_text(body, encoding="utf-8")
        return tmp_path

    def test_passing_test_is_success(self, tmp_path):
        proj = self._project(tmp_path, "def test_ok():\n    assert 1 + 1 == 2\n")
        v = compile_and_run_test_python(proj, "test_mod.py", "test_ok", tmp_path)
        assert v.status == TestVerdict.SUCCESS

    def test_failing_test_is_failure(self, tmp_path):
        proj = self._project(tmp_path, "def test_bad():\n    assert 1 + 1 == 3\n")
        v = compile_and_run_test_python(proj, "test_mod.py", "test_bad", tmp_path)
        assert v.status == TestVerdict.FAILURE

    def test_syntax_error_is_syntax_err(self, tmp_path):
        proj = self._project(tmp_path, "def test_syn(:\n    pass\n")
        v = compile_and_run_test_python(proj, "test_mod.py", "test_syn", tmp_path)
        assert v.status == TestVerdict.SYNTAX_ERR

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compile_and_run_test_python(tmp_path, "does_not_exist.py", "test_x", tmp_path)

    def test_timeout_is_timeout_verdict(self, tmp_path):
        proj = self._project(tmp_path, "import time\ndef test_slow():\n    time.sleep(30)\n")
        v = compile_and_run_test_python(proj, "test_mod.py", "test_slow", tmp_path, timeout=1)
        assert v.status == TestVerdict.TIMEOUT
