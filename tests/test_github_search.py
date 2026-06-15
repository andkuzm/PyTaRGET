"""Tests for repository_search/GitHubSearch.py.

All network (`requests`), sleeping (`time.sleep`), subprocess, multiprocessing
and venv interactions are stubbed so the search/orchestration logic is tested
deterministically and offline.
"""

import os

import pytest

import GitHubSearch
from GitHubSearch import GitHubSearch as GHS


# --------------------------------------------------------------------------- #
#  Fakes                                                                      #
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, links=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.links = links or {}

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Never actually sleep in any GitHubSearch test."""
    monkeypatch.setattr(GitHubSearch.time, "sleep", lambda *_a, **_k: None)


def make_searcher(tmp_path, version="v1"):
    return GHS(github_token="tok", version=version, cwd=str(tmp_path))


# --------------------------------------------------------------------------- #
#  __init__ path derivation                                                   #
# --------------------------------------------------------------------------- #
class TestInit:
    def test_version_mode_paths(self, tmp_path):
        s = make_searcher(tmp_path, "exp")
        assert s.output_csv == tmp_path / "annotated_cases_exp.csv"
        assert s.processed_file == tmp_path / "processed_repositories_exp.txt"
        assert s.repository_path == str(tmp_path / "repos_exp")

    def test_deprecated_mode_warns(self, tmp_path):
        with pytest.warns(DeprecationWarning):
            s = GHS(github_token="tok", out_path=str(tmp_path), repository_path="rp")
        assert s.output_csv == tmp_path / "annotated_cases.csv"


# --------------------------------------------------------------------------- #
#  get_latest_commit                                                          #
# --------------------------------------------------------------------------- #
class TestGetLatestCommit:
    def test_returns_sha(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(GitHubSearch.requests, "get",
                            lambda *a, **k: FakeResponse(200, [{"sha": "abc123"}]))
        assert s.get_latest_commit("o/r") == "abc123"

    def test_empty_on_non_200(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(GitHubSearch.requests, "get",
                            lambda *a, **k: FakeResponse(404, {}))
        assert s.get_latest_commit("o/r") == ""

    def test_empty_on_empty_list(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(GitHubSearch.requests, "get",
                            lambda *a, **k: FakeResponse(200, []))
        assert s.get_latest_commit("o/r") == ""


# --------------------------------------------------------------------------- #
#  _rate_limit_sleep                                                          #
# --------------------------------------------------------------------------- #
class TestRateLimitSleep:
    def test_long_wait_when_exhausted(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        slept = []
        monkeypatch.setattr(GitHubSearch.time, "sleep", lambda n: slept.append(n))
        monkeypatch.setattr(GitHubSearch.time, "time", lambda: 1000)
        resp = FakeResponse(headers={"X-RateLimit-Remaining": "1", "X-RateLimit-Reset": "1010"})
        s._rate_limit_sleep(resp)
        assert slept and slept[0] >= 10

    def test_short_wait_when_plenty(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        slept = []
        monkeypatch.setattr(GitHubSearch.time, "sleep", lambda n: slept.append(n))
        resp = FakeResponse(headers={"X-RateLimit-Remaining": "100"})
        s._rate_limit_sleep(resp)
        assert slept == [2]


# --------------------------------------------------------------------------- #
#  _github_search_request                                                     #
# --------------------------------------------------------------------------- #
class TestGithubSearchRequest:
    def test_returns_on_200(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        ok = FakeResponse(200, {"total_count": 0})
        monkeypatch.setattr(GitHubSearch.requests, "get", lambda *a, **k: ok)
        assert s._github_search_request({}, {}) is ok

    def test_retries_after_rate_limit_then_succeeds(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        seq = [FakeResponse(403, headers={"Retry-After": "1"}),
               FakeResponse(200, {"total_count": 1})]
        calls = iter(seq)
        monkeypatch.setattr(GitHubSearch.requests, "get", lambda *a, **k: next(calls))
        result = s._github_search_request({}, {})
        assert result.status_code == 200

    def test_returns_none_after_exhausting_retries(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(GitHubSearch.requests, "get",
                            lambda *a, **k: FakeResponse(500, {}))
        assert s._github_search_request({}, {}, retries=2) is None


# --------------------------------------------------------------------------- #
#  _search_size_range                                                         #
# --------------------------------------------------------------------------- #
class TestSearchSizeRange:
    def test_zero_total_count_processes_nothing(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        s._run_count = 0
        monkeypatch.setattr(s, "_github_search_request",
                            lambda h, p, **k: FakeResponse(200, {"total_count": 0}))
        processed = set()
        called = []
        monkeypatch.setattr(s, "process_repository_with_timeout",
                            lambda fn: called.append(fn))
        s._search_size_range("q", 0, 100, {}, processed)
        assert called == []

    def test_splits_range_when_over_1000(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        s._run_count = 0
        queries = []

        def fake_request(headers, params, **k):
            q = params["q"]
            queries.append(q)
            # Only the original full range is "too big"; sub-ranges are empty.
            if "size:>=0 size:<100" in q:
                return FakeResponse(200, {"total_count": 1000, "items": []})
            return FakeResponse(200, {"total_count": 0})

        monkeypatch.setattr(s, "_github_search_request", fake_request)
        monkeypatch.setattr(s, "process_repository_with_timeout", lambda fn: True)
        s._search_size_range("q", 0, 100, {}, set())

        # Recursion split into [0,50) and [50,100).
        assert any("size:>=0 size:<50" in q for q in queries)
        assert any("size:>=50 size:<100" in q for q in queries)

    def test_processes_items_and_writes_blacklist(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        s._run_count = 0
        resp = FakeResponse(200, {
            "total_count": 2,
            "items": [{"full_name": "o/already"}, {"full_name": "o/new"}],
        }, links={})  # no 'next' -> single page
        monkeypatch.setattr(s, "_github_search_request", lambda h, p, **k: resp)

        processed_calls = []
        monkeypatch.setattr(s, "process_repository_with_timeout",
                            lambda fn: processed_calls.append(fn) or True)
        monkeypatch.setattr(s, "get_latest_commit", lambda fn: "sha-" + fn.split("/")[-1])
        monkeypatch.setattr(s, "run_pytest_check", lambda fn: None)

        processed = {"o/already"}  # pre-seeded -> should be skipped
        s._search_size_range("q", 0, 100, {}, processed)

        # Only the not-yet-seen repo is processed.
        assert processed_calls == ["o/new"]
        assert "o/new" in processed
        assert s._run_count == 1

        blacklist = s.processed_file.read_text(encoding="utf-8")
        assert "o/new|sha-new" in blacklist


# --------------------------------------------------------------------------- #
#  find_and_process_repositories (resume / blacklist reading)                 #
# --------------------------------------------------------------------------- #
class TestFindAndProcess:
    def test_resumes_from_blacklist(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        s.processed_file.write_text("o/old|deadbeef\n", encoding="utf-8")

        captured = {}

        def fake_range(query_base, ss, se, headers, processed_repos):
            captured["processed"] = set(processed_repos)

        monkeypatch.setattr(s, "_search_size_range", fake_range)
        s.find_and_process_repositories()

        assert "o/old" in captured["processed"]
        assert s._run_count == 0

    def test_counts_found_cases_from_csv(self, tmp_path, monkeypatch, capsys):
        s = make_searcher(tmp_path)
        s.output_csv.write_text("header\nrow1\nrow2\n", encoding="utf-8")
        monkeypatch.setattr(s, "_search_size_range", lambda *a, **k: None)
        s.find_and_process_repositories()
        out = capsys.readouterr().out
        assert "found 2 test cases" in out


# --------------------------------------------------------------------------- #
#  run_pytest_check / reinstall_pytest / run_pytest_trace                     #
# --------------------------------------------------------------------------- #
class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestPytestCheck:
    def test_ok_returncode_returns_quietly(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(s, "run_pytest_trace", lambda: _Proc(0))
        assert s.run_pytest_check("o/r") is None

    def test_no_tests_returncode_5_is_ok(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(s, "run_pytest_trace", lambda: _Proc(5))
        assert s.run_pytest_check("o/r") is None

    def test_persistent_failure_exits(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(s, "run_pytest_trace", lambda: _Proc(1, "out", "err"))
        monkeypatch.setattr(s, "reinstall_pytest", lambda: None)
        with pytest.raises(SystemExit):
            s.run_pytest_check("o/r")

    def test_reinstall_runs_uninstall_then_install(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        cmds = []
        monkeypatch.setattr(GitHubSearch.subprocess, "run",
                            lambda cmd, **k: cmds.append(cmd) or _Proc(0))
        s.reinstall_pytest()
        assert any("uninstall" in c for c in cmds)
        assert any("install" in c for c in cmds)

    def test_run_pytest_trace_invokes_subprocess(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        sentinel = _Proc(0)
        monkeypatch.setattr(GitHubSearch.subprocess, "run", lambda *a, **k: sentinel)
        assert s.run_pytest_trace() is sentinel


# --------------------------------------------------------------------------- #
#  process_repository_with_timeout                                            #
# --------------------------------------------------------------------------- #
class FakeProcess:
    def __init__(self, target, args, alive=False, exitcode=0):
        self.target = target
        self.args = args
        self._alive = alive
        self.exitcode = exitcode
        self.started = False

    def start(self):
        self.started = True

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False


class TestProcessRepositoryWithTimeout:
    def _patch_common(self, monkeypatch, tmp_path):
        monkeypatch.setattr(GitHubSearch, "create_virtualenv", lambda p: None)
        monkeypatch.setattr(GitHubSearch.tempfile, "mkdtemp",
                            lambda **k: str(tmp_path / "venv"))
        monkeypatch.setattr(GitHubSearch.shutil, "rmtree", lambda *a, **k: None)

    def test_success(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        self._patch_common(monkeypatch, tmp_path)
        monkeypatch.setattr(GitHubSearch.multiprocessing, "Process",
                            lambda target, args: FakeProcess(target, args, alive=False, exitcode=0))
        assert s.process_repository_with_timeout("o/r") is True

    def test_nonzero_exit_is_failure(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        self._patch_common(monkeypatch, tmp_path)
        monkeypatch.setattr(GitHubSearch.multiprocessing, "Process",
                            lambda target, args: FakeProcess(target, args, alive=False, exitcode=1))
        assert s.process_repository_with_timeout("o/r") is False

    def test_timeout_is_failure(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        self._patch_common(monkeypatch, tmp_path)
        monkeypatch.setattr(GitHubSearch.multiprocessing, "Process",
                            lambda target, args: FakeProcess(target, args, alive=True, exitcode=None))
        assert s.process_repository_with_timeout("o/r") is False


# --------------------------------------------------------------------------- #
#  module-level helpers                                                       #
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    def test_create_virtualenv_uses_envbuilder(self, monkeypatch):
        captured = {}

        class FakeBuilder:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def create(self, path):
                captured["path"] = path

        monkeypatch.setattr(GitHubSearch.venv, "EnvBuilder", FakeBuilder)
        GitHubSearch.create_virtualenv("/some/venv")
        assert captured["kwargs"] == {"with_pip": True, "clear": True}
        assert captured["path"] == "/some/venv"

    def test_run_processor_in_venv_builds_command_and_env(self, monkeypatch):
        run_calls = []
        check_calls = []
        monkeypatch.setattr(GitHubSearch.subprocess, "run",
                            lambda cmd, **k: run_calls.append((cmd, k)))
        monkeypatch.setattr(GitHubSearch.subprocess, "check_call",
                            lambda cmd, **k: check_calls.append((cmd, k)))

        GitHubSearch.run_processor_in_venv("o/r", "/rp", "/out.csv", "/venv")

        # pip install pytest/coverage happened.
        assert any("install" in cmd for cmd, _ in run_calls)
        # miner invoked with the right positional args.
        miner_cmd, miner_kwargs = check_calls[0]
        assert "o/r" in miner_cmd and "/rp" in miner_cmd and "/out.csv" in miner_cmd
        assert miner_cmd[0].endswith(os.path.join("bin", "python")) or \
               miner_cmd[0].endswith(os.path.join("Scripts", "python"))
        # PYTHONPATH points at the project root.
        assert "PYTHONPATH" in miner_kwargs["env"]
