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
        rev = GitHubSearch.PIPELINE_FIX_REVISION
        assert s.output_csv == tmp_path / f"annotated_cases_exp_{rev}.csv"
        assert s.processed_file == tmp_path / f"processed_repositories_exp_{rev}.txt"
        assert s.repository_path == str(tmp_path / f"repos_exp_{rev}")

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


# --------------------------------------------------------------------------- #
#  progress: what a finished, killed or crashed attempt means                 #
# --------------------------------------------------------------------------- #
import processed_log


class TestProcessAndRecord:
    def _searcher(self, tmp_path, monkeypatch, miner_effect, success=True):
        s = make_searcher(tmp_path)
        monkeypatch.setattr(s, "process_repository_with_timeout",
                            lambda name: (miner_effect(s), success)[1])
        monkeypatch.setattr(s, "get_latest_commit", lambda name: "latest")
        return s

    def test_finished_repository_stays_as_the_miner_marked_it(self, tmp_path, monkeypatch):
        s = self._searcher(tmp_path, monkeypatch, lambda s: processed_log.mark_done(s.processed_file, "o/r", "head"))
        s._process_and_record("o/r")
        assert s.processed_file.read_text().splitlines() == ["o/r|head"]

    def test_miner_that_never_wrote_anything_is_recorded_with_the_latest_commit(self, tmp_path, monkeypatch):
        s = self._searcher(tmp_path, monkeypatch, lambda s: None, success=False)
        s._process_and_record("o/r")
        assert s.processed_file.read_text().splitlines() == ["o/r|latest"]

    def test_killed_after_progress_is_left_for_the_next_run(self, tmp_path, monkeypatch):
        s = self._searcher(tmp_path, monkeypatch,
                           lambda s: processed_log.mark_progress(s.processed_file, "o/r", "head", "commit7"),
                           success=False)
        s._process_and_record("o/r")
        assert processed_log.resume_state(s.processed_file, "o/r").marker == "commit7"

    def test_killed_without_progress_is_given_up(self, tmp_path, monkeypatch):
        s = self._searcher(tmp_path, monkeypatch,
                           lambda s: processed_log.mark_started(s.processed_file, "o/r", "head"), success=False)
        s._process_and_record("o/r")
        assert s.processed_file.read_text().splitlines() == ["o/r|head"]

    def test_a_second_attempt_that_got_no_further_is_given_up(self, tmp_path, monkeypatch):
        s = self._searcher(tmp_path, monkeypatch, lambda s: None, success=False)
        processed_log.mark_progress(s.processed_file, "o/r", "head", "commit7")
        s._process_and_record("o/r")  # still at commit7 afterwards
        assert s.processed_file.read_text().splitlines() == ["o/r|head"]

    def test_a_second_attempt_that_advanced_stays_in_progress(self, tmp_path, monkeypatch):
        s = self._searcher(tmp_path, monkeypatch,
                           lambda s: processed_log.mark_progress(s.processed_file, "o/r", "head", "commit9"),
                           success=False)
        processed_log.mark_progress(s.processed_file, "o/r", "head", "commit7")
        s._process_and_record("o/r")
        assert processed_log.resume_state(s.processed_file, "o/r").marker == "commit9"


class TestUnfinishedRepositoriesAreResumedFirst:
    def test_unfinished_ones_run_before_the_search_and_are_not_run_twice(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)
        s.processed_file.write_text("o/done|d\no/busy|head|commit3\n", encoding="utf-8")
        ran = []
        monkeypatch.setattr(s, "_process_and_record", lambda name: ran.append(("resume", name)))
        monkeypatch.setattr(s, "run_pytest_check", lambda name: None)

        def fake_search(query_base, size_start, size_end, headers, processed_repos):
            for name in ("o/done", "o/busy", "o/new"):
                if name not in processed_repos:
                    ran.append(("search", name))
        monkeypatch.setattr(s, "_search_size_range", fake_search)

        s.find_and_process_repositories()
        resumed = [name for kind, name in ran if kind == "resume"]
        searched = {name for kind, name in ran if kind == "search"}
        assert resumed == ["o/busy"] and ran[0] == ("resume", "o/busy")
        assert searched == {"o/new"}


# --------------------------------------------------------------------------- #
#  several repositories at the same time                                      #
# --------------------------------------------------------------------------- #
import threading


def make_parallel(tmp_path, monkeypatch, workers, fake_process):
    s = GHS(github_token="tok", version="par", cwd=str(tmp_path), workers=workers)
    monkeypatch.setattr(s, "process_repository_with_timeout", fake_process)
    monkeypatch.setattr(s, "run_pytest_check", lambda name: None)
    monkeypatch.setattr(s, "get_latest_commit", lambda name: "latest")
    s._run_count = 0
    return s


class TestWorkers:
    def test_repositories_run_at_the_same_time_each_in_its_own_directory(self, tmp_path, monkeypatch):
        lock, running, seen = threading.Lock(), [], []
        peak, all_busy = [0], threading.Event()

        def fake(name, repository_path=None):
            with lock:
                running.append(name)
                seen.append((name, repository_path, sorted(running)))
                peak[0] = max(peak[0], len(running))
                if len(running) == 3:
                    all_busy.set()
            all_busy.wait(10)  # nobody finishes before three are inside: proves they overlap
            with lock:
                running.remove(name)
            return True

        s = make_parallel(tmp_path, monkeypatch, 3, fake)
        s._start_pool()
        for n in range(7):
            s._dispatch(f"owner{n}/utils")  # same last name on purpose
        s._finish_pool()

        assert peak[0] == 3
        assert s._run_count == 7
        assert {name for name, _, _ in seen} == {f"owner{n}/utils" for n in range(7)}
        # every repository got a worker directory, and workers running together never share one
        assert all(path is not None and os.path.basename(path).startswith("slot") for _, path, _ in seen)
        directory = {name: path for name, path, _ in seen}
        for _, _, together in seen:  # whoever was running at the same moment had a directory of its own
            assert len({directory[n] for n in together}) == len(together)
        assert (tmp_path / "repos_par_pr58" / "slot0").is_dir()
        # all seven lines ended up in the processed file even though they were written concurrently
        names = {line.split("|")[0] for line in s.processed_file.read_text().splitlines()}
        assert names == {f"owner{n}/utils" for n in range(7)}

    def test_one_worker_runs_in_place_without_a_slot_directory(self, tmp_path, monkeypatch):
        calls = []
        s = make_parallel(tmp_path, monkeypatch, 1, lambda name, *rest: calls.append((name, rest)) or True)
        s._start_pool()
        s._dispatch("o/a")
        s._dispatch("o/b")
        s._finish_pool()
        assert calls == [("o/a", ()), ("o/b", ())]
        assert s._pool is None

    def test_the_repository_limit_stops_the_dispatching(self, tmp_path, monkeypatch):
        s = GHS(github_token="tok", version="lim", cwd=str(tmp_path), max_repositories=2)
        done = []
        monkeypatch.setattr(s, "_process_and_record", lambda name, *rest: done.append(name))
        monkeypatch.setattr(s, "run_pytest_check", lambda name: None)
        monkeypatch.setattr(s, "_search_size_range",
                            lambda query, a, b, headers, processed: [s._dispatch(f"o/r{i}") for i in range(5)])
        s.find_and_process_repositories()
        assert done == ["o/r0", "o/r1"]

    def test_a_broken_environment_stops_the_search(self, tmp_path, monkeypatch):
        s = make_parallel(tmp_path, monkeypatch, 2, lambda name, repository_path=None: True)

        def give_up(name):
            raise SystemExit(1)
        monkeypatch.setattr(s, "run_pytest_check", give_up)
        s._start_pool()
        s._dispatch("o/a")
        s._finish_pool()
        with pytest.raises(GitHubSearch.StopSearch):
            s._dispatch("o/b")

    def test_a_worker_that_fails_does_not_take_the_others_down(self, tmp_path, monkeypatch):
        def fake(name, repository_path=None):
            if name == "o/bad":
                raise RuntimeError("boom")
            return True
        s = make_parallel(tmp_path, monkeypatch, 2, fake)
        s._start_pool()
        for name in ("o/bad", "o/good1", "o/good2"):
            s._dispatch(name)
        s._finish_pool()
        assert s._run_count == 2


class TestProcessGroups:
    def test_a_timeout_kills_the_whole_process_group(self, tmp_path, monkeypatch):
        killed = []
        monkeypatch.setattr(GitHubSearch.os, "killpg", lambda pgid, sig: killed.append(pgid), raising=False)
        s = make_searcher(tmp_path)
        monkeypatch.setattr(GitHubSearch, "create_virtualenv", lambda p: None)
        monkeypatch.setattr(GitHubSearch.tempfile, "mkdtemp", lambda **k: str(tmp_path / "venv"))
        monkeypatch.setattr(GitHubSearch.shutil, "rmtree", lambda *a, **k: None)

        class Hung(FakeProcess):
            pid = 4242

        monkeypatch.setattr(GitHubSearch.multiprocessing, "Process",
                            lambda target, args: Hung(target, args, alive=True, exitcode=None))
        assert s.process_repository_with_timeout("o/r") is False
        assert 4242 in killed
        assert s._active == set()

    def test_the_worker_entry_leads_its_own_group_and_then_runs_the_miner(self, monkeypatch):
        events = []
        monkeypatch.setattr(GitHubSearch.os, "setsid", lambda: events.append("setsid"), raising=False)
        monkeypatch.setattr(GitHubSearch, "run_processor_in_venv", lambda *a: events.append(("run", a)))
        GitHubSearch._worker_entry("o/r", "/rp", "/out.csv", "/venv", "/processed.txt")
        assert events == ["setsid", ("run", ("o/r", "/rp", "/out.csv", "/venv", "/processed.txt"))]

    def test_killing_a_group_that_is_already_gone_is_harmless(self, monkeypatch):
        def gone(pgid, sig):
            raise ProcessLookupError()
        monkeypatch.setattr(GitHubSearch.os, "killpg", gone, raising=False)

        class P:
            pid = 1
        GitHubSearch._kill_process_group(P())
        GitHubSearch._kill_process_group(object())  # no pid at all


class TestImportedAsAPackage:
    def test_the_searcher_can_be_imported_the_way_a_notebook_does(self, tmp_path):
        # "from repository_search.GitHubSearch import GitHubSearch" with only the project root on the path;
        # its sibling module processed_log must be found without repository_search/ being on the path
        import subprocess
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = ("import sys; sys.path.insert(0, %r); "
                "from repository_search.GitHubSearch import GitHubSearch; "
                "s = GitHubSearch(github_token='t', version='v', cwd=%r, workers=2); print(s.workers)"
                % (root, str(tmp_path)))
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("2")


class TestInterruption:
    def test_an_interrupted_repository_is_not_closed_as_failed(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)

        def killed_before_any_progress(name, *rest):
            processed_log.mark_started(s.processed_file, name, "head")
            s._interrupted = True
            return False
        monkeypatch.setattr(s, "process_repository_with_timeout", killed_before_any_progress)
        s._process_and_record("o/r")
        # without the interruption this would count as "no progress" and be closed
        assert processed_log.resume_state(s.processed_file, "o/r") == processed_log.Entry("o/r", "head", "", True)

    def test_a_repository_the_miner_never_reached_stays_unwritten_when_interrupted(self, tmp_path, monkeypatch):
        s = make_searcher(tmp_path)

        def killed_at_once(name, *rest):
            s._interrupted = True
            return False
        monkeypatch.setattr(s, "process_repository_with_timeout", killed_at_once)
        s._process_and_record("o/r")
        assert not s.processed_file.exists() or "o/r" not in s.processed_file.read_text()

    def test_stopping_the_workers_mid_run_keeps_every_unfinished_repository_resumable(self, tmp_path, monkeypatch):
        s = GHS(github_token="tok", version="intr", cwd=str(tmp_path), workers=3)
        started, release = threading.Barrier(4, timeout=10), threading.Event()

        def fake(name, repository_path=None):
            processed_log.mark_started(s.processed_file, name, "head-" + name[-1])
            started.wait()          # three workers and the test are inside
            release.wait(10)        # "running" until the user interrupts
            return False            # killed
        monkeypatch.setattr(s, "process_repository_with_timeout", fake)
        monkeypatch.setattr(s, "run_pytest_check", lambda name: None)
        monkeypatch.setattr(s, "get_latest_commit", lambda name: "latest")
        s._run_count = 0
        s._start_pool()
        for n in range(3):
            s._dispatch(f"o/r{n}")
        started.wait()
        s._stop_workers()           # what Ctrl-C does
        release.set()
        s._finish_pool()

        entries = processed_log.read_entries(s.processed_file)
        assert sorted(entries) == ["o/r0", "o/r1", "o/r2"]
        assert all(entry.in_progress for entry in entries.values())
