"""
Plausibility run on tiny local git repositories (no network): parallel repository processes, cleanup, time limits
(a test that times out FAILS), shared control runs, resuming and the stripped environment of the tests.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from data_processing import plausibility_test as pt

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]

PASSING = "def test_value():\n    x = 1 + 1\n    assert x == 2\n"
SLOW = "import time\n\n\ndef test_slow():\n    time.sleep(60)\n    assert True\n"


def make_repo(root, name, source, filename="test_value.py"):
    path = Path(root) / name
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / filename).write_text(source, encoding="utf-8")
    subprocess.run(GIT + ["add", "-A"], cwd=path, check=True)
    subprocess.run(GIT + ["commit", "-qm", "repaired"], cwd=path, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()


def write_model(tmp_path, name, repo, sha, test, candidates, filename="test_value.py", reference="\n    assert x == 2"):
    row = {"repository_name": repo, "repaired_hash": sha, "relative_path": filename, "output": reference,
           "test_name": test, "annotated_code": "x" * 50, "outdated_test_log": "log"}
    split = tmp_path / f"{name}_test.json"
    split.write_text("[\n" + json.dumps(row) + "\n]", encoding="utf-8")        # the encoder's layout
    predictions = tmp_path / f"{name}_predictions.json"
    predictions.write_text(json.dumps([{"ID": 0, "target": reference, "preds": candidates}]), encoding="utf-8")
    return {"name": name, "test_json": split, "predictions_json": predictions, "out_path": tmp_path / f"{name}_plaus.json"}


@pytest.fixture
def tester_for(tmp_path):
    def make(**options):
        return pt.PlausibilityTester(tmp_path / "work", k=5, timeout=options.pop("timeout", 30), use_venv=False,
                                     repo_url=str(tmp_path / "git" / "{name}"), **options)
    return make


def tagged_processes(under=None):
    """Processes started by a plausibility run (they all carry PLAUSIBILITY_TAG); none may outlive the run. With
    `under`, only those whose environment mentions that directory (another run may be going on elsewhere)."""
    found = []
    needle = str(under).encode() if under else None
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                environment = (entry / "environ").read_bytes()
                if b"PLAUSIBILITY_TAG=" in environment and (needle is None or needle in environment):
                    found.append(entry.name)
            except OSError:
                pass
    return found


def wait_until_gone(under, timeout=15):
    deadline = time.time() + timeout
    while tagged_processes(under) and time.time() < deadline:
        time.sleep(0.5)                                   # a killed process takes a moment to disappear
    return tagged_processes(under)


def results_of(task):
    return json.loads(task["out_path"].read_text(encoding="utf-8"))["results"]["0"]


def test_candidates_are_judged_and_a_timeout_is_a_failure(tmp_path, tester_for):
    sha = make_repo(tmp_path / "git", "acme/proj", PASSING)
    task = write_model(tmp_path, "m1", "acme/proj", sha, "test_value", [
        "\n    assert x == 2",                     # the reference itself
        "\n    assert x == 3",                     # wrong
        "\n    time_out = 1\n    while True:\n        pass",   # hangs
        "",                                       # empty
        "\n    assert x > 1"])                    # a different but valid repair
    summary = tester_for(timeout=12).run_many([task])["m1"]
    result = results_of(task)
    assert result["status"] == "evaluated"
    assert result["verdicts"] == ["pass", "fail", "fail", "empty", "pass"]
    assert result["timeouts"] == 1                 # the hanging repair was counted as a failure
    assert summary["plausible@1"] == 100.0 and summary["candidate_timeouts"] == 1


def test_a_control_that_times_out_leaves_the_case_out(tmp_path, tester_for):
    sha = make_repo(tmp_path / "git", "acme/slow", SLOW, "test_slow.py")
    task = write_model(tmp_path, "m1", "acme/slow", sha, "test_slow", ["\n    assert True"],
                       filename="test_slow.py", reference="\n    assert True")
    tester_for(timeout=12).run_many([task])
    assert results_of(task)["status"] == "reference_fails" and results_of(task)["detail"] == "timeout"


def test_models_share_the_repository_and_the_workdir_is_cleaned(tmp_path, tester_for):
    sha = make_repo(tmp_path / "git", "acme/proj", PASSING)
    first = write_model(tmp_path, "m1", "acme/proj", sha, "test_value", ["\n    assert x == 3", "\n    assert x == 2"])
    second = write_model(tmp_path, "m2", "acme/proj", sha, "test_value", ["\n    assert x == 2"])
    tester = tester_for()
    tester.run_many([first, second])
    assert results_of(first)["verdicts"] == ["fail", "pass"] and results_of(second)["verdicts"] == ["pass"]
    assert list((tmp_path / "work").iterdir()) == []         # clones, environments and scratch are all gone
    # a second run has nothing left to do
    again = tester_for()
    again.run_many([first, second])
    assert results_of(first)["verdicts"] == ["fail", "pass"]


def test_repositories_run_in_parallel(tmp_path, tester_for):
    tasks = []
    for i in range(3):
        sha = make_repo(tmp_path / "git", f"acme/p{i}", PASSING)
        tasks.append(write_model(tmp_path, f"m{i}", f"acme/p{i}", sha, "test_value", ["\n    assert x == 2"]))
    summaries = tester_for(workers=3).run_many(tasks)
    assert all(s["evaluated"] == 1 and s["plausible@1"] == 100.0 for s in summaries.values())


def test_a_repository_over_its_time_limit_is_killed_and_recorded(tmp_path, tester_for):
    sha = make_repo(tmp_path / "git", "acme/slow", SLOW, "test_slow.py")
    task = write_model(tmp_path, "m1", "acme/slow", sha, "test_slow", ["\n    assert True"],
                       filename="test_slow.py", reference="\n    assert True")
    started = time.time()
    tester_for(timeout=120, repo_timeout=7).run_many([task])
    assert results_of(task)["status"] == "repo_timeout"
    assert time.time() - started < 40
    assert wait_until_gone(tmp_path) == []


def test_a_missing_repository_is_a_setup_failure_not_a_crash(tmp_path, tester_for):
    task = write_model(tmp_path, "m1", "acme/nothing", "0" * 40, "test_value", ["\n    assert x == 2"])
    tester_for().run_many([task])
    assert results_of(task)["status"] == "setup_failed"


def test_tests_run_without_the_callers_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "secret-token")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    env = pt.build_environment(tmp_path / "home")
    assert "GH_TOKEN" not in env and "HF_TOKEN" not in env
    assert env["HOME"] == str(tmp_path / "home") and env["TMPDIR"].startswith(str(tmp_path / "home"))


def test_the_test_run_sees_a_private_home_and_no_tokens(tmp_path, tester_for, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "secret-token")
    source = ("import os\n\n\ndef test_value():\n    x = 2\n    assert 'GH_TOKEN' not in os.environ\n"
              "    assert os.environ['HOME'].endswith('home')\n")
    sha = make_repo(tmp_path / "git", "acme/env", source)
    task = write_model(tmp_path, "m1", "acme/env", sha, "test_value", ["\n    x = 2"], reference="\n    x = 2")
    tester_for().run_many([task])
    assert results_of(task)["verdicts"] == ["pass"]


def test_rows_are_streamed_in_both_layouts(tmp_path):
    rows = [{"a": i, "text": "line\nbreak"} for i in range(3)]
    encoder_layout = tmp_path / "enc.json"
    encoder_layout.write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]", encoding="utf-8")
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    assert list(pt.iter_rows(encoder_layout)) == rows and list(pt.iter_rows(pretty)) == rows
    single = tmp_path / "one.json"
    single.write_text("[\n" + json.dumps(rows[0]) + "\n]", encoding="utf-8")
    assert list(pt.iter_rows(single)) == rows[:1]


def test_an_interrupted_run_stops_its_repositories_and_cleans_up(tmp_path):
    import os
    import signal
    sha = make_repo(tmp_path / "git", "acme/slow", SLOW, "test_slow.py")
    task = write_model(tmp_path, "m1", "acme/slow", sha, "test_slow", ["\n    assert True"],
                       filename="test_slow.py", reference="\n    assert True")
    command = [sys.executable, "-u", "-m", "data_processing.plausibility_test", str(task["test_json"]),
               str(task["predictions_json"]), "--out", str(task["out_path"]), "--workdir", str(tmp_path / "work"),
               "--no-venv", "--workers", "1", "--timeout", "120", "--repo-url", str(tmp_path / "git" / "{name}")]
    run = subprocess.Popen(command, cwd=pt.ROOT, env=dict(os.environ, PYTHONPATH=str(pt.ROOT)),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 40
    while time.time() < deadline and not list((tmp_path / "work").glob("run_*/acme__slow_*/work/repos/*")):
        time.sleep(0.5)                                   # the test is being run
    run.send_signal(signal.SIGINT)
    run.communicate(timeout=40)
    assert run.returncode != 0
    assert list((tmp_path / "work").iterdir()) == []
    assert wait_until_gone(tmp_path) == []
    saved = json.loads(task["out_path"].read_text(encoding="utf-8"))["results"]
    assert saved == {}                                    # interrupted cases get no result: they run again next time


def test_a_project_whose_addopts_hide_the_summary_still_passes(tmp_path, tester_for):
    sha = make_repo(tmp_path / "git", "acme/quiet", PASSING)
    repo = tmp_path / "git" / "acme" / "quiet"
    (repo / "pytest.ini").write_text("[pytest]\naddopts = -q\n", encoding="utf-8")      # -q -q: no "1 passed" line
    subprocess.run(GIT + ["add", "-A"], cwd=repo, check=True)
    subprocess.run(GIT + ["commit", "-qm", "quiet"], cwd=repo, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    task = write_model(tmp_path, "m1", "acme/quiet", sha, "test_value", ["\n    assert x == 3", "\n    assert x == 2"])
    tester_for().run_many([task])
    assert results_of(task)["status"] == "evaluated" and results_of(task)["verdicts"] == ["fail", "pass"]


def test_verdicts_from_the_junit_file():
    suite = '<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="%d" tests="%d">'
    assert pt.verdict_from(0, "", suite % (0, 1)) == "pass"
    assert pt.verdict_from(0, "", suite % (1, 1)) == "fail"        # skipped: nothing was checked
    assert pt.verdict_from(0, "", suite % (0, 0)) == "fail"        # nothing collected
    assert pt.verdict_from(1, "", None) == "fail" and pt.verdict_from(2, "", None) == "error"
    assert pt.verdict_from(124, "", None) == "timeout"


def test_missing_packages_come_only_from_the_known_lists(tmp_path):
    tester = pt.PlausibilityTester(tmp_path)
    output = ("ModuleNotFoundError: No module named 'yaml'\nModuleNotFoundError: No module named 'cv2.dnn'\n"
              "ModuleNotFoundError: No module named 'totally_unknown_pkg'\nERROR: Unknown config option: timeout\n"
              "ERROR: Unknown config option: some_other_option")
    assert tester.missing_packages("py", output) == ["pyyaml", "opencv-python-headless", "pytest-timeout"]
    assert tester.missing_packages("py", "async def functions are not natively supported.") == ["pytest-asyncio"]
    tester.installed["py"] = {"pyyaml"}
    assert "pyyaml" not in tester.missing_packages("py", output)


def test_cases_with_a_retry_status_run_again(tmp_path, tester_for):
    sha = make_repo(tmp_path / "git", "acme/proj", PASSING)
    task = write_model(tmp_path, "m1", "acme/proj", sha, "test_value", ["\n    assert x == 3"])
    task["out_path"].write_text(json.dumps({"summary": {}, "results": {
        "0": {"status": "reference_fails", "nodeid": None, "verdicts": [], "detail": "error"}}}), encoding="utf-8")
    tester_for().run_many([task])                        # a finished status is kept
    assert results_of(task)["status"] == "reference_fails"
    tester_for().run_many([task], retry_statuses=["reference_fails"])
    assert results_of(task)["status"] == "evaluated" and results_of(task)["verdicts"] == ["fail"]
