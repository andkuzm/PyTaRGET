"""Tests for data_processing/funnel_report.py."""
import csv
import json
import os
import subprocess
import sys

from data_processing import funnel_report

COLUMNS = ["repository_name", "bucket", "stage", "commits", "relevant_commits", "relevant_pairs", "visited_pairs",
           "pairs_with_changed_test", "changed_tests", "parent_success", "child_success", "cases", "details"]


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_ALL)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row.get(c, "") for c in COLUMNS])


def repo(name, stage):
    return {"repository_name": name, "bucket": "repo", "stage": stage}


def bucket(name, number, **counts):
    row = {"repository_name": name, "bucket": number, "stage": ""}
    row.update(counts)
    return row


ROWS = [
    repo("a/one", "started"),
    bucket("a/one", 0, commits=10, relevant_pairs=4, visited_pairs=4, changed_tests=5, parent_success=3,
           child_success=2, cases=1, details=json.dumps({"parent_failure": 2, "child_unknown": 1})),
    bucket("a/one", 9, commits=10, relevant_pairs=6, visited_pairs=6, changed_tests=5, parent_success=5,
           child_success=5, cases=4),
    repo("a/one", "searched"),
    repo("b/two", "started"), repo("b/two", "no_tests"),
    repo("c/three", "started"), repo("c/three", "failed"),
    repo("d/four", "started"),  # killed
    # a repository resumed after a kill: its second attempt writes rows again
    repo("a/one", "started"),
    bucket("a/one", 0, commits=0, relevant_pairs=2, visited_pairs=2, changed_tests=1, parent_success=0,
           details=json.dumps({"parent_failure": 1})),
    repo("a/one", "searched"),
]


def test_stages_count_how_each_repository_ended(tmp_path):
    write(tmp_path / "f.csv", ROWS)
    stages, buckets = funnel_report.summarize(funnel_report.load(tmp_path / "f.csv"))
    assert dict(stages) == {"searched": 1, "no_tests": 1, "failed": 1, "never finished": 1}


def test_counters_are_summed_over_repositories_and_attempts(tmp_path):
    write(tmp_path / "f.csv", ROWS)
    _, buckets = funnel_report.summarize(funnel_report.load(tmp_path / "f.csv"))
    assert buckets["0"]["relevant_pairs"] == 6 and buckets["0"]["changed_tests"] == 6
    assert buckets["0"]["parent_failure"] == 3 and buckets["0"]["child_unknown"] == 1
    assert buckets["9"]["cases"] == 4


def test_the_report_shows_the_table_totals_and_losses(tmp_path):
    write(tmp_path / "f.csv", ROWS)
    text = funnel_report.render(*funnel_report.summarize(funnel_report.load(tmp_path / "f.csv")))
    assert "never finished: 1" in text and "searched: 1" in text
    assert "CASES" in text and "parent_failure" in text
    assert "of 11 changed tests: 8 passed at the parent (73%), 7 also at the child, 5 became cases (45%)" in text
    assert "lost: parent_failure 3, child_unknown 1" in text


def test_an_empty_funnel_does_not_crash(tmp_path):
    write(tmp_path / "f.csv", [repo("a/one", "started")])
    text = funnel_report.render(*funnel_report.summarize(funnel_report.load(tmp_path / "f.csv")))
    assert "never finished: 1" in text and "no search rows yet" in text


def test_it_runs_as_a_plain_script_with_the_standard_library_only(tmp_path):
    write(tmp_path / "f.csv", ROWS)
    script = os.path.join(os.path.dirname(funnel_report.__file__), "funnel_report.py")
    result = subprocess.run([sys.executable, script, str(tmp_path / "f.csv")], capture_output=True, text=True)
    assert result.returncode == 0 and "CASES" in result.stdout


def test_reasons_are_ranked_and_do_not_become_table_columns(tmp_path):
    rows = [
        repo("a/one", "started"),
        bucket("a/one", 0, changed_tests=9, details=json.dumps({
            "parent_unknown": 9, "why:parent:missing_module:numpy": 6, "why:parent:setup_error": 2,
            "why:child:other": 1})),
        repo("a/one", "searched"),
    ]
    write(tmp_path / "f.csv", rows)
    text = funnel_report.render(*funnel_report.summarize(funnel_report.load(tmp_path / "f.csv")))
    header = text.splitlines()[3]
    assert "parent_unknown" in header and "why:" not in header
    ranking = text.split("why tests ended 'unknown'")[1]
    assert ranking.index("parent:missing_module:numpy") < ranking.index("parent:setup_error") < ranking.index("child:other")
    assert "     6  parent:missing_module:numpy" in ranking


REPO_ROWS = [
    repo("big/monorepo", "started"),
    bucket("big/monorepo", 3, relevant_pairs=40, visited_pairs=40, changed_tests=30, parent_success=0,
           details=json.dumps({"parent_unknown": 30, "why:parent:missing_module:langchain_tests": 28,
                               "why:parent:setup_error": 2})),
    bucket("big/monorepo", 9, relevant_pairs=10, visited_pairs=10, changed_tests=10, parent_success=0,
           details=json.dumps({"parent_unknown": 10, "why:parent:missing_module:langchain_tests": 9})),
    repo("big/monorepo", "searched"),
    repo("small/good", "started"),
    bucket("small/good", 5, relevant_pairs=6, visited_pairs=6, changed_tests=4, parent_success=4, child_success=3,
           cases=2),
    repo("small/good", "searched"),
    repo("old/format", "started"),  # a funnel written before the reasons were recorded
    bucket("old/format", 1, relevant_pairs=3, visited_pairs=3, changed_tests=2,
           details=json.dumps({"parent_failure": 2})),
    repo("old/format", "searched"),
]


def test_the_per_repository_view_ranks_by_tests_and_names_the_main_reason(tmp_path):
    write(tmp_path / "f.csv", REPO_ROWS)
    repos = funnel_report.summarize_repos(funnel_report.load(tmp_path / "f.csv"))
    lines = funnel_report.render_repos(repos).splitlines()
    assert lines[2].startswith("big/monorepo") and lines[3].startswith("small/good") and lines[4].startswith("old/format")
    assert "missing_module:langchain_tests (37)" in lines[2] and "50" in lines[2].split()  # planned 40 + 10
    assert lines[3].rstrip().endswith("-")  # nothing lost, no reason
    assert "parent_failure (2)" in lines[4]  # falls back to the verdict counter


def test_the_per_repository_view_works_from_the_command_line(tmp_path):
    write(tmp_path / "f.csv", REPO_ROWS)
    script = os.path.join(os.path.dirname(funnel_report.__file__), "funnel_report.py")
    result = subprocess.run([sys.executable, script, str(tmp_path / "f.csv"), "--repos"], capture_output=True, text=True)
    assert result.returncode == 0 and "big/monorepo" in result.stdout and "main reason" in result.stdout
    wrong = subprocess.run([sys.executable, script, str(tmp_path / "f.csv"), "--nonsense"], capture_output=True, text=True)
    assert wrong.returncode != 0
