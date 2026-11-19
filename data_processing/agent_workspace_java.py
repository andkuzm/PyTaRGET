"""
Leak-safe workspaces for the agentic repair of Java tests (TaRBench rows), the Java counterpart of agent_workspace.py.

State given to the agent = the project at the AFTER commit (aCommit: the source change is in place) with EVERY test file that the repair changed put back to its
BEFORE version (bCommit); test files added by the repair are removed. The target test method is then genuinely broken and other tests fixed in the same commit
cannot be copied (the "sibling-safe" mode of the Python workspaces). The directory is a fresh single-commit repository without history or remote.

Scoring: the agent's final METHOD text replaces the whole method of the AFTER file (derived row: hunk.targetChanges = all lines of aSource), so
data_processing/java_plausibility.py runs it unchanged (reference control included).
"""
import re
import shutil
import subprocess
from pathlib import Path

GIT_USER = ["-c", "user.name=agent", "-c", "user.email=agent@example.invalid"]
TEST_PATH = re.compile(r"(^|/)(src/test|test|tests|testing)/|(^|/)[^/]*(Test|Tests|IT|TestCase)\.java$|(^|/)[^/]*Test[^/]*\.java$")
CACHE_PATH = re.compile(r"(^|/)(target|\.mvn|\.idea|__pycache__)(/|$)|\.class$|\.log$")


def _run(args, cwd=None, timeout=1800):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def is_test_file(path):
    return bool(TEST_PATH.search(path))


def clone_repository(repo, commit, dest):
    done = _run(["git", "clone", "--quiet", "--filter=blob:none", f"https://github.com/{repo}.git", str(dest)])
    if done.returncode != 0:
        raise RuntimeError(f"clone failed: {done.stderr.strip()[-200:]}")
    done = _run(["git", "checkout", "--quiet", "--force", commit], cwd=dest)
    if done.returncode != 0:
        _run(["git", "fetch", "--quiet", "origin", commit], cwd=dest)
        done = _run(["git", "checkout", "--quiet", "--force", commit], cwd=dest)
        if done.returncode != 0:
            raise RuntimeError(f"checkout failed: {done.stderr.strip()[-200:]}")


def method_range(source):
    """(first line, last line) 1-based of a TaRBench aSource/bSource method: startLine and the number of lines of its code."""
    first = int(source["startLine"])
    return first, first + len(source["code"].split("\n")) - 1


def test_class_and_method(row):
    """('GrpcChannelPoolTest', 'testGroupSize') from 'alluxio.grpc.GrpcChannelPoolTest.testGroupSize()'."""
    parts = row["name"].split("(")[0].split(".")
    return parts[-2], parts[-1]


def prepare_workspace(row, dest, clone=None):
    """Build the agent's directory: after-commit tree, every changed test file back at its before version, history removed. Returns the case info."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    if row["aPath"] != row["bPath"]:
        raise ValueError("the test file was moved by the repair (aPath != bPath): not supported")
    (clone or clone_repository)(row["ID"].split(":")[0], row["aCommit"], dest)
    try:
        names = _run(["git", "diff", "--name-status", row["bCommit"], "HEAD"], cwd=dest).stdout.splitlines()
        for line in names:
            parts = line.split("\t")
            status, path = parts[0][0], parts[-1]
            if not path.endswith(".java") or not is_test_file(path):
                continue
            if status == "A":
                _run(["git", "rm", "-q", "-f", path], cwd=dest)
            else:
                done = _run(["git", "checkout", row["bCommit"], "--", path], cwd=dest)
                if done.returncode != 0:
                    raise RuntimeError(f"could not restore {path}: {done.stderr.strip()[-150:]}")
        text = (dest / row["bPath"]).read_text(encoding="utf-8", errors="ignore").split("\n")
        first, last = method_range(row["bSource"])
        if "\n".join(text[first - 1:last]).strip() != row["bSource"]["code"].strip():
            raise ValueError("the before method is not at its recorded lines in the reverted file")
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    shutil.rmtree(dest / ".git")
    _run(["git", "init", "--quiet"], cwd=dest)
    _run(["git", "add", "-A"], cwd=dest)
    _run(["git", *GIT_USER, "commit", "--quiet", "-m", "snapshot: failing test"], cwd=dest)
    cls, method = test_class_and_method(row)
    return {"test_file": row["bPath"], "class": cls, "method": method, "method_lines": [first, last]}


def task_prompt(row, command):
    cls, method = test_class_and_method(row)
    return (
        f"The test method `{method}` in `{row['bPath']}` fails (it does not compile or it fails) because the project's source code was changed. "
        "Repair the TEST so that it passes again with the current source code. Rules:\n"
        f"- Edit ONLY the test method `{method}` in `{row['bPath']}`; do not change any other file or method.\n"
        "- Do not weaken the test: keep what it checks (assertions) and adapt it to the new behaviour of the code.\n"
        f"- You may read any file and run the test with exactly this command (it works offline): `{command}`\n"
        "- Do not use git history, the network or dependency downloads; everything you need is in this directory.\n"
        "When the test passes, stop."
    )


def collect_result(row, dest):
    """What the agent left: the final method text (the lines between the unchanged text before and after the method), touched files, outside edits."""
    dest = Path(dest)
    status = [l for l in _run(["git", "status", "--porcelain", "-uall"], cwd=dest).stdout.splitlines() if not CACHE_PATH.search(l[3:])]
    new_files = sorted(l[3:] for l in status if l.startswith("??"))
    modified = sorted(l[3:] for l in status if not l.startswith("??"))
    other = [m for m in modified if m != row["bPath"]]
    before = _run(["git", "show", f"HEAD:{row['bPath']}"], cwd=dest).stdout.split("\n")
    path = dest / row["bPath"]
    after = path.read_text(encoding="utf-8", errors="ignore").split("\n") if path.exists() else []
    first, last = method_range(row["bSource"])
    head, tail = before[:first - 1], before[last:]
    result = {"touched_files": sorted(modified + new_files), "other_files_touched": other, "new_files": new_files, "method": None,
              "method_found": False, "test_file_changed_outside_method": False}
    if not after or len(after) < len(head) + len(tail) or after[:len(head)] != head or (tail and after[len(after) - len(tail):] != tail):
        result["test_file_changed_outside_method"] = bool(after)
        return result
    result["method"] = "\n".join(after[len(head):len(after) - len(tail)])
    result["method_found"] = bool(result["method"].strip())
    return result


def derived_row(row):
    """The row for scoring: the target lines are ALL lines of the after method, so the candidate (a whole method) replaces the whole method."""
    first, last = method_range(row["aSource"])
    new = dict(row)
    new["hunk"] = dict(row["hunk"], targetChanges=[{"lineNo": n, "type": "ADD", "line": ""} for n in range(first, last + 1)])
    return new
