"""
Leak-safe workspaces for the agentic repair of Java tests (TaRBench rows), the Java counterpart of agent_workspace.py.

State given to the agent = the project at the AFTER commit (aCommit: the source change is in place) with the target test METHOD put back to its BEFORE text
(bSource) ("span_only", the default). Java compiles all test files of a module together, so the other tests fixed by the same commit must stay repaired or nothing
compiles; the case records how many OTHER test lines the repair changed (`other_test_lines_changed`): 0 means nothing else can be copied (clean case), more means the
agent may copy the fix pattern from repaired siblings (analyse separately). Mode "revert_all_tests" puts every changed test file back (only for modules where that
still compiles). The directory is a fresh single-commit repository without history or remote.

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


def other_test_lines_changed(dest, row):
    """Test lines changed between the before and the after commit OUTSIDE the target method (other methods, other test files); the clone has the history."""
    first, last = method_range(row["aSource"])
    total = 0
    for line in _run(["git", "diff", "--numstat", row["bCommit"], "HEAD"], cwd=dest).stdout.splitlines():
        added, deleted, path = line.split("\t", 2)
        if not path.endswith(".java") or not is_test_file(path) or added == "-":
            continue
        if path != row["aPath"]:
            total += int(added) + int(deleted)
            continue
        for hunk in _run(["git", "diff", "-U0", row["bCommit"], "HEAD", "--", path], cwd=dest).stdout.splitlines():
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", hunk)
            if m:
                new_start, new_len = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
                new_end = new_start + max(new_len, 1) - 1
                if not (first <= new_start and new_end <= last):
                    total += max(new_len, 1) + int(m.group(2) if m.group(2) is not None else 1)
    return total


def prepare_workspace(row, dest, clone=None, mode="span_only"):
    """
    Build the agent's directory: after-commit tree, the target method back at its before text (span_only), history removed, one snapshot commit.
    mode "revert_all_tests": every test file changed by the repair is restored to its before version instead. Returns the case info
    (class, method, method_lines = the method's lines IN THE SNAPSHOT, other_test_lines_changed).
    """
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    if row["aPath"] != row["bPath"]:
        raise ValueError("the test file was moved by the repair (aPath != bPath): not supported")
    (clone or clone_repository)(row["ID"].split(":")[0], row["aCommit"], dest)
    try:
        other = other_test_lines_changed(dest, row)
        if mode == "revert_all_tests":
            for line in _run(["git", "diff", "--name-status", row["bCommit"], "HEAD"], cwd=dest).stdout.splitlines():
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
            lines = [first, last]
        else:
            path = dest / row["aPath"]
            text = path.read_text(encoding="utf-8", errors="ignore").split("\n")
            first, last = method_range(row["aSource"])
            if "\n".join(text[first - 1:last]).strip() != row["aSource"]["code"].strip():
                raise ValueError("the after method is not at its recorded lines (the repository state differs from the dataset)")
            before_code = row["bSource"]["code"].split("\n")
            indent = text[first - 1][:len(text[first - 1]) - len(text[first - 1].lstrip())]
            if before_code[0] == before_code[0].lstrip():                 # the dataset stores the first line of a method without its indentation
                before_code[0] = indent + before_code[0]
            path.write_text("\n".join(text[:first - 1] + before_code + text[last:]), encoding="utf-8")
            lines = [first, first + len(before_code) - 1]
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    shutil.rmtree(dest / ".git")
    _run(["git", "init", "--quiet"], cwd=dest)
    _run(["git", "add", "-A"], cwd=dest)
    _run(["git", *GIT_USER, "commit", "--quiet", "-m", "snapshot: failing test"], cwd=dest)
    cls, method = test_class_and_method(row)
    return {"test_file": row["bPath"], "class": cls, "method": method, "method_lines": lines, "other_test_lines_changed": other}


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


def collect_result(row, dest, method_lines=None):
    """What the agent left: the final method text (the lines between the unchanged text before and after the method), touched files, outside edits."""
    dest = Path(dest)
    status = [l for l in _run(["git", "status", "--porcelain", "-uall"], cwd=dest).stdout.splitlines() if not CACHE_PATH.search(l[3:])]
    new_files = sorted(l[3:] for l in status if l.startswith("??"))
    modified = sorted(l[3:] for l in status if not l.startswith("??"))
    other = [m for m in modified if m != row["bPath"]]
    before = _run(["git", "show", f"HEAD:{row['bPath']}"], cwd=dest).stdout.split("\n")
    path = dest / row["bPath"]
    after = path.read_text(encoding="utf-8", errors="ignore").split("\n") if path.exists() else []
    first, last = method_lines or method_range(row["bSource"])          # the method's lines in the snapshot (before text)
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


def swap_method(dest, row, method_lines, to="after"):
    """Write the AFTER (repaired) or BEFORE (broken) text of the target method into the workspace file; returns the file's previous text.
    method_lines = the method's lines in the file as it is now (the manifest's method_lines for the broken snapshot)."""
    path = Path(dest) / row["bPath"]
    old = path.read_text(encoding="utf-8", errors="ignore")
    lines = old.split("\n")
    first, last = method_lines
    code = (row["aSource"] if to == "after" else row["bSource"])["code"].split("\n")
    indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
    if code[0] == code[0].lstrip():
        code[0] = indent + code[0]
    path.write_text("\n".join(lines[:first - 1] + code + lines[last:]), encoding="utf-8")
    return old
