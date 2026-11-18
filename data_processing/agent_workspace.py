"""
Leak-safe workspaces for agentic test repair (WP5, level 3): a coding agent gets a project directory with the failing test and may explore and run
anything in it; its result is scored by the same plausibility runner as every other model.

State given to the agent = the project at the REPAIRED commit (the source change is in place) with the test's repaired span put back to the BROKEN
lines (the broken test fails). The directory is a fresh single-commit repository: no upstream history, no remote, so `git log`, `git show`,
other branches or tags cannot reveal the repaired test. (The network is not blocked here: `audit_trajectory` flags any command that could fetch it.)

  workspace = prepare_workspace(row, dest, clone=<callable>)      # tested with a local repository, no network
  task      = task_prompt(row)                                    # the instruction for the agent
  ... agent runs in `dest`, edits the test file ...
  result    = collect_result(row, dest)                           # final test function, files touched, diff
  flags     = audit_trajectory(commands)                          # network / history access, edits outside the test file

Scoring of an agent result: the final test FUNCTION replaces the whole function of the repaired file (a derived case whose reference is the whole repaired
function), so data_processing/plausibility_test.py runs it unchanged; the "does the test need the repaired span" flag is taken from the span control of
the same case in the normal runs (a span control on a whole function would be meaningless). See docs/agent_plan.md.
Nothing here runs an agent or touches the network by itself, except `clone_repository` (git clone of a public repository).
"""
import ast
import re
import shutil
import subprocess
from pathlib import Path

from data_processing.plausibility_test import find_repair_site, find_test_name, iter_functions, patch_source

NETWORK_PATTERNS = [r"\bgit\s+(clone|fetch|pull|remote|ls-remote|log|show|reflog|checkout\s+\S+|branch\s+-a|tag)\b", r"\bcurl\b", r"\bwget\b",
                    r"\bpip3?\s+(download|install)\b", r"\bgh\s+", r"https?://(github|raw\.githubusercontent|pypi)", r"\.git/"]
CACHE_PATH = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.hypothesis|[^/]*\.egg-info)(/|$)|\.py[co]$")
GIT_USER = ["-c", "user.name=agent", "-c", "user.email=agent@example.invalid"]


def _run(args, cwd=None, timeout=1800):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def clone_repository(repo, commit, dest):
    """Partial clone of github.com/<repo> checked out at `commit` in `dest` (with history; prepare_workspace strips it)."""
    done = _run(["git", "clone", "--quiet", "--filter=blob:none", f"https://github.com/{repo}.git", str(dest)])
    if done.returncode != 0:
        raise RuntimeError(f"clone failed: {done.stderr.strip()[-200:]}")
    done = _run(["git", "checkout", "--quiet", "--force", commit], cwd=dest)
    if done.returncode != 0:
        raise RuntimeError(f"checkout failed: {done.stderr.strip()[-200:]}")


def broken_lines(row):
    """The broken lines of the case (what the old test had where the repair is)."""
    match = re.search(r"\[<BREAKAGE>](.*?)\[</BREAKAGE>]", row["input"], re.DOTALL)
    if not match:
        raise ValueError("no [<BREAKAGE>] block in the case")
    return match.group(1).strip("\n")


def test_name_of(row):
    recorded = row.get("test_name")
    return recorded.split(".")[-1] if isinstance(recorded, str) and recorded else find_test_name(row.get("annotated_code", ""))


def revert_tests(dest, broken_hash, repaired_hash=None):
    """Put every TEST file that differs between the broken and the repaired commit back to its broken version (files the repair added are removed):
    the state a developer is in after changing the source and before touching any test. Other tests fixed in the same commit are then still broken."""
    names = _run(["git", "diff", "--name-status", broken_hash, "HEAD"], cwd=dest).stdout.splitlines()
    reverted = []
    for line in names:
        parts = line.split("\t")
        status, path = parts[0][0], parts[-1]
        if not is_test_file(path):
            continue
        if status == "A":
            _run(["git", "rm", "-q", "-f", path], cwd=dest)
        else:
            done = _run(["git", "checkout", broken_hash, "--", path], cwd=dest)
            if done.returncode != 0:
                raise RuntimeError(f"could not restore {path} from the broken commit: {done.stderr.strip()[-150:]}")
        reverted.append(path)
    return reverted


def is_test_file(path):
    """A path that belongs to the tests (test directories, test_*.py, *_test.py, conftest.py)."""
    from data_processing.context_ablation import is_test_path
    return is_test_path(path)


def prepare_workspace(row, dest, clone=None, sibling_safe=False):
    """
    Build the agent's directory `dest` for a case: repaired-commit source, history removed, one snapshot commit.
    Default: only the test's repaired span is put back to the broken lines (other tests of the commit stay REPAIRED and can be copied by the agent).
    sibling_safe=True: ALL test files that the repair changed are reverted to their broken-commit version (needs row["broken_hash"]); the target test is then
    the genuinely broken one, and its repaired span must be gone.
    `clone(repo, commit, dest)` fetches the project (default `clone_repository`). Returns {"test_file", "test_name", "site"} .
    """
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    (clone or clone_repository)(row["repository_name"], row["repaired_hash"], dest)
    test_file = dest / row["relative_path"]
    name = test_name_of(row)
    repaired = function_source(test_file.read_text(encoding="utf-8"), name) if name and test_file.exists() else None   # the repaired test, kept for scoring
    if sibling_safe:
        try:
            revert_tests(dest, row["broken_hash"])
            text = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
            site = find_repair_site(text, name, row["output"]) if name else None
            if not test_file.exists() or function_source(text, name) is None:
                raise ValueError("the target test does not exist in the broken version of its file")
            if site is not None:
                raise ValueError("the repaired span is still present after reverting the tests (the case is not a broken workspace)")
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        site, parts = None, nodeid_parts(text, name)
    else:
        source = test_file.read_text(encoding="utf-8")
        site = find_repair_site(source, name, row["output"]) if name else None
        if site is None:
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError("repair site not found in the repaired test (the case cannot be turned into a broken workspace)")
        test_file.write_text(patch_source(source, site, broken_lines(row)), encoding="utf-8")
        parts = list(site.nodeid_parts)
    shutil.rmtree(dest / ".git")                                    # no history, no remote: nothing in the repository knows the repair
    _run(["git", "init", "--quiet"], cwd=dest)
    _run(["git", "add", "-A"], cwd=dest)
    _run(["git", *GIT_USER, "commit", "--quiet", "-m", "snapshot: failing test"], cwd=dest)
    return {"test_file": row["relative_path"], "test_name": name, "site": site, "nodeid_parts": parts, "repaired_function": repaired[0] if repaired else None}


def task_prompt(row):
    """The instruction for the agent (the same task as the prompt-only runs: repair the test, not the source)."""
    return (
        f"The test `{test_name_of(row)}` in `{row['relative_path']}` fails because the project's source code was changed. "
        "Repair the TEST so that it passes again with the current source code. Rules:\n"
        f"- Edit ONLY the test function `{test_name_of(row)}` in `{row['relative_path']}`; do not change any other file or function.\n"
        "- Do not weaken the test: keep what it checks (assertions) and adapt it to the new behaviour of the code.\n"
        "- You may read any file and run the test (for example `python -m pytest <file>::<test>`).\n"
        "- Do not use git history, the network or package installation; everything you need is in this directory.\n"
        "When the test passes, stop."
    )


def nodeid_parts(source, name):
    """[enclosing classes..., name] of the single function `name` in `source` (for the pytest node id); [name] if not unique or absent."""
    try:
        found = [chain for node, chain in iter_functions(ast.parse(source)) if node.name == name]
    except SyntaxError:
        return [name]
    return list(found[0]) + [name] if len(found) == 1 else [name]


def function_source(source, name, chain=None):
    """Full source (decorators to last line) of the function `name` (inside class `chain` if given) and its 0-based first line; None if absent or ambiguous."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    found = []

    def walk(node, classes):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, classes + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == name and (chain is None or classes == chain):
                    found.append((child, classes))
            else:
                walk(child, classes)
    walk(tree, [])
    if len(found) != 1:
        return None
    node = found[0][0]
    first = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
    return "\n".join(source.split("\n")[first:node.end_lineno]), first


def collect_result(row, dest, chain=None):
    """What the agent left: the final function text, the files that differ from the snapshot, and whether anything outside the test function changed."""
    dest = Path(dest)
    name = test_name_of(row)
    status = [l for l in _run(["git", "status", "--porcelain", "-uall"], cwd=dest).stdout.splitlines() if not CACHE_PATH.search(l[3:])]
    new_files = sorted(l[3:] for l in status if l.startswith("??"))                  # created by the agent: scratch scripts, backups
    modified = sorted(l[3:] for l in status if not l.startswith("??"))               # tracked files changed or deleted
    touched = sorted(modified + new_files)
    final = function_source((dest / row["relative_path"]).read_text(encoding="utf-8"), name, chain)
    before = _run(["git", "show", f"HEAD:{row['relative_path']}"], cwd=dest).stdout
    old = function_source(before, name, chain)
    outside = [t for t in modified if t != row["relative_path"]]                      # a tracked file other than the test file was changed
    if final is not None and old is not None and row["relative_path"] in modified:
        a, b = before.split("\n"), (dest / row["relative_path"]).read_text(encoding="utf-8").split("\n")
        outside_function_changed = a[:old[1]] != b[:final[1]] or a[old[1] + len(old[0].split("\n")):] != b[final[1] + len(final[0].split("\n")):]
    else:
        outside_function_changed = False
    return {"function": final[0] if final else None, "touched_files": touched, "other_files_touched": outside, "new_files": new_files,
            "test_file_changed_outside_function": outside_function_changed, "function_found": final is not None}


def audit_trajectory(commands):
    """Flags for a list of shell commands the agent ran: anything that could fetch the repaired test (history, network, installs)."""
    flags = []
    for command in commands:
        for pattern in NETWORK_PATTERNS:
            if re.search(pattern, command):
                flags.append({"command": command[:200], "pattern": pattern})
                break
    return flags


def commands_from_trajectory(trajectory):
    """The shell commands of a mini-swe-agent trajectory (messages[].extra.actions[].command; tool-call arguments as a fallback)."""
    import json
    commands = []
    for message in trajectory.get("messages", []):
        actions = (message.get("extra") or {}).get("actions") or []
        for action in actions:
            if isinstance(action, dict) and action.get("command"):
                commands.append(action["command"])
        if not actions:
            for call in message.get("tool_calls") or []:
                try:
                    commands.append(json.loads(call["function"]["arguments"]).get("command", ""))
                except (KeyError, TypeError, ValueError):
                    pass
    return [c for c in commands if c]
