"""
Context for the Java prompts (the Java counterpart of context_ablation.py; same sections, same variants B / C / BC):
  B  "### Test Failure Output:"            the Maven output of the BROKEN test: the project at the after commit (source change in place) with the target method put back to its before
                                          text, the test run once (compile errors or test failure), cleaned: only error lines, no machine paths, at most 2500 characters
  C  "### Definitions of Changed Symbols:" signatures of the identifiers of the broken lines, looked up in the project's NON-test Java sources at the after commit
No leakage by construction: the broken run shows the OLD test text, the definitions come from main sources only (test files and the target test file are excluded).
  PYTHONPATH=. python -m data_processing.java_context build --split-dir ~/eval_results_java_llm/contrast273 --out ~/java_ctx/context.json --workdir <scratch> [--workers 4]
Needs the NETWORK (git clone, Maven downloads) and runs third-party build code: not on a login node. Resumable.
"""
import argparse
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data_processing.agent_workspace_java import derived_row, is_test_file, method_range
from data_processing.context_ablation import definitions_context
from data_processing.java_plausibility import JavaPlausibility, SUCCESS, split_method

FAILURE_MAX_CHARS = 2500
JAVA_KEYWORDS = set("abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for goto if implements import "
                    "instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try "
                    "void volatile while true false null var record".split())
COMMON = JAVA_KEYWORDS | set("assertEquals assertTrue assertFalse assertNull assertNotNull assertThat assertThrows assertSame assertNotSame fail verify when thenReturn mock spy "
                              "String Integer Long List Map Set Object Arrays Collections Test Before After this".split())
PATH_PREFIX = re.compile(r"/(?:gpfs|home|tmp|scratch)/\S*?/repos/[^/\s]+/")
NOISE = re.compile(r"^\[ERROR\]\s*(To see the full stack trace|Re-run Maven|For more information|\[Help|Help \d|-> \[Help)|^\[ERROR\]\s*$|^\[ERROR\] After correcting|^\[ERROR\]\s+mvn <")
METHOD_DECL = re.compile(r"^\s*(?:(?:public|protected|private|static|final|abstract|synchronized|default|native)\s+)*(?:<[^>]+>\s+)?[\w.<>\[\],?]+(?:\s*<[^>]*>)?\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w., ]+)?\s*[{;]")
TYPE_DECL = re.compile(r"^\s*(?:(?:public|protected|private|static|final|abstract)\s+)*(class|interface|enum|record)\s+(\w+)([^{]*)\{")


def failure_text(log, max_chars=FAILURE_MAX_CHARS):
    """The error lines of a Maven log (compile errors, failed tests, exceptions), without machine paths, download noise and help text; head + tail when long."""
    lines = []
    for line in PATH_PREFIX.sub("", log.replace("\r", "")).split("\n"):
        s = line.rstrip()
        if NOISE.search(s):
            continue
        if s.startswith("[ERROR]") or s.startswith("[WARNING] COMPILATION") or re.search(r"Tests run:.*(FAIL|ERROR)|^\s*(java|org|junit)\.[\w.$]+(Exception|Error)|^Caused by:|^\s+at .*Test", s):
            lines.append(re.sub(r"^\[ERROR\]\s?", "", s))
    text = "\n".join(dict.fromkeys(lines)).strip()            # drop repeated lines (Maven prints the same error in several places)
    if len(text) > max_chars:
        head = max_chars * 45 // 100
        text = text[:head].rstrip() + "\n...\n" + text[-(max_chars - head):].lstrip()
    return text


def symbols_from_broken(broken_lines, limit=8):
    seen, out = set(), []
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", broken_lines or ""):
        if name in seen or name in COMMON or len(name) < 3:
            continue
        seen.add(name)
        out.append(name)
    return out[:limit]


def index_java_source(root, exclude=()):
    """{name: [(relative path, declaration)]} for the types and methods of the non-test .java files under root (regex based, signatures only)."""
    root = Path(root)
    index = {}
    for path in sorted(root.rglob("*.java")):
        rel = str(path.relative_to(root))
        if is_test_file(rel) or rel in exclude or "/target/" in f"/{rel}" or path.stat().st_size > 500_000:
            continue
        try:
            lines = path.read_text(errors="ignore").split("\n")
        except OSError:
            continue
        for line in lines:
            m = TYPE_DECL.match(line)
            if m:
                index.setdefault(m.group(2), []).append((rel, line.strip().rstrip("{").strip()))
                continue
            m = METHOD_DECL.match(line)
            if m and m.group(1) not in JAVA_KEYWORDS and not line.strip().startswith(("return", "new ", "throw", "else", "if", "for", "while", "}")):
                index.setdefault(m.group(1), []).append((rel, line.strip().rstrip("{;").strip()))
    return index


def broken_method_text(row, original):
    """The before text of the target method with the indentation of the after file's first method line."""
    first, _ = method_range(row["aSource"])
    lines = original.split("\n")
    code = row["bSource"]["code"].split("\n")
    indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
    if code[0] == code[0].lstrip():
        code[0] = indent + code[0]
    return "\n".join(code)


def build_case(runner, row):
    """One case: checkout of the after commit, the broken method run once, the definitions looked up. Returns the context entry."""
    repo = row["ID"].split(":")[0]
    project = runner.prepare(repo, row["aCommit"])
    test_rel = Path(row["aPath"])
    original = (project / test_rel).read_text()
    cls, method = split_method(row["name"])
    status, _, log = runner.execute(project, derived_row(row), original, broken_method_text(row, original), test_rel, cls, method)
    broken = row["bSource"]["code"]
    definitions = definitions_context(symbols_from_broken(broken), index_java_source(project, exclude={str(test_rel)}))
    return {"failure": failure_text(log) if status != SUCCESS else "", "definitions": definitions, "broken_status": status}


def build(split_dir, out_path, workdir, workers=3, limit=None, m2=None):
    rows = json.load(open(Path(split_dir) / "splits" / "test.json"))[:limit]
    out_path = Path(out_path)
    context = json.loads(out_path.read_text()) if out_path.exists() else {}
    runner = JavaPlausibility(workdir, m2=m2, timeout=900)
    by_repo, lock = {}, threading.Lock()
    for row in rows:
        if row["ID"] not in context:
            by_repo.setdefault(row["ID"].split(":")[0], []).append(row)

    def work(item):
        repo, group = item
        for row in group:
            try:
                entry = build_case(runner, row)
            except Exception as error:
                entry = {"failure": "", "definitions": "", "broken_status": "error", "detail": repr(error)[:200]}
            with lock:
                context[row["ID"]] = entry
                tmp = out_path.with_suffix(".tmp")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(context))
                tmp.replace(out_path)
            print(f"{row['ID']}: {entry['broken_status']}, failure {len(entry['failure'])} chars, definitions {len(entry['definitions'])} chars", flush=True)
        shutil.rmtree(Path(workdir) / "repos" / repo.replace("/", "__"), ignore_errors=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, by_repo.items()))
    import collections
    print("broken-run statuses:", dict(collections.Counter(v["broken_status"] for v in context.values())))
    return context


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--workdir", required=True)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--m2", default=None)
    args = ap.parse_args()
    build(args.split_dir, args.out, args.workdir, args.workers, args.limit, args.m2)


if __name__ == "__main__":
    main()
