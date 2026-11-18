"""
Context ablation (WP5): extra context for the LLM prompt, added one source at a time so that each can be judged alone.

Variants (the base prompt is the one of llm_test.py, unchanged):
  A  baseline                                   (nothing added; use the normal runners)
  B  + "### Test Failure Output:"               the pytest output of the broken test (dataset column outdated_test_log), cleaned
  C  + "### Definitions of Changed Symbols:"    signatures and docstrings of the names used in the broken lines, looked up in the
                                                project's NON-test source at the repaired commit
  BC both

No leakage by construction: the failure output is the run of the OLD test, the definitions come from source files only (test files, test
directories and the case's own test file are excluded) and `build` reports how often the reference repair appears verbatim in the added
context (it should be ~0).

  # 1) build the context file for a split (clones each project at its repaired commit, partial clone; run in a scratch dir)
  python -m data_processing.context_ablation build --split-dir ~/eval_results_perproject_llm/contrast300 --out ~/eval_ablation/context.json \
      --workdir <scratch>
  # 2) the runners take it:  --context-variant B|C|BC --context-file ~/eval_ablation/context.json   (vllm_run.py, openrouter_run.py)
"""
import argparse
import ast
import builtins
import json
import keyword
import re
import shutil
import subprocess
import warnings
from pathlib import Path

VARIANTS = ("B", "C", "BC")
FAILURE_MAX_CHARS = 2500
DEFINITIONS_MAX_CHARS = 3000
MAX_SYMBOLS = 8
MAX_DEFS_PER_SYMBOL = 2
TEST_DIRS = {"test", "tests", "testing", "unittests", "unittest"}
COMMON_NAMES = set(dir(builtins)) | set(keyword.kwlist) | {
    "self", "cls", "assert", "pytest", "mock", "patch", "Mock", "MagicMock", "unittest", "np", "pd", "os", "sys", "json", "time", "re",
    "fixture", "raises", "parametrize", "mark", "assertEqual", "assertTrue", "assertFalse", "assertRaises", "assertIn", "assertIsNone",
    "assertIsNotNone", "assertAlmostEqual", "assertNotEqual", "setUp", "tearDown", "main", "args", "kwargs", "result", "expected", "actual",
}
PATH_PREFIX = re.compile(r"/(?:gpfs|home|tmp|scratch)/\S*?/(?:repos[\w.\-]*|java_work\d*|work)/[^/\s]+/")


def case_key(row):
    """Stable key of a case in the context file."""
    return f"{row['repository_name']}|{row['broken_hash']}|{row['test_name']}"


# ---- B: failure output -----------------------------------------------------------------------------------------------------

def failure_context(row, max_chars=FAILURE_MAX_CHARS):
    """The pytest failure output of the broken test, without the marker, with machine paths removed and cut to max_chars (head + tail)."""
    log = (row.get("outdated_test_log") or "").replace("[<TESTLOG>]", "").strip()
    log = PATH_PREFIX.sub("", log)
    lines = []
    for line in log.splitlines():
        stripped = re.sub(r"^[=_\-]{5,}\s*|\s*[=_\-]{5,}$", "", line).rstrip()      # pytest's ruler decoration around headers
        if stripped or not line.strip(" =_-"):
            lines.append(stripped if stripped else "")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) > max_chars:
        head = max_chars * 45 // 100
        text = text[:head].rstrip() + "\n...\n" + text[-(max_chars - head):].lstrip()
    return text


# ---- C: definitions of the symbols of the broken lines ---------------------------------------------------------------------

def symbols_from_broken(broken_lines, limit=MAX_SYMBOLS):
    """Names a repair may depend on: identifiers of the broken lines in order of appearance, without keywords, builtins and test boilerplate."""
    seen, out = set(), []
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", broken_lines or ""):
        if name in seen or name in COMMON_NAMES or len(name) < 3:
            continue
        seen.add(name)
        out.append(name)
    return out[:limit]


def is_test_path(path):
    parts = Path(path).parts
    name = Path(path).name
    return bool(TEST_DIRS & {p.lower() for p in parts[:-1]}) or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _signature(node):
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    returns = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) is not None else ""
    return f"{prefix} {node.name}({args}){returns}"


def _doc(node, lines=2):
    doc = ast.get_docstring(node)
    return " ".join(doc.strip().splitlines()[:lines]) if doc else ""


def index_source(root, exclude=()):
    """{name: [(relative path, rendered definition)]} for the classes, functions and methods of the non-test .py files under root."""
    root = Path(root)
    index = {}
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        if is_test_path(rel) or rel in exclude or any(p.startswith(".") for p in Path(rel).parts) or path.stat().st_size > 500_000:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")           # third-party code: invalid escape sequences etc. are not our business
                tree = ast.parse(path.read_text(errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][:15]
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                text = f"class {node.name}({bases})" + (f"  # {_doc(node)}" if _doc(node) else "") + (f"\n    methods: {', '.join(methods)}" if methods else "")
                index.setdefault(node.name, []).append((rel, text))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                text = _signature(node) + (f"  # {_doc(node)}" if _doc(node) else "")
                index.setdefault(node.name, []).append((rel, text))
    return index


def definitions_context(symbols, index, max_chars=DEFINITIONS_MAX_CHARS):
    """Rendered definitions of the symbols found in the index (at most MAX_DEFS_PER_SYMBOL each), cut to max_chars."""
    blocks = []
    for name in symbols:
        for rel, text in index.get(name, [])[:MAX_DEFS_PER_SYMBOL]:
            blocks.append(f"# {rel}\n{text[:500]}")
    out, size = [], 0
    for block in blocks:
        if size + len(block) > max_chars:
            break
        out.append(block)
        size += len(block) + 2
    return "\n\n".join(out)


# ---- the prompt ------------------------------------------------------------------------------------------------------------

def add_sections(prompt, failure=None, definitions=None):
    """Insert the extra sections before the final '### Repaired Code:' tail of the base prompt."""
    tail = "### Repaired Code:\n"
    if not prompt.endswith(tail):
        raise ValueError("unexpected prompt tail")
    extra = ""
    if failure:
        extra += f"### Test Failure Output:\n{failure}\n\n"
    if definitions:
        extra += f"### Definitions of Changed Symbols:\n{definitions}\n\n"
    return prompt[:-len(tail)] + extra + tail


def attach(tester, variant, context):
    """Make `tester.build_prompt` add the sections of `variant` (B, C or BC) from `context` {case key: {"failure", "definitions"}}."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    base = tester.build_prompt

    def build_prompt(row, max_hunks=None):
        key = case_key(row)
        if key not in context:                    # never fall back silently to the baseline prompt: the run would be mislabelled
            raise KeyError(f"case not in the context file: {key}")
        entry = context[key]
        return add_sections(base(row, max_hunks), entry.get("failure") if "B" in variant else None,
                            entry.get("definitions") if "C" in variant else None)

    tester.build_prompt = build_prompt
    return tester


# ---- building the context file ---------------------------------------------------------------------------------------------

def _git(*args, cwd=None, timeout=1800):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def checkout(workdir, repo, commit):
    """Partial clone of github.com/<repo> (once) and a clean checkout of `commit`; returns the directory."""
    project = Path(workdir) / repo.replace("/", "__")
    if not project.exists():
        project.parent.mkdir(parents=True, exist_ok=True)
        done = _git("clone", "--quiet", "--filter=blob:none", f"https://github.com/{repo}.git", str(project))
        if done.returncode != 0:
            shutil.rmtree(project, ignore_errors=True)
            raise RuntimeError(f"clone failed: {done.stderr.strip()[-200:]}")
    _git("reset", "--hard", "--quiet", cwd=project)
    if _git("checkout", "--quiet", "--force", commit, cwd=project).returncode != 0:
        _git("fetch", "--quiet", "origin", commit, cwd=project)
        done = _git("checkout", "--quiet", "--force", commit, cwd=project)
        if done.returncode != 0:
            raise RuntimeError(f"checkout failed: {done.stderr.strip()[-200:]}")
    return project


def build(split_dir, out_path, workdir, limit=None):
    """Context file for splits/test.json of `split_dir` (resumable; one entry per case; the clone of a project is removed afterwards)."""
    rows = json.load(open(Path(split_dir) / "splits" / "test.json"))[:limit]
    out_path = Path(out_path)
    context = json.loads(out_path.read_text()) if out_path.exists() else {}
    by_repo = {}
    for row in rows:
        if case_key(row) not in context:
            by_repo.setdefault((row["repository_name"], row["repaired_hash"]), []).append(row)
    for (repo, commit), group in by_repo.items():
        try:
            project = checkout(workdir, repo, commit)
            index = index_source(project, exclude={g["relative_path"] for g in group})
        except Exception as error:
            print(f"{repo}: {error!r}", flush=True)
            index = None
        for row in group:
            broken = (re.search(r"\[<BREAKAGE>](.*?)\[</BREAKAGE>]", row["input"], re.DOTALL) or [None, ""])[1]
            definitions = definitions_context(symbols_from_broken(broken), index) if index is not None else ""
            context[case_key(row)] = {"failure": failure_context(row), "definitions": definitions, "index_ok": index is not None}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(context))
        tmp.replace(out_path)
        print(f"{repo}@{commit[:8]}: {len(group)} cases", flush=True)
        shutil.rmtree(Path(workdir) / repo.replace("/", "__"), ignore_errors=True)
    report(rows, context)
    return context


def report(rows, context):
    """Coverage and leakage summary."""
    n = len(rows)
    have = [context[case_key(r)] for r in rows if case_key(r) in context]
    leaks = 0
    for row in rows:
        entry = context.get(case_key(row))
        target = re.sub(r"\s+", "", row["output"].replace("[<REPAIREDTEST>]", "").replace("[</REPAIREDTEST>]", ""))
        if entry and len(target) > 25 and target in re.sub(r"\s+", "", entry["failure"] + entry["definitions"]):
            leaks += 1
    print(f"cases {n}, with context {len(have)}; failure output present {sum(bool(e['failure']) for e in have)}, definitions present "
          f"{sum(bool(e['definitions']) for e in have)}, index failed {sum(not e['index_ok'] for e in have)}, "
          f"reference appears verbatim in the added context: {leaks}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--workdir", required=True)
    b.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.split_dir, args.out, args.workdir, args.limit)


if __name__ == "__main__":
    main()
