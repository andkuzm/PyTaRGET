"""Score a finished run: python score_task.py <run workspace> <task dir> > result JSON. Success = the module compiles, every test of the OLD suite exists and passes, no test has fewer assertions than in the
reference tests, and no main source file or pom was modified. Also reports tests added, per-test results and the files the agent changed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtools import common  # noqa: E402
from prepare_task import hashes  # noqa: E402


def methods_of(root):
    out = {}
    for f in sorted(Path(root).rglob("*.java")):
        text = f.read_text(errors="ignore"); lines = text.split("\n")
        for name, a, b in common.find_methods(text):
            out[f"{f.stem}#{name}"] = common.count_asserts("\n".join(lines[a - 1:b]))
    return out


def main(ws, task):
    ws, task = Path(ws), Path(task)
    cfg = common.load_config(ws); module = cfg["module"] if cfg["module"] != "." else ""
    rc, log = common.mvn(ws, timeout=2400)
    kind, data = common.result_of(ws, rc, log)
    old = methods_of(task / "old_tests"); ref = methods_of(task / "ref")
    final = methods_of(ws / cfg["test_roots"][0]) if (ws / cfg["test_roots"][0]).exists() else {}
    passed = {}
    if kind == "tests":
        for cls, m, status, _ in data: passed[f"{cls.split('.')[-1].split('$')[0]}#{m.split('(')[0]}"] = status
    # Tests are compared per CLASS, not by name: a renamed test is not a deletion. A class is "shrunk" when it has fewer test methods than in the old suite, and "weakened" when its total number of
    # assertions is lower than in the reference tests. Name-based differences are reported for information only.
    by_class = lambda d: {c: [v for k, v in d.items() if k.split("#")[0] == c] for c in {k.split("#")[0] for k in d}}
    oc, rc_, fc = by_class(old), by_class(ref), by_class(final)
    # a test class renamed together with the class under test (EntryTest -> LedgerEntryTest) is the same class: map by overlap of method names
    names = lambda d, c: {k.split("#")[1] for k in d if k.split("#")[0] == c}
    for c in list(oc):
        if c in fc: continue
        cand = max((x for x in fc if x not in oc), key=lambda x: len(names(old, c) & names(final, x)), default=None)
        if cand and len(names(old, c) & names(final, cand)) >= max(1, len(names(old, c)) // 2): fc[c] = fc.pop(cand)
    shrunk = sorted(c for c in oc if len(fc.get(c, [])) < len(oc[c]))
    weakened = sorted(c for c in oc if sum(fc.get(c, [])) < sum(rc_.get(c, oc[c])))
    missing = [t for t in old if t not in final]                      # information: by name (renames included)
    failing = sorted(t for t, st in passed.items() if st in ("fail", "error")) if kind == "tests" else list(final)
    # new (sandboxed) projects: tests that fail from the environment on the OLD code, or already with the reference tests, are not counted against a run (listed in the result)
    ex = task / "env_excluded.json"
    excluded = {(x.split("#")[0].split(".")[-1].split("$")[0], x.split("#")[1]) for v in json.loads(ex.read_text()).values() for x in v} if ex.exists() else set()
    excluded_failing = [t for t in failing if (t.split("#")[0], t.split("#")[1].split("[")[0]) in excluded]
    failing = [t for t in failing if t not in excluded_failing]
    now = hashes(ws, module); before = json.loads((task / "source_hashes.json").read_text())
    source_modified = sorted(k for k in set(now) | set(before) if now.get(k) != before.get(k))
    res = {"compile_ok": kind != "compile_error", "tests_expected": len(old), "tests_pass": sum(1 for t in final if passed.get(t) == "pass"), "tests_final": len(final), "missing_by_name": missing, "shrunk_classes": shrunk, "failing": failing, "weakened_classes": weakened,
           "new_tests": sorted(t for t in final if t not in old), "source_modified": source_modified, "excluded_env_failing": excluded_failing,
           "success": kind != "compile_error" and not shrunk and not failing and not weakened and not source_modified}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
