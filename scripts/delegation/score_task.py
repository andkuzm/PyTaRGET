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
        for cls, m, status, _ in data: passed[f"{cls.split('.')[-1].split('$')[0]}#{m.split('(')[0]}"] = status == "pass"
    missing = [t for t in old if t not in final]
    failing = [t for t in old if t in final and not passed.get(t, False)]
    weakened = [t for t in old if t in final and final[t] < ref.get(t, old[t])]
    now = hashes(ws, module); before = json.loads((task / "source_hashes.json").read_text())
    source_modified = sorted(k for k in set(now) | set(before) if now.get(k) != before.get(k))
    res = {"compile_ok": kind != "compile_error", "tests_expected": len(old), "tests_pass": sum(1 for t in old if passed.get(t)), "missing": missing, "failing": failing, "weakened": weakened,
           "new_tests": sorted(t for t in final if t not in old), "source_modified": source_modified,
           "success": kind != "compile_error" and not missing and not failing and not weakened and not source_modified}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
