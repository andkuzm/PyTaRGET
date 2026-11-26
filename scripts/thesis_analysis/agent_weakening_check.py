"""Heuristic weakening signals in the PASSING final functions of an agent run, compared with the real repair (only candidates that differ from it). Signals, not proof: read the flagged ones.
Java: usage agent_weakening_check.py java <agent out dir> ; Python: ... python <agent plaus layout dir> <agent out dir>"""
import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
H = Path.home(); n = lambda s: "".join((s or "").split())
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b|\braises\b")
EXACT = re.compile(r"assertEquals|isEqualTo\b|containsExactly|hasMessage\(|assertArrayEquals|assert [^\n]*==|assertEqual\b|assert_called\w*_with")
LOOSE = re.compile(r"assertNotNull|isNotNull|isNotEmpty|assertTrue\s*\(\s*true|assert True\b|is not None|isinstance\(|anyOf|greaterThan|isGreaterThan|any\(\)|\bANY\b")
SKIP = re.compile(r"@Ignore|@Disabled|assumeTrue|pytest\.skip|mark\.(skip|xfail)|SkipTest")
CATCH = re.compile(r"catch\s*\(\s*(Throwable|Exception)\b|except\s+(Exception|BaseException)\b|except\s*:")


def signals(cand, ref):
    c = lambda p: (len(p.findall(cand)), len(p.findall(ref)))
    out = []
    a, b = c(ASSERT); out += ["fewer assertions"] if a < b else []
    a, b = c(EXACT); out += ["fewer exact-value checks"] if a < b else []
    a, b = c(LOOSE); out += ["more loose checks"] if a > b else []
    a, b = c(SKIP); out += ["adds a skip"] if a > b else []
    a, b = c(CATCH); out += ["adds a broad catch"] if a > b else []
    return out


pairs = []
if sys.argv[1] == "java":
    rows = json.load(open(H / "eval_results_java_llm/contrast300/splits/test.json"))
    for f in Path(sys.argv[2]).glob("*/result.json"):
        r = json.load(open(f)); pairs.append((f.parent.name, r.get("method") or "", rows[int(f.parent.name)]["aSource"]["code"]))
else:
    from data_processing import prediction_files
    base = Path(sys.argv[2]) / "0.8"; cm = json.load(open(base / "case_map.json"))["case_map"]
    res = json.load(open(base / "plausibility_v3.json")); res = res.get("results", res)
    for p in prediction_files.load(base / "test_predictions.json"):
        if res.get(str(p["ID"]), {}).get("verdicts", [""])[0] == "pass": pairs.append((cm[str(p["ID"])], p["preds"][0], p["target"]))
diff = [(i, c, t) for i, c, t in pairs if n(c) != n(t)]
flag = {i: s for i, c, t in diff if (s := signals(c, t))}
print(f"candidates {len(pairs)}, differ from the real repair {len(diff)}, with at least one weakening signal {len(flag)}")
print("  signals:", dict(collections.Counter(x for s in flag.values() for x in s)))
print("  flagged cases:", {k: v for k, v in sorted(flag.items(), key=lambda kv: int(kv[0]))})
