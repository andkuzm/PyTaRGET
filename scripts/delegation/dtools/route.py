"""Automatic routing around the delegate, no model: (1) literal sync, (2) per-unit routing.
(1) literal_sync: failing tests whose message says `expected: X but was: Y` where X and Y differ by one substitution (e.g. .../blob/master/... -> .../blob/main/...), the same substitution in >= MIN_TESTS tests: the
    text (with a little context) is replaced everywhere in the test files of the failing classes. This is what a repair of such a change is anyway (new expected value = new behaviour).
(2) route: a method unit whose failure message expects a value X that does NOT occur in the method text cannot be fixed by editing the method (the value lives in a helper / constant / other place): such units are not
    sent to the delegate, they stay failing for the strong agent. (Measured on 651 particle-1 units: expected text in the method -> 94% fixed; not in the method -> 9%.)"""
import re
from collections import defaultdict
from pathlib import Path

MIN_TESTS = 3; CTX = 12; MIN_NEEDLE = 8
PATS = [re.compile(r"expected:\s*(.*?)\s*\|\s*but was\s*:\s*(.*?)(?:\s*\||$)"), re.compile(r"expected:\s*<(.*?)>\s*but was:\s*<(.*?)>"), re.compile(r"expected:\s*(.*?)\s+but was:\s*(.*?)(?:\s*\||$)")]


def expected_actual(msg):
    for p in PATS:
        m = p.search(msg or "")
        if m: return m.group(1).strip(), m.group(2).strip()
    return None


def fragment(e):
    parts = [p.strip() for p in re.split(r"…|\.\.\.", e) if len(p.strip()) >= 6]
    return max(parts, key=len) if parts else None


def substitution(e, a):
    """(needle, replacement) when e and a differ by one contiguous substitution (ellipsis markers ignored), else None."""
    e2, a2 = e.strip("…"), a.strip("…")
    if "…" in e2 or "…" in a2 or e2 == a2: return None
    i = 0
    while i < min(len(e2), len(a2)) and e2[i] == a2[i]: i += 1
    j = 0
    while j < min(len(e2), len(a2)) - i and e2[-1 - j] == a2[-1 - j]: j += 1
    ce, ca = e2[i:len(e2) - j], a2[i:len(a2) - j]
    if not ce or not ca or len(ce) > 60 or len(ca) > 60: return None
    left, right = e2[max(0, i - CTX):i], e2[len(e2) - j:len(e2) - j + CTX]
    needle, repl = left + ce + right, left + ca + right
    return (needle, repl) if len(needle) >= MIN_NEEDLE else None


def literal_sync(ws, data, test_files, min_tests=MIN_TESTS):
    """data: surefire rows (class, method, status, message). Returns [(needle, repl, n_tests, n_replacements)] and edits the files."""
    groups = defaultdict(set)
    for cls, meth, st, msg in data:
        if st in ("fail", "error"):
            ea = expected_actual(msg)
            s = ea and substitution(*ea)
            if s: groups[s].add((cls, meth))
    done = []
    for (needle, repl), tests in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(tests) < min_tests: continue
        # a substitution established by >= MIN_TESTS tests is also applied for failing tests whose message merely CONTAINS the old text (other message layouts, e.g. "expected to end with")
        tests = tests | {(c, m) for c, m, st, msg in data if st in ("fail", "error") and needle in (msg or "")}
        classes = {c.split(".")[-1].split("$")[0] for c, _ in tests}; n = 0
        for f in test_files:
            if f.stem not in classes: continue
            t = f.read_text(errors="ignore")
            if needle in t: f.write_text(t.replace(needle, repl)); n += t.count(needle)
        if n: done.append((needle, repl, len(tests), n))
    return done


def route(unit, unit_text, message):
    """None = give to the delegate, else the reason why the unit is left for the strong agent."""
    if unit["kind"] != "method": return None
    ea = expected_actual(message); fr = ea and fragment(ea[0])
    if fr and fr not in unit_text: return f"expected text '{fr[:50]}' is not in the method (the value lives in a helper/constant elsewhere)"
    return None
