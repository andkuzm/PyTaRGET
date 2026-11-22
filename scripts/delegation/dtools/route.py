"""Automatic routing around the delegate, no model: (1) literal sync, (2) per-unit routing.
(1) literal_sync: failing tests whose message says `expected: X but was: Y` where X and Y differ by one substitution (e.g. .../blob/master/... -> .../blob/main/...), the same substitution in >= MIN_TESTS tests: the
    text (with a little context) is replaced everywhere in the test files of the failing classes. This is what a repair of such a change is anyway (new expected value = new behaviour).
(2) route: a method unit whose failure message expects a value X that does NOT occur in the method text cannot be fixed by editing the method (the value lives in a helper / constant / other place): such units are not
    sent to the delegate, they stay failing for the strong agent. (Measured on 651 particle-1 units: expected text in the method -> 94% fixed; not in the method -> 9%.)"""
import re
from collections import defaultdict
from pathlib import Path

MIN_TESTS = 3; CTX = 12; MIN_NEEDLE = 8
ANGLE = re.compile(r"expected:\s*<(.*?)>\s*but was:\s*<(.*?)>")
SPLIT = re.compile(r"expected:\s*(.*?)\s*(?:\|\s*)?but was\s*:\s*(.*)$")


def expected_actual(msg):
    """(expected, actual) from `expected: X but was: Y` (JUnit angle form, Truth, multi-line values joined with ' | ')."""
    msg = msg or ""
    m = ANGLE.search(msg) or SPLIT.search(msg)
    if not m: return None
    e, a = (x.strip().strip("|").strip() for x in m.groups())
    return (e, a) if e and a else None


def fragment(e):
    parts = [p.strip() for p in re.split(r"…|\.\.\.|\s\|\s", e) if len(p.strip()) >= 6]
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


def append_suffix(e, a):
    """The actual message is the expected one plus extra trailing text (new line appended to a message): (expected tail, java suffix) else None."""
    e2 = e.strip("…").strip()
    if not e2: return None
    a2 = a.strip("…").strip()
    tail = fragment(e) or e2
    if tail is None or e2 not in a2 or not a2.startswith(e2) or len(a2) <= len(e2): return None
    suffix = a2[len(e2):]
    if not suffix.strip(" |") or len(suffix) > 200: return None
    return tail, suffix.replace(" | ", "\\n", 1) if suffix.startswith(" | ") else suffix


def append_sync(ws, data, test_files, find_methods, min_tests=MIN_TESTS):
    """Same appended text (e.g. a new `See https://...` line) in >= min_tests failing tests: append it to the matching string literal INSIDE each failing test method (only there: other tests may use the same literal)."""
    groups = defaultdict(list)
    for cls, meth, st, msg in data:
        if st in ("fail", "error"):
            ea = expected_actual(msg); s = ea and append_suffix(*ea)
            if s: groups[s[1]].append((cls, meth, s[0]))
    done = []
    for suffix, tests in groups.items():
        if len(tests) < min_tests: continue
        n = 0
        for cls, meth, tail in tests:
            f = next((f for f in test_files if f.stem == cls.split(".")[-1].split("$")[0]), None)
            if not f: continue
            text = f.read_text(errors="ignore"); lines = text.split("\n")
            hit = next((m for m in find_methods(text) if m[0] == meth.split("(")[0]), None)
            if not hit: continue
            body = "\n".join(lines[hit[1] - 1:hit[2]]); needle = tail + '"'
            if needle in body:
                new = body.replace(needle, tail + suffix + '"'); lines[hit[1] - 1:hit[2]] = new.split("\n"); f.write_text("\n".join(lines)); n += 1
        if n: done.append((suffix, len(tests), n))
    return done


def route(unit, unit_text, message):
    """None = give to the delegate, else the reason why the unit is left for the strong agent."""
    if unit["kind"] != "method": return None
    ea = expected_actual(message); fr = ea and fragment(ea[0])
    if fr and fr not in unit_text: return f"expected text '{fr[:50]}' is not in the method (the value lives in a helper/constant elsewhere)"
    return None
