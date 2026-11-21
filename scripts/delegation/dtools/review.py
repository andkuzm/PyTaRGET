"""tools/review: compact review of what changed in the TEST files compared with the old tests (git HEAD of the workspace). For every test method it compares the assertions of the old and the new version
by their MEANING (assertion kinds and literal values), so that pure renames are counted but not shown; methods whose assertions changed are listed with the old and new assertion lines, with flags:
  ASSERTIONS DROPPED  - fewer assertion statements than before
  FAILURE CHECK LOST  - the old test expected a failure/false/exception (assertFalse, assertThrows, fail(...), expected=...) and the new one expects none
  REMOVED             - the old test has no counterpart any more
Use it after the tests pass: it is a cheap way to see where a repair (yours or the delegate's) changed what a test checks. usage: tools/review [file ...]"""
import difflib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtools import common  # noqa: E402

ASSERT = re.compile(r"\b(assert\w*|fail|verify|expect\w*)\s*\(")
FAILING = re.compile(r"\bassertFalse\b|\bassertThrows\b|\bfail\s*\(|\bexpected\s*=|\bassertNull\b|\bassertNotEquals\b")
LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\b\d+(?:\.\d+)?[LlFfDd]?\b')


def git(ws, *a):
    return subprocess.run(["git", *a], cwd=ws, capture_output=True, text=True).stdout


def methods(text):
    lines = text.split("\n")
    return {n: "\n".join(lines[a - 1:b]) for n, a, b in common.find_methods(text)}


def assertion_lines(body):
    out, cur = [], None
    for l in body.split("\n"):
        if ASSERT.search(l): cur = [l.strip()]; out.append(cur)
        elif cur and not l.strip().endswith((";", "{", "}")) or (cur and l.strip().startswith(("+", "."))): cur.append(l.strip())
        else: cur = None
    return [" ".join(x) for x in out]


def meaning(body):
    kinds = sorted(m.group(1) for m in ASSERT.finditer(body)); lits = sorted(LITERAL.findall(body))
    return kinds, lits


def main(ws, only):
    ws = Path(ws); cfg = common.load_config(ws); root = cfg["test_roots"][0]
    old_files = {p: git(ws, "show", f"HEAD:{p}") for p in git(ws, "ls-tree", "-r", "--name-only", "HEAD", root).split("\n") if p.endswith(".java")}
    same = changed = removed = 0; shown = []
    for f in sorted((ws / root).rglob("*.java")):
        rel = str(f.relative_to(ws)); new_text = f.read_text(errors="ignore")
        if only and not any(o in rel for o in only): continue
        old_text = old_files.pop(rel, None)
        if old_text is None:                                           # renamed file: the old file with most method names in common
            nm = set(methods(new_text))
            best = max(old_files, key=lambda p: len(nm & set(methods(old_files[p]))), default=None)
            if best and len(nm & set(methods(old_files[best]))) >= 1: rel_old, old_text = best, old_files.pop(best)
            else: continue
        if old_text == new_text: continue
        om, nm_ = methods(old_text), methods(new_text); unmatched_new = {k: v for k, v in nm_.items() if k not in om}
        for name, ob in om.items():
            nb = nm_.get(name); renamed_to = None
            if nb is None and unmatched_new:                           # renamed method: most similar unmatched new body
                score = lambda k: difflib.SequenceMatcher(None, ob, unmatched_new[k]).ratio() + (0.6 if bool(FAILING.search(ob)) == bool(FAILING.search(unmatched_new[k])) else 0)
                cand = max(unmatched_new, key=score)
                if difflib.SequenceMatcher(None, ob, unmatched_new[cand]).ratio() >= 0.5: nb = unmatched_new.pop(cand); renamed_to = cand
            if nb is None: removed += 1; shown.append((rel, name, ["REMOVED"], assertion_lines(ob), [])); continue
            if meaning(ob) == meaning(nb): same += 1; continue
            changed += 1; oa, na = assertion_lines(ob), assertion_lines(nb); flags = []
            if len(na) < len(oa): flags.append("ASSERTIONS DROPPED")
            if FAILING.search(ob) and not FAILING.search(nb): flags.append("FAILURE CHECK LOST")
            shown.append((rel, name + (f" -> {renamed_to}" if renamed_to else ""), flags, oa, na))
    print(f"review: {same} test methods changed only by renames (not shown), {changed} changed in meaning, {removed} removed; flagged first")
    shown.sort(key=lambda s: (not s[2], s[0]))
    flagged = [x for x in shown if x[2]]; plain = [x for x in shown if not x[2]]
    for rel, name, flags, oa, na in flagged[:20]:
        print(f"\n{Path(rel).name}#{name}   [{'; '.join(flags)}]")
        for l in oa[:4]: print("  - " + l[:150])
        for l in na[:4]: print("  + " + l[:150])
    if plain: print(f"\nchanged in meaning but not flagged ({len(plain)}; expected values/behaviour updated, nothing dropped): " + ", ".join(f"{Path(r).stem}#{n}" for r, n, *_ in plain[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
