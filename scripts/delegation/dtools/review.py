"""tools/review: compact review of what changed in the TEST files compared with the old tests (git HEAD of the workspace). For every test method it compares the assertions of the old and the new version
by their MEANING (assertion kinds and literal values), so that pure renames are counted but not shown; methods whose assertions changed are listed with the old and new assertion lines, with flags:
  ASSERTIONS DROPPED  - fewer assertion statements than before
  FAILURE CHECK LOST  - the old test expected a failure/false/exception (assertFalse, assertThrows, fail(...), expected=...) and the new one expects none
  REMOVED             - the old test has no counterpart any more
  SANDBOX-ONLY FAILURE EDITED - the test already failed in the agent sandbox on the OLD code (environment) and its code was changed: only what compiling needs is acceptable
  EXPECTED EXCEPTION LOST - the old test expected an exception (expected=, assertThrows, fail ...) and the new one expects no exception at all (an assertFalse does not replace it), or now expects a ClassCastException, which a cast in the test itself can raise
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
# the old test expects an EXCEPTION; an assertFalse / assertNull in the new one does not replace that (fix R3)
THROWS = re.compile(r"\bexpected\s*=|\bassertThrows\b|\bexpectThrows\b|\bassertThatThrownBy\b|\bassertThatExceptionOfType\b|\bassertThatCode\b|\.expect\(\s*\w*(?:Exception|Error)\b|\bfail\s*\(")
CAST = re.compile(r"(?:expected\s*=\s*|assertThrows\(\s*)ClassCastException\b")
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


DECL_METHOD = re.compile(r"^\s*(?:(?:public|protected|private|static|final|abstract|synchronized|default|native)\s+)*(?:<[^>]+>\s*)?[\w.\[\]<>?, ]+?\s+(\w+)\s*\(")
DECL_TYPE = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Z]\w*)")
NOT_DECL = re.compile(r"^\s*(return|new|else|if|for|while|switch|throw|case|catch)\b")


def declared(lines):
    out = set()
    for l in lines:
        if l.lstrip().startswith(("//", "*", "/*")): continue
        out.update(DECL_TYPE.findall(l))
        m = None if NOT_DECL.match(l) else DECL_METHOD.match(l)
        if m: out.add(m.group(1))
    return out


def removed_declarations(ws):
    """names declared on removed lines of SOURCE_CHANGES.diff that are declared nowhere in the current src/main: a HINT that tested code may be gone (not proof: signatures can also change)."""
    diff = ws / "SOURCE_CHANGES.diff"
    if not diff.exists(): return set()
    gone = declared(l[1:] for l in diff.read_text(errors="ignore").split("\n") if l.startswith("-") and not l.startswith("---"))
    if not gone: return set()
    now = set()
    for f in ws.rglob("*.java"):
        if "src/main" in str(f.relative_to(ws)): now |= declared(f.read_text(errors="ignore").split("\n"))
    return gone - now


def meaning(body):
    kinds = sorted(m.group(1) for m in ASSERT.finditer(body)); lits = sorted(LITERAL.findall(body))
    return kinds, lits, sorted(re.findall(r"expected\s*=\s*([\w.]+)", body))          # the expected exception is part of what a test checks


def code_only(body):
    """the code of a method without comments and layout."""
    out, k = [], 0
    while k < len(body):
        k2 = common._skip_literals(body, k)
        if k2 == k: out.append(body[k]); k += 1; continue
        if not body.startswith(("//", "/*"), k): out.append(body[k:k2])
        k = k2
    return re.sub(r"\s+", "", "".join(out))


def main(ws, only):
    ws = Path(ws); cfg = common.load_config(ws); root = cfg["test_roots"][0]
    old_files = {p: git(ws, "show", f"HEAD:{p}") for p in git(ws, "ls-tree", "-r", "--name-only", "HEAD", root).split("\n") if p.endswith(".java")}
    same = changed = removed = 0; shown = []; bodies = {}
    env = common.env_failing(ws); env_seen = set(); ENV = "SANDBOX-ONLY FAILURE EDITED"
    for f in sorted((ws / root).rglob("*.java")):
        rel = str(f.relative_to(ws)); new_text = f.read_text(errors="ignore")
        if only and not any(o in rel for o in only): continue
        old_text = old_files.pop(rel, None); rel_old = None
        if old_text is None:                                           # renamed file: the old file with most method names in common
            nm = set(methods(new_text))
            best = max((p for p in old_files if not (ws / p).exists()), key=lambda p: len(nm & set(methods(old_files[p]))), default=None)
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
            key = (Path(rel_old if rel_old else rel).stem, name); is_env = key in env
            if is_env: env_seen.add(key)
            if nb is None: removed += 1; bodies[(rel, name)] = ob; shown.append((rel, name, ["REMOVED"] + ([ENV] if is_env else []), assertion_lines(ob), [])); continue
            env_edit = is_env and code_only(ob) != code_only(nb)                # any edit of the code counts, also one that keeps all assertions
            if meaning(ob) == meaning(nb) and not env_edit: same += 1; continue
            changed += 1; oa, na = assertion_lines(ob), assertion_lines(nb); flags = [ENV] if env_edit else []
            if len(na) < len(oa): flags.append("ASSERTIONS DROPPED")
            if FAILING.search(ob) and not FAILING.search(nb): flags.append("FAILURE CHECK LOST")
            if THROWS.search(ob) and not THROWS.search(nb): flags.append("EXPECTED EXCEPTION LOST")
            elif CAST.search(nb) and not CAST.search(ob): flags.append("EXPECTED EXCEPTION LOST")       # an exception that a cast in the test itself can raise
            bodies[(rel, name + (f" -> {renamed_to}" if renamed_to else ""))] = ob
            shown.append((rel, name + (f" -> {renamed_to}" if renamed_to else ""), flags, oa, na))
    for rel, old_text in sorted(old_files.items()):                    # test files that are gone altogether
        if (ws / rel).exists() or (only and not any(o in rel for o in only)): continue
        for name, ob in methods(old_text).items():
            if not (ASSERT.search(ob) or re.search(r"@(Test|ParameterizedTest|RepeatedTest)\b", ob)): continue
            key = (Path(rel).stem, name); env_seen.add(key)
            removed += 1; bodies[(rel, name)] = ob; shown.append((rel, name, ["REMOVED"] + ([ENV] if key in env else []), assertion_lines(ob), []))
    for cls, name in sorted(env - env_seen):                            # a failure without a test method of that name (class set-up): any change of the file counts
        for f in (ws / root).rglob(cls + ".java"):
            rel = str(f.relative_to(ws)); before = git(ws, "show", f"HEAD:{rel}")
            if before and not any(n == name for n, *_ in common.find_methods(before)) and code_only(before) != code_only(f.read_text(errors="ignore")):
                shown.append((rel, name or "(class)", [ENV], [], []))
    gone = removed_declarations(ws)
    print(f"review: {same} test methods changed only by renames (not shown), {changed} changed in meaning, {removed} removed; flagged first")
    shown.sort(key=lambda s: (not s[2], s[0]))
    flagged = [x for x in shown if x[2]]; plain = [x for x in shown if not x[2]]
    for rel, name, flags, oa, na in flagged[:20]:
        print(f"\n{Path(rel).name}#{name}   [{'; '.join(flags)}]")
        for l in oa[:4]: print("  - " + l[:150])
        for l in na[:4]: print("  + " + l[:150])
        used = sorted(g for g in gone if re.search(r"\b" + re.escape(g) + r"\b", bodies.get((rel, name), "")))
        if used: print("  (hint only: names the old test used that are no longer declared in src/main, removed OR renamed: " + ", ".join(used[:6]) + ")")
    if plain: print(f"\nchanged in meaning but not flagged ({len(plain)}; expected values/behaviour updated, nothing dropped): " + ", ".join(f"{Path(r).stem}#{n}" for r, n, *_ in plain[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
