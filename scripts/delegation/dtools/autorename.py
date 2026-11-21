"""Deterministic plain-replace step (no model): finds pure identifier renames in SOURCE_CHANGES.diff and applies them to the test sources.
A rename = a removed line and an added line of the same change block that differ in exactly ONE identifier (method, constant, field, type); a deleted file paired with an added file of the
same shape = class rename. Renames are applied with word boundaries to src/test only. usage: tools/autorename [--dry-run]  ; prints what it replaced and how often."""
import difflib
import re
import sys
from pathlib import Path

ID = re.compile(r"[A-Za-z_]\w*")
STOP = {"get", "set", "is", "add", "put", "run", "of", "to", "id", "value", "name", "size", "type", "data", "list", "map", "test", "main", "new", "this", "null", "true", "false", "return"}
SKIP_KW = {"public", "private", "protected", "static", "final", "void", "int", "long", "double", "boolean", "String", "class", "interface", "enum", "return", "new", "if", "for", "while", "else", "throw", "throws"}


def blocks(diff):
    """yield (file, removed_lines, added_lines) per contiguous change block."""
    cur, rem, add = None, [], []
    for l in diff.split("\n"):
        if l.startswith("+++ "): cur = l[4:].split("\t")[0]; continue
        if l.startswith("--- "): continue
        if l.startswith("-"): 
            if add: yield cur, rem, add; rem, add = [], []
            rem.append(l[1:])
        elif l.startswith("+"): add.append(l[1:])
        else:
            if rem or add: yield cur, rem, add
            rem, add = [], []
    if rem or add: yield cur, rem, add


def ident_diffs(a, b):
    ta, tb = ID.findall(a), ID.findall(b)
    if len(ta) != len(tb) or ta == tb: return []
    if ID.sub("_", a) != ID.sub("_", b): return []          # the non-identifier skeleton must match too
    return [(x, y) for x, y in zip(ta, tb) if x != y]


def find_renames(diff):
    ren = {}
    for f, rem, add in blocks(diff):
        if not f or "src/main" not in f or len(rem) != len(add): continue
        for a, b in zip(rem, add):
            for x, y in ident_diffs(a, b):
                if x not in SKIP_KW and y not in SKIP_KW: ren.setdefault(x, y)
    # class renames: file removed + file added with similar text
    gone = {}
    added = {}
    for m in re.finditer(r"^diff -ruN a/(\S+) b/(\S+)\n--- (\S+)\s.*\n\+\+\+ (\S+)\s", diff, re.M): pass
    return ren


def class_renames(ws, diff):
    old = {}; new = {}
    parts = re.split(r"^diff -ruN ", diff, flags=re.M)[1:]
    for p in parts:
        head = p.split("\n", 1)[0]; body = p
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m or not m.group(1).endswith(".java") or "src/main" not in m.group(1): continue
        hunk = "\n".join(l for l in body.split("\n")[3:])
        if re.search(r"^@@ -0,0 ", hunk, re.M): new[m.group(2)] = "\n".join(l[1:] for l in hunk.split("\n") if l.startswith("+"))
        elif re.search(r"^@@ -1,\d+ \+0,0 ", hunk, re.M): old[m.group(1)] = "\n".join(l[1:] for l in hunk.split("\n") if l.startswith("-"))
    out = {}
    for o, ot in old.items():
        pass
    for o, ot in old.items():
        best = max(((difflib.SequenceMatcher(None, re.sub(r"\b\w+\b", "", ot) and ot, nt).ratio(), n) for n, nt in new.items()), default=(0, None))
        if best[0] > 0.7:
            out[Path(o).stem] = Path(best[1]).stem
            al, bl = ot.split("\n"), new[[n for n in new if Path(n).stem == Path(best[1]).stem][0]].split("\n")
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, al, bl, autojunk=False).get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:
                    for x, y in zip(al[i1:i2], bl[j1:j2]):
                        for p_, q_ in ident_diffs(x, y): out.setdefault(p_, q_)
    return out


def main():
    ws = Path(".").resolve(); dry = "--dry-run" in sys.argv
    diff = (ws / "SOURCE_CHANGES.diff").read_text(errors="replace")
    ren = find_renames(diff); cls = class_renames(ws, diff); ren.update(cls)
    added_ids = set(ID.findall("\n".join(l for l in diff.split("\n") if l.startswith("+"))))
    removed_ids = set(ID.findall("\n".join(l for l in diff.split("\n") if l.startswith("-"))))
    ren = {o: n for o, n in ren.items() if len(o) > 3 and o.lower() not in STOP and o != n and (o not in added_ids or o in cls) and n not in {x for x in ren if x != o} - set()}
    files = [f for f in (ws / "src" / "test").rglob("*.java")] if (ws / "src" / "test").exists() else [f for f in ws.rglob("src/test/**/*.java")]
    texts = {f: f.read_text(errors="replace") for f in files}
    done = []
    for o, n in sorted(ren.items()):
        pat = re.compile(r"\b" + re.escape(o) + r"\b")
        cnt = 0
        for f, t in texts.items():
            t2, k = pat.subn(n, t)
            if k: texts[f] = t2; cnt += k
        if cnt: done.append((o, n, cnt))
    if not dry:
        for f, t in texts.items():
            if t != f.read_text(errors="replace"): f.write_text(t)
    for o, n, c in done: print(f"{o} -> {n}: {c}")
    print(f"autorename: {len(done)} renames applied to tests ({sum(c for _, _, c in done)} occurrences)" + (" [dry run]" if dry else "") + "; run tools/run_tests next. Behaviour changes are NOT handled.")


if __name__ == "__main__":
    main()
