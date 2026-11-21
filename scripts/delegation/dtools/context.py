"""Automatic context for the delegate (the counterpart of the prompts of the prompt pipeline, without oracle breakage lines): which test unit to repair, its errors, the part of the failure
output that concerns it, and the source hunks of SOURCE_CHANGES.diff ranked by TF-IDF similarity to the test and its errors."""
import math
import re
from collections import Counter
from pathlib import Path

from . import common

FILE_UNIT_MAX_LINES = 400
HUNK_BUDGET_CHARS = 4500
FAILURE_BUDGET_CHARS = 1500


def tokens(text):
    out = []
    for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text):
        for part in re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").lower().split():
            if len(part) > 2: out.append(part)
    return out


def parse_hunks(diff_text):
    """[{'file': path, 'text': header + hunk}] from a unified diff (git or diff -u)."""
    hunks, header, current = [], "", None
    for line in diff_text.split("\n"):
        if line.startswith("--- ") or line.startswith("diff "):
            if current: hunks.append(current); current = None
            header = line if line.startswith("--- ") else ""
        elif line.startswith("+++ "):
            header += "\n" + line
        elif line.startswith("@@"):
            if current: hunks.append(current)
            current = {"file": header.split("\n")[-1][4:].strip(), "text": header + "\n" + line}
        elif current is not None:
            current["text"] += "\n" + line
    if current: hunks.append(current)
    return hunks


def rank_hunks(hunks, query, budget=HUNK_BUDGET_CHARS):
    """Top hunks by TF-IDF cosine similarity to the query, within a character budget (each hunk is cut to 1800 characters)."""
    docs = [Counter(tokens(h["text"])) for h in hunks]
    df = Counter(t for d in docs for t in d)
    n = max(len(docs), 1)
    idf = lambda t: math.log((n + 1) / (df.get(t, 0) + 1)) + 1
    q = Counter(tokens(query)); qn = math.sqrt(sum((c * idf(t)) ** 2 for t, c in q.items())) or 1
    scored = []
    for h, d in zip(hunks, docs):
        dn = math.sqrt(sum((c * idf(t)) ** 2 for t, c in d.items())) or 1
        scored.append((sum(q[t] * idf(t) * d[t] * idf(t) for t in q if t in d) / (qn * dn), h))
    out, used = [], 0
    for score, h in sorted(scored, key=lambda x: -x[0]):
        if score <= 0 or used >= budget: break
        text = h["text"][:1800]; out.append(text); used += len(text)
    return out


def test_files(ws, cfg):
    roots = [Path(ws) / r for r in cfg.get("test_roots", ["src/test/java"])]
    return [p for r in roots for p in r.rglob("*.java")]


def units_from_state(ws, kind, data, cfg):
    """Units to repair from the last test run. compile_error: one unit per test file with errors (whole file when it has at most FILE_UNIT_MAX_LINES lines, otherwise one unit per
    method with errors plus a 'header' unit for errors outside methods). tests: one unit per failing method."""
    ws = Path(ws); units = []
    if kind == "compile_error":
        for rel, errs in sorted(data.items()):
            path = ws / rel
            if not path.exists() or not any(str(path).startswith(str(ws / r)) for r in cfg.get("test_roots", ["src/test/java"])):
                continue                                           # an error in main source: not ours to repair
            text = path.read_text(errors="ignore"); n = text.count("\n") + 1
            if n <= FILE_UNIT_MAX_LINES:
                units.append({"kind": "file", "file": rel, "name": Path(rel).name, "errors": errs}); continue
            methods = common.find_methods(text); first = min([m[1] for m in methods] or [n])
            per = {}
            for line, msg in errs:
                hit = next((m for m in methods if m[1] <= line <= m[2]), None)
                key = hit[0] if hit else "<header>"
                per.setdefault(key, []).append((line, msg))
            for key, e in per.items():
                if key == "<header>":
                    units.append({"kind": "header", "file": rel, "name": Path(rel).name + " (header)", "errors": e, "last_line": first - 1 if first > 1 else n})
                else:
                    a, b = next((m[1], m[2]) for m in methods if m[0] == key)
                    units.append({"kind": "method", "file": rel, "name": f"{Path(rel).stem}#{key}", "method": key, "first": a, "last": b, "errors": e})
    elif kind == "tests":
        for cls, method, status, msg in data:
            if status not in ("fail", "error"): continue
            simple = cls.split(".")[-1].split("$")[0]
            path = next((p for p in test_files(ws, cfg) if p.stem == simple), None)
            if not path: continue
            text = path.read_text(errors="ignore")
            hit = next((m for m in common.find_methods(text) if m[0] == method.split("(")[0]), None)
            if hit:
                units.append({"kind": "method", "file": str(path.relative_to(ws)), "name": f"{simple}#{hit[0]}", "method": hit[0], "first": hit[1], "last": hit[2], "errors": [(0, msg)]})
    return units


def unit_text(ws, unit):
    lines = (Path(ws) / unit["file"]).read_text(errors="ignore").split("\n")
    if unit["kind"] == "file": return "\n".join(lines)
    if unit["kind"] == "header": return "\n".join(lines[:unit["last_line"]])
    return "\n".join(lines[unit["first"] - 1:unit["last"]])


def failure_for(log, unit):
    keep = [l for l in log.split("\n") if Path(unit["file"]).name in l or l.strip().startswith(("[ERROR]   symbol", "[ERROR]   location", "[ERROR]   required", "[ERROR]   found"))]
    text = "\n".join(re.sub(r"^\[ERROR\]\s?", "", l) for l in dict.fromkeys(keep))
    text = re.sub(r"/\S*/(src/)", r"\1", text)
    return text[:FAILURE_BUDGET_CHARS]


INSTRUCTION = (
    "You are given a Java test ({scope}) that no longer works because the project's main source code changed. "
    "Repair it so that it compiles and passes with the changed source, changing only what is needed. Keep everything the test checks: do not delete tests or assertions, "
    "adapt them to the new behaviour. Do not change main source code.\n"
    "Output the complete repaired {scope} (annotations included) inside [REPAIR] ... [/REPAIR]. If import lines must be added, put only the new import lines inside [IMPORTS] ... [/IMPORTS]. "
    "Do not output anything else.")


def build_prompt(ws, unit, log, hunks, mode="auto", strong=None):
    """mode: 'auto' (ranked hunks), 'strong' (the strong agent's source_changes/hint replace the ranked hunks), 'both' (ranked hunks plus the strong agent's additions)."""
    code = unit_text(ws, unit)
    query = code + " " + " ".join(m for _, m in unit["errors"])
    auto = rank_hunks(hunks, query)
    entry = (strong or {}).get(unit["name"]) or (strong or {}).get(unit["file"]) or {}
    if mode == "strong" and entry.get("source_changes"):
        changes = [entry["source_changes"]]
    elif mode == "both":
        changes = auto + ([entry["source_changes"]] if entry.get("source_changes") else [])
    else:
        changes = auto
    scope = {"file": "file", "header": "file header (package and imports)", "method": "method"}[unit["kind"]]
    errs = "\n".join(f"line {l}: {m}" if l else m for l, m in unit["errors"][:25])
    parts = [f"### Instruction:\n{INSTRUCTION.format(scope=scope)}", f"### Test Code ({unit['file']}):\n{code}", f"### Errors:\n{errs}"]
    fail = failure_for(log, unit)
    if fail: parts.append(f"### Test Failure Output:\n{fail}")
    parts.append("### Source Code Changes:\n" + "\n\n".join(changes))
    if mode in ("strong", "both") and entry.get("hint"): parts.append(f"### Hint:\n{entry['hint']}")
    return "\n\n".join(parts) + "\n\n### Repaired Code:\n"
