"""Shared helpers of the delegation tools (run in a task workspace; standard library only): configuration, an offline Maven run, a compact parse of its result, Java method location, assertion counting."""
import json
import os
import re
import subprocess
from pathlib import Path

ASSERT = re.compile(r"\bassert\w*\s*\(|\bverify\w*\s*\(|\bfail\s*\(|\bassertThat\b|\.should|\bexpect\w*\s*\(")
TEST_ANNOTATION = re.compile(r"^\s*@(Test|ParameterizedTest|RepeatedTest|TestFactory)\b")


def load_config(ws):
    return json.loads((Path(ws) / ".delegation" / "config.json").read_text())


def env_failing(ws):
    """{(simple outer class name, method)} of the tests that already failed in the agent sandbox on the old code (written by delegate.py); empty if unknown."""
    f = Path(ws) / ".delegation" / "env_failing.json"
    if not f.exists(): return set()
    return {(x.split("#")[0].split(".")[-1].split("$")[0], x.split("#")[1]) for x in json.loads(f.read_text())}


def drop_build_output(ws, only_tests=False):
    """remove Maven's build folders (a `target` next to a pom.xml; with only_tests just the compiled tests), so nothing compiled from an earlier version of a file can be run."""
    for pom in Path(ws).rglob("pom.xml"):
        for d in ([pom.parent / "target" / "test-classes", pom.parent / "target" / "generated-test-sources"] if only_tests else [pom.parent / "target"]):
            if d.is_dir(): shutil_rm(d)


def mvn(ws, tests=None, offline=True, timeout=1500):
    """Run the tests of the workspace with Maven. Returns (returncode, log); the log is saved in .delegation/last_test.log."""
    ws = Path(ws); cfg = load_config(ws)
    cmd = [str(Path(cfg["maven_home"]) / "bin" / "mvn"), "test", "-nsu", "--batch-mode", f"-Dmaven.repo.local={cfg['m2']}", "-DfailIfNoTests=false",
           "-Dsurefire.failIfNoSpecifiedTests=false", *cfg.get("mvn_extra", [])]
    if offline:
        cmd.insert(2, "-o")
    if tests:
        cmd.append("-Dtest=" + ",".join(tests))
    env = dict(os.environ, JAVA_HOME=cfg["java_home"], MAVEN_OPTS="-Xmx1g")
    env["PATH"] = cfg["java_home"] + "/bin:" + env["PATH"]
    for d in ws.rglob("surefire-reports"):
        shutil_rm(d)
    try:
        if cfg.get("jail"):                                    # new, untrusted project: sandbox (dtools/jail.py); offline=False keeps the network, for filling the cache only
            from dtools import jail
            p = jail.run(ws, cfg, cmd, network=not offline, timeout=timeout)
        else:
            p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=timeout)
        rc, log = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, log = 124, (e.stdout or b"").decode("utf-8", "ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
    (ws / ".delegation").mkdir(exist_ok=True)
    (ws / ".delegation" / "last_test.log").write_text(log)
    return rc, log


def shutil_rm(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def short(path, ws):
    p = str(path)
    root = str(Path(ws).resolve()) + "/"
    return p.replace(root, "").replace(str(ws) + "/", "")


def parse_compile_errors(log, ws):
    """{relative file: [(line, message)]} from the [ERROR] lines of a compilation failure; 'symbol:' / 'location:' continuation lines are appended to the message.
    Maven prints every error twice (error section and failure summary): each (file, line, message) is kept once."""
    errors, current, seen = {}, None, set()
    for line in log.split("\n"):
        m = re.match(r"\[ERROR\] (/\S+\.java):\[(\d+),(\d+)\] (.*)", line)
        if m:
            f, ln, msg = short(m.group(1), ws), int(m.group(2)), m.group(4).strip()
            if (f, ln, msg) in seen:
                current = None                                     # the repeated copy: skip it and its continuation lines
                continue
            seen.add((f, ln, msg)); errors.setdefault(f, []).append([ln, msg]); current = (f, len(errors[f]) - 1)
            continue
        c = re.match(r"(?:\[ERROR\])?\s+(symbol|location|required|found|reason):\s*(.*)", line)
        if c and current:
            errors[current[0]][current[1]][1] += f" [{c.group(1)}: {c.group(2).strip()}]"
        elif not line.startswith("[ERROR]"):
            current = current if line.strip() == "" else None
    return {f: [tuple(x) for x in v] for f, v in errors.items()}


def parse_surefire(ws):
    """[(class, method, status, message)] from target/surefire-reports/TEST-*.xml; status is pass / fail / error / skipped."""
    import xml.etree.ElementTree as ET
    out = []
    for f in sorted(Path(ws).rglob("surefire-reports/TEST-*.xml")):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for case in root.iter("testcase"):
            status, message = "pass", ""
            for tag in ("failure", "error", "skipped"):
                node = case.find(tag)
                if node is not None:
                    status = {"failure": "fail", "error": "error", "skipped": "skipped"}[tag]
                    lines = [l.strip() for l in ((node.get("message") or node.text or "").strip().split("\n")) if l.strip()]
                    message = " | ".join(lines[:8])[:1000]         # keeps "expected: ... | but was : ..." (Truth/JUnit) including multi-line values (the difference can be in a later line)
            out.append((case.get("classname"), case.get("name"), status, message))
    return out


def result_of(ws, rc, log):
    """('compile_error', {file: [(line, msg)]}) | ('tests', [surefire results]) | ('other', log tail)."""
    errs = parse_compile_errors(log, ws)
    if "COMPILATION ERROR" in log or "Compilation failure" in log or (errs and not any(Path(ws).rglob("surefire-reports/TEST-*.xml"))):
        return "compile_error", errs
    results = parse_surefire(ws)
    if results:
        return "tests", results
    return "other", "\n".join(log.strip().split("\n")[-15:])


def compact(ws, kind, data, rc, max_lines=14):
    """The few lines the strong agent reads after a test run."""
    ws = str(ws)
    if kind == "compile_error":
        n = sum(len(v) for v in data.values())
        lines = [f"COMPILE ERRORS: {n} in {len(data)} file(s). Tests did not run."]
        shown = 0
        for f, errs in data.items():
            for line, msg in errs[:3]:
                if shown >= max_lines: break
                lines.append(f"  {f}:{line}: {msg[:150]}"); shown += 1
            if len(errs) > 3: lines.append(f"  {f}: ... {len(errs) - 3} more")
        lines.append("  (full log: .delegation/last_test.log)")
        return "\n".join(lines)
    if kind == "tests":
        bad = [r for r in data if r[2] in ("fail", "error")]
        lines = [f"TESTS: {len(data)} run, {sum(1 for r in data if r[2]=='pass')} passed, {len(bad)} failed, {sum(1 for r in data if r[2]=='skipped')} skipped"]
        for c, m, s, msg in bad[:max_lines]:
            lines.append(f"  FAIL {c.split('.')[-1]}#{m}: {msg[:140]}")
        if len(bad) > max_lines: lines.append(f"  ... {len(bad) - max_lines} more")
        return "\n".join(lines)
    return "NO TEST RESULT (build problem). Last lines:\n" + data


# ---------------- Java source helpers ----------------
def _skip_literals(text, i):
    """index after a string/char literal or comment that starts at i, or i itself."""
    if text.startswith("//", i):
        j = text.find("\n", i); return len(text) if j < 0 else j
    if text.startswith("/*", i):
        j = text.find("*/", i + 2); return len(text) if j < 0 else j + 2
    if text[i] in "\"'":
        q, j = text[i], i + 1
        if text.startswith('"""', i):
            j = text.find('"""', i + 3); return len(text) if j < 0 else j + 3
        while j < len(text) and text[j] != q:
            j += 2 if text[j] == "\\" else 1
        return j + 1
    return i


def brace_delta(text):
    """(number of '{') - (number of '}') outside strings and comments; also returns False for a negative running depth (a '}' without '{')."""
    depth, k, ok = 0, 0, True
    while k < len(text):
        k2 = _skip_literals(text, k)
        if k2 != k: k = k2; continue
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth < 0: ok = False
        k += 1
    return depth, ok


def mask_comments_and_strings(text):
    """the same text with the contents of comments and string/char literals blanked (same length, same line breaks), so a word inside them is never read as code."""
    out, k = list(text), 0
    while k < len(text):
        k2 = _skip_literals(text, k)
        if k2 == k: k += 1; continue
        a, b = (k + 1, k2 - 1) if text[k] in "\"'" else (k, k2)            # keep the quotes
        for j in range(a, min(b, len(text))):
            if out[j] != "\n": out[j] = " "
        k = k2
    return "".join(out)


def find_methods(text):
    """[(name, first_line, last_line)] (1-based, annotations included) of the test methods of a Java file."""
    lines = mask_comments_and_strings(text).split("\n")
    offsets, pos = [], 0
    for l in lines:
        offsets.append(pos); pos += len(l) + 1
    methods = []
    for idx, l in enumerate(lines):
        if not TEST_ANNOTATION.match(l):
            continue
        sig = lambda line: re.sub(r"@\w+(\([^)]*\))?", "", line)      # a line without its annotations
        j = idx
        while j < len(lines) and not re.search(r"\b(\w+)\s*\(", sig(lines[j])):
            j += 1
        m = re.search(r"\b(\w+)\s*\(", sig(lines[j])) if j < len(lines) else None
        if not m: continue
        start = offsets[j]; i = text.find("{", start)
        if i < 0: continue
        depth, k = 0, i
        while k < len(text):
            k2 = _skip_literals(text, k)
            if k2 != k: k = k2; continue
            if text[k] == "{": depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0: break
            k += 1
        last = text.count("\n", 0, k) + 1
        methods.append((m.group(1), idx + 1, last))
    return methods


def count_asserts(text):
    return len(ASSERT.findall(mask_comments_and_strings(text)))   # (until 2026-11-28 a regex with DOTALL dropped everything after the first "//" of a method, URLs included)


def assertions_per_method(path):
    text = Path(path).read_text(errors="ignore"); lines = text.split("\n")
    return {name: count_asserts("\n".join(lines[a - 1:b])) for name, a, b in find_methods(text)}
