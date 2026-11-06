"""
Filter the mined cases (annotated_cases_<version>.csv) into a smaller, less repetitive dataset, and report what every
step does to the numbers (cases, repositories, commit pairs, concentration), so the effect can be quoted in a text.

    python data_processing/filter_cases.py annotated_cases_v2_pr59.csv annotated_cases_v2_pr59_filtered.csv
    python data_processing/filter_cases.py in.csv out.csv --cap 5 --report filter_report.json
    python data_processing/filter_cases.py in.csv --dry-run          # only the report, no output file

Standard library only. The input is read once, row by row (the file is several GB), and never modified; the output
is only written to a new path (an existing file is not overwritten without --force). Rows keep their order and are
written in the same format as the miner writes them ('|' delimited, everything quoted).

The report is written next to the output as <output>.report.json (machine readable) and <output>.report.md (a table
for the text), and is printed.

Steps (applied in this order; each one sees only the cases the previous ones kept; --steps chooses and orders them)
  1. environment_errors
     Drops cases whose failing test log ends in an error caused by the machine and not by the source change:
       - a missing THIRD-PARTY module (`ModuleNotFoundError: No module named 'mcp'`); a module that is the project's
         own (its name is in the repository name or the test path, or the repair edits it) is a real break and stays;
       - network errors (ConnectionError, ReadTimeout, HTTPError, SSLError, URLError, ... as the exception class);
       - missing credentials / API keys (not for TypeError, AssertionError and other code-level errors);
       - a missing executable (`ffmpeg`, `ffprobe`, `docker`, ... not found), pytest reading stdin, PermissionError.
     The patterns are deliberately strict (high precision): FileNotFoundError is NOT dropped, because it is mostly
     a test that looks for a file it creates itself. The number of such cases is reported in the details.
  2. non_repairs
     Drops cases whose "repair" is not a repair of the code under test: the repair ADDS skip / xfail / pass (the test
     is disabled), the repaired block is under 30% of the broken block (mostly deleted), or the edit changes more than
     40 lines (a rewrite). The reasons are counted separately in the details (a case can match several).
  3. same_pair_error_change_cap
     A commit pair (repository, broken commit, repaired commit) often yields many cases that are the same repair
     repeated over different tests: a renamed function breaks 190 tests with `NameError: name 'main' is not defined`
     and every test is fixed by `main` -> `cli_module.main`. Cases are grouped by
        (commit pair, final error line of the failing test log, change pattern)
     and only the first `cap` cases of every group are kept (file order).
       - final error line: the last line of the pytest log that starts with "E" (e.g. "NameError: name 'x' is
         not defined"), compared exactly;
       - change pattern: the set of token substitutions that turns the [<BREAKAGE>] block into the
         [<REPAIREDTEST>] block (e.g. {("main",) -> ("cli_module", ".", "main")}); only the first MAX_BLOCK
         characters of both blocks are compared.
     Cases with a different error, or a different change pattern, are different repairs and are never merged.
"""
import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

DEFAULT_CAP = 5
MAX_BLOCK = 4000      # characters of the breakage / repaired block that are compared
MAX_OP_TOKENS = 6     # tokens of one side of a substitution that are compared
TOKEN = re.compile(r"\w+|[^\w\s]")
ERROR_LINE = re.compile(r"^E\s")


def between(text, start, end):
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i)
    return text[i:j] if j >= 0 else text[i:]


def final_error(log):
    """Last line of the test log that starts with "E" (pytest's error lines), stripped; "" when there is none."""
    last = ""
    for line in log.split("\n"):
        if ERROR_LINE.match(line) and line[1:].strip():
            last = line[1:].strip()
    return last


def change_pattern(annotated_code):
    """The set of token substitutions between the [<BREAKAGE>] block and the [<REPAIREDTEST>] block."""
    broken = TOKEN.findall(between(annotated_code, "[<BREAKAGE>]", "[</BREAKAGE>]")[:MAX_BLOCK])
    fixed = TOKEN.findall(between(annotated_code, "[<REPAIREDTEST>]", "[</REPAIREDTEST>]")[:MAX_BLOCK])
    matcher = difflib.SequenceMatcher(None, broken, fixed, autojunk=False)
    ops = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            ops.add((tuple(broken[i1:i2][:MAX_OP_TOKENS]), tuple(fixed[j1:j2][:MAX_OP_TOKENS])))
    return frozenset(ops)


def pattern_of(row):
    """change_pattern of a row, computed once (the steps share it)."""
    if "_pattern" not in row:
        row["_pattern"] = change_pattern(row["annotated_code"])
    return row["_pattern"]


def removed_tokens(row):
    return {token for old, _new in pattern_of(row) for token in old}


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


EXCEPTION = r"^(?:\w+\.)*"
CODE_LEVEL_ERRORS = r"^(?:\w+\.)*(TypeError|AssertionError|AttributeError|NameError|KeyError|IndexError|ValueError|ZeroDivisionError)\b"
MISSING_MODULE = re.compile(r"No module named '([\w.]+)'")
ENVIRONMENT_RULES = [
    ("network", re.compile(EXCEPTION + r"(ConnectionError|ConnectTimeout|ReadTimeout|HTTPError|SSLError|URLError|MaxRetryError|"
                           r"gaierror|ConnectionRefusedError|ConnectionResetError)\b")),
    ("missing executable", re.compile(r"command not found|not found in PATH|executable file not found|"
                                      r"\[Errno 2\] No such file or directory: '(ffmpeg|ffprobe|docker|git|node|npm|java|gcc|make|ssh|curl)'",
                                      re.I)),
    ("permission / stdin", re.compile(EXCEPTION + r"PermissionError\b|reading from stdin while output is captured")),
]
CREDENTIAL_SUBJECT = re.compile(r"api[_ ]?key|credential|access[_ ]token|auth(entication)? (failed|error)|environment variable", re.I)
CREDENTIAL_PROBLEM = re.compile(r"missing|not set|must be set|is required|not provided|invalid|incorrect|unauthorized", re.I)
FILE_NOT_FOUND = re.compile(r"FileNotFoundError|No such file or directory")


def environment_reason(row):
    """Why the failing log shows an environment problem ('' when it does not)."""
    message = final_error(row["outdated_test_log"])
    module = MISSING_MODULE.search(message)
    if module and re.match(r"(ModuleNotFoundError|ImportError)", message):
        parts = module.group(1).split(".")
        own = normalized(row["repository_name"]) + " " + normalized(row["relative_path"])
        if any(part in removed_tokens(row) for part in parts) or normalized(parts[0]) in own:
            return ""
        return "missing third-party module"
    for name, pattern in ENVIRONMENT_RULES:
        if pattern.search(message):
            return name
    if (not re.match(CODE_LEVEL_ERRORS, message) and CREDENTIAL_SUBJECT.search(message)
            and CREDENTIAL_PROBLEM.search(message)):
        return "missing credentials / API key"
    return ""


class EnvironmentErrors:
    name = "environment_errors"
    description = ("drop cases whose failing test log ends in a machine-side error: missing third-party module, "
                   "network error, missing credentials, missing executable, permission / stdin")
    parameters = {}

    def __init__(self):
        self.reasons = Counter()
        self.file_not_found_kept = 0

    def keep(self, row):
        reason = environment_reason(row)
        if reason:
            self.reasons[reason] += 1
            return False
        if FILE_NOT_FOUND.search(final_error(row["outdated_test_log"])):
            self.file_not_found_kept += 1
        return True

    def extra(self):
        return {"dropped_by_reason": dict(self.reasons.most_common()),
                "file_not_found_kept_not_dropped": self.file_not_found_kept}


SKIP_MARKERS = re.compile(r"pytest\.skip|@pytest\.mark\.skip|@unittest\.skip|skipTest|xfail|pytest\.importorskip|^\s*pass\s*$", re.M)
MIN_BLOCK_FOR_DELETION = 200
DELETION_RATIO = 0.3
MAX_CHANGED_LINES = 40


def changed_lines(broken, fixed):
    """Number of lines that differ between the broken and the repaired block (ignoring blank lines and indentation)."""
    a = [line.strip() for line in broken[:MAX_BLOCK].split("\n") if line.strip()]
    b = [line.strip() for line in fixed[:MAX_BLOCK].split("\n") if line.strip()]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum((i2 - i1) + (j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


class NonRepairs:
    name = "non_repairs"
    description = ("drop cases whose repair is not a repair of the code under test: it adds skip/xfail/pass, "
                   "it mostly deletes the test (repaired block under %d%% of the broken one), or it rewrites it "
                   "(more than %d changed lines)" % (DELETION_RATIO * 100, MAX_CHANGED_LINES))
    parameters = {"deletion_ratio": DELETION_RATIO, "min_block_for_deletion": MIN_BLOCK_FOR_DELETION,
                  "max_changed_lines": MAX_CHANGED_LINES}

    def __init__(self):
        self.reasons = Counter()
        self.dropped = 0

    def keep(self, row):
        code = row["annotated_code"]
        broken = between(code, "[<BREAKAGE>]", "[</BREAKAGE>]")
        fixed = between(code, "[<REPAIREDTEST>]", "[</REPAIREDTEST>]")
        reasons = []
        if SKIP_MARKERS.search(fixed) and not SKIP_MARKERS.search(broken):
            reasons.append("adds skip/xfail/pass")
        if len(broken.strip()) > MIN_BLOCK_FOR_DELETION and len(fixed.strip()) < DELETION_RATIO * len(broken.strip()):
            reasons.append("mostly deletion")
        if changed_lines(broken, fixed) > MAX_CHANGED_LINES:
            reasons.append("more than %d changed lines" % MAX_CHANGED_LINES)
        for reason in reasons:
            self.reasons[reason] += 1
        if reasons:
            self.dropped += 1
        return not reasons

    def extra(self):
        return {"cases_dropped": self.dropped, "matches_per_reason (a case can match several)": dict(self.reasons.most_common())}


def digest(*parts):
    h = hashlib.blake2b(digest_size=12)
    for part in parts:
        h.update(repr(part).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.digest()


class CapSamePairErrorChange:
    name = "same_pair_error_change_cap"

    def __init__(self, cap=DEFAULT_CAP):
        self.cap = cap
        self.groups = Counter()
        self.description = ("keep at most %d cases per group of (commit pair, final error line, change pattern); "
                            "see the module docstring" % cap)
        self.parameters = {"cap": cap, "max_block_chars": MAX_BLOCK, "max_op_tokens": MAX_OP_TOKENS}

    def keep(self, row):
        key = digest(row["repository_name"], row["broken_hash"], row["repaired_hash"],
                     final_error(row["outdated_test_log"]), sorted(pattern_of(row)))
        self.groups[key] += 1
        return self.groups[key] <= self.cap

    def extra(self):
        sizes = list(self.groups.values())
        return {"groups": len(sizes), "groups_with_more_than_cap": sum(1 for s in sizes if s > self.cap),
                "largest_group": max(sizes) if sizes else 0}


STEP_FACTORIES = {
    EnvironmentErrors.name: lambda cap: EnvironmentErrors(),
    NonRepairs.name: lambda cap: NonRepairs(),
    CapSamePairErrorChange.name: lambda cap: CapSamePairErrorChange(cap),
}
DEFAULT_STEPS = [EnvironmentErrors.name, NonRepairs.name, CapSamePairErrorChange.name]


class Tally:
    """Numbers describing a set of cases, collected while the rows go by."""

    def __init__(self):
        self.cases = 0
        self.per_repo = Counter()
        self.per_pair = Counter()

    def add(self, row):
        self.cases += 1
        self.per_repo[row["repository_name"]] += 1
        self.per_pair[(row["repository_name"], row["broken_hash"], row["repaired_hash"])] += 1

    def summary(self):
        repos = sorted(self.per_repo.values(), reverse=True)
        pairs = sorted(self.per_pair.values(), reverse=True)
        cases = self.cases

        def share(values, k):
            return round(100.0 * sum(values[:k]) / cases, 2) if cases else 0.0
        return {
            "cases": cases,
            "repositories": len(repos),
            "commit_pairs": len(pairs),
            "cases_per_repo_mean": round(cases / len(repos), 2) if repos else 0.0,
            "cases_per_repo_median": repos[len(repos) // 2] if repos else 0,
            "cases_per_repo_max": repos[0] if repos else 0,
            "cases_per_pair_mean": round(cases / len(pairs), 2) if pairs else 0.0,
            "cases_per_pair_median": pairs[len(pairs) // 2] if pairs else 0,
            "cases_per_pair_max": pairs[0] if pairs else 0,
            "top3_repositories_share_percent": share(repos, 3),
            "top10_repositories_share_percent": share(repos, 10),
            "top3_pairs_share_percent": share(pairs, 3),
        }


def run(src, dst, steps, dry_run=False):
    """Stream src through the steps; write the kept rows to dst (unless dry_run). Returns the report dict."""
    tallies = [Tally() for _ in range(len(steps) + 1)]    # [0] = input, [i + 1] = after step i
    out = None
    try:
        with open(src, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            header = next(reader)
            index = {name: i for i, name in enumerate(header)}
            if not dry_run:
                out = open(dst, "w", newline="", encoding="utf-8")
                writer = csv.writer(out, delimiter="|", quoting=csv.QUOTE_ALL)
                writer.writerow(header)
            for values in reader:
                if len(values) != len(header):
                    raise ValueError("row %d has %d fields, expected %d" % (tallies[0].cases + 2, len(values), len(header)))
                row = {name: values[i] for name, i in index.items()}
                tallies[0].add(row)
                kept = True
                for i, step in enumerate(steps):
                    if not step.keep(row):
                        kept = False
                        break
                    tallies[i + 1].add(row)
                if kept and not dry_run:
                    writer.writerow(values)
    finally:
        if out is not None:
            out.close()
    report = {"input": os.path.abspath(src), "output": None if dry_run else os.path.abspath(dst),
              "initial": tallies[0].summary(), "steps": []}
    for i, step in enumerate(steps):
        before, after = tallies[i].summary(), tallies[i + 1].summary()
        report["steps"].append({
            "name": step.name, "description": step.description, "parameters": step.parameters,
            "cases_before": before["cases"], "cases_after": after["cases"],
            "removed": before["cases"] - after["cases"],
            "removed_percent": round(100.0 * (before["cases"] - after["cases"]) / before["cases"], 2) if before["cases"] else 0.0,
            "before": before, "after": after, "details": step.extra()})
    report["final"] = tallies[-1].summary()
    return report


def markdown(report):
    lines = ["| Step | Cases before -> after | Removed | Repositories | Commit pairs | Max cases / pair | Top-3 repos share |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    s = report["initial"]
    lines.append("| initial | %d | | %d | %d | %d | %.1f%% |" % (
        s["cases"], s["repositories"], s["commit_pairs"], s["cases_per_pair_max"], s["top3_repositories_share_percent"]))
    for step in report["steps"]:
        a = step["after"]
        lines.append("| %s | %d -> %d | -%d (-%.1f%%) | %d | %d | %d | %.1f%% |" % (
            step["name"], step["cases_before"], step["cases_after"], step["removed"], step["removed_percent"],
            a["repositories"], a["commit_pairs"], a["cases_per_pair_max"], a["top3_repositories_share_percent"]))
    lines.append("")
    for step in report["steps"]:
        lines.append("%s: %d -> %d" % (step["name"], step["cases_before"], step["cases_after"]))
    lines.append("")
    lines.append("Chain: " + " -> ".join([str(report["initial"]["cases"])] + [
        "%s: %d" % (step["name"], step["cases_after"]) for step in report["steps"]]))
    for step in report["steps"]:
        lines.append("")
        lines.append("`%s`: %s %s" % (step["name"], step["description"], json.dumps(step["details"])))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", help="filtered CSV (not needed with --dry-run)")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="cases kept per group (default %d)" % DEFAULT_CAP)
    parser.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                        help="comma separated steps, applied in this order (default %s; available: %s)"
                             % (",".join(DEFAULT_STEPS), ",".join(STEP_FACTORIES)))
    parser.add_argument("--report", help="path of the JSON report (default <output>.report.json)")
    parser.add_argument("--dry-run", action="store_true", help="compute the report only, write no CSV")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args(argv)
    if args.cap < 1:
        parser.error("--cap must be at least 1")
    names = [name.strip() for name in args.steps.split(",") if name.strip()]
    unknown = [name for name in names if name not in STEP_FACTORIES]
    if not names or unknown:
        parser.error("unknown or empty --steps %s (available: %s)" % (unknown or args.steps, ",".join(STEP_FACTORIES)))
    steps = [STEP_FACTORIES[name](args.cap) for name in names]
    if not args.dry_run:
        if not args.output:
            parser.error("an output path is needed unless --dry-run is given")
        if os.path.abspath(args.output) == os.path.abspath(args.input):
            parser.error("the output must not be the input file")
        if os.path.exists(args.output) and not args.force:
            parser.error("%s already exists (use --force to overwrite)" % args.output)
    report_path = args.report or ((args.output + ".report.json") if args.output else None)
    report = run(args.input, args.output, steps, dry_run=args.dry_run)
    text = markdown(report)
    print(text)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(os.path.splitext(report_path)[0] + ".md", "w", encoding="utf-8") as f:
            f.write(text)
        print("report written to", report_path, "and", os.path.splitext(report_path)[0] + ".md")


if __name__ == "__main__":
    main()
