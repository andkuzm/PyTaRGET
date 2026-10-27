"""
Remove the one-space artifact from annotated_cases.csv files made before the `annotate_code` fix.

`annotate_code` kept unchanged lines of the test as raw unified-diff lines, i.e. with the diff's leading " ".
Every unchanged line of [<TESTCONTEXT>] therefore had one extra space (" def test_x():"), and so did the unchanged
lines between changes that end up in both [<BREAKAGE>] and [<REPAIREDTEST>]. Changed lines were stripped correctly,
which left the repair with mixed indentation (4 spaces, then 5 for an unchanged line in the middle).

Those unchanged in-between lines are exactly the ones the breakage and the repair have in common, so they are
found by aligning the two blocks. The file is only ever written to a new path and refuses to run on data that
already looks clean, because stripping a second time would damage real indentation.

    python -m data_processing.clean_annotated_cases annotated_cases.csv annotated_cases_clean.csv
"""
import argparse
import csv
import re
from difflib import SequenceMatcher

CONTEXT = re.compile(r"(\[<TESTCONTEXT>])(.*?)(\[</TESTCONTEXT>])(.*?)(\[<REPAIREDTEST>])(.*?)(\[</REPAIREDTEST>])", re.S)
METHOD_HEAD = re.compile(r" (@|def |async def )")


def strip_prefix(line):
    return line[1:] if line.startswith(" ") else line


def context_flags(breakage, repair):
    """Which lines of the breakage and of the repair are unchanged context (identical in both, in order)."""
    in_breakage, in_repair = [False] * len(breakage), [False] * len(repair)
    for a, b, size in SequenceMatcher(None, breakage, repair, autojunk=False).get_matching_blocks():
        for k in range(size):
            if breakage[a + k].startswith(" "):  # context lines carry the diff's " ", changed lines never match it
                in_breakage[a + k] = in_repair[b + k] = True
    return in_breakage, in_repair


def has_artifact(annotated_code):
    """True when the method's first line is still prefixed (" def ...", " @...") in the test context."""
    match = CONTEXT.search(annotated_code)
    return bool(match) and any(METHOD_HEAD.match(line) for line in match.group(2).split("\n"))


def clean_annotated_code(annotated_code):
    """(cleaned text, number of stripped lines), or (None, 0) when the structure is not what annotate_code writes."""
    match = CONTEXT.search(annotated_code)
    if not match:
        return None, 0
    lines = match.group(2).split("\n")
    markers = [(i, line.rstrip("\r")) for i, line in enumerate(lines) if line.rstrip("\r") in ("[<BREAKAGE>]", "[</BREAKAGE>]")]
    if [m for _, m in markers] != ["[<BREAKAGE>]", "[</BREAKAGE>]"]:
        return None, 0
    start, end = markers[0][0], markers[1][0]
    repair_match = re.fullmatch(r"\n(.*)\n", match.group(6), re.S)
    if repair_match is None:
        return None, 0

    before, breakage, after = lines[:start], lines[start + 1:end], lines[end + 1:]
    repair = repair_match.group(1).split("\n")
    stripped = 0

    head = next((i for i, line in enumerate(before) if METHOD_HEAD.match(line)), None)
    if head is not None:  # everything above is the import block, which has no artifact
        stripped += len(before) - head
        before = before[:head] + [strip_prefix(line) for line in before[head:]]
    stripped += len(after)
    after = [strip_prefix(line) for line in after]

    in_breakage, in_repair = context_flags(breakage, repair)
    breakage = [strip_prefix(line) if flag else line for line, flag in zip(breakage, in_breakage)]
    repair = [strip_prefix(line) if flag else line for line, flag in zip(repair, in_repair)]
    stripped += sum(in_breakage)

    new_context = "\n".join(before + [lines[start]] + breakage + [lines[end]] + after)
    new_repair = "\n" + "\n".join(repair) + "\n"
    cleaned = (annotated_code[:match.start(2)] + new_context + match.group(3) + match.group(4) + match.group(5)
               + new_repair + annotated_code[match.start(7):])
    return cleaned, stripped


def clean_file(csv_in, csv_out, force=False):
    csv.field_size_limit(2**31 - 1)
    with open(csv_in, newline="", encoding="utf-8") as src:
        rows = list(csv.reader(src, delimiter="|"))
    header, body = rows[0], rows[1:]
    column = header.index("annotated_code")

    with_artifact = sum(has_artifact(row[column]) for row in body)
    if not force and with_artifact < len(body) / 2:
        raise SystemExit(f"Only {with_artifact}/{len(body)} rows show the artifact, the file looks already cleaned. "
                         "Refusing to strip real indentation (use force=True to override).")

    skipped = 0
    changed_lines = 0
    for row in body:
        cleaned, stripped = clean_annotated_code(row[column])
        if cleaned is None:
            skipped += 1
            continue
        row[column] = cleaned
        changed_lines += stripped

    with open(csv_out, "w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst, delimiter="|")
        writer.writerow(header)
        writer.writerows(body)
    print(f"{len(body)} rows, {skipped} left untouched (unexpected structure), {changed_lines} lines fixed -> {csv_out}")
    return len(body), skipped, changed_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in")
    parser.add_argument("csv_out")
    parser.add_argument("--force", action="store_true", help="clean even if the file looks already cleaned")
    args = parser.parse_args()
    clean_file(args.csv_in, args.csv_out, args.force)
