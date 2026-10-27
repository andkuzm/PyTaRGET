"""
Summarize the funnel CSV the miner writes next to the dataset (<dataset name>_funnel.csv).

    python data_processing/funnel_report.py annotated_cases_trial_pr58_funnel.csv

Standard library only, so it runs in the bare environment of the searcher.

It prints, summed over all repositories and over every attempt (a resumed repository writes its rows again for
the part it processed):
  - how the repositories ended: searched, no tests, failed, or started without ever finishing (killed by the
    time limit or still running)
  - per tenth of the history (0 = oldest): the pairs planned, looked at, with a changed test, how many tests passed
    at the parent and at the child, the cases found, and the verdicts behind every loss
"""
import csv
import json
import sys
from collections import Counter, defaultdict

MAIN = ["commits", "relevant_commits", "relevant_pairs", "visited_pairs", "pairs_with_changed_test",
        "changed_tests", "parent_success", "child_success", "cases"]
SHORT = {"commits": "commits", "relevant_commits": "rel.commits", "relevant_pairs": "planned",
         "visited_pairs": "looked_at", "pairs_with_changed_test": "pairs+chg", "changed_tests": "tests",
         "parent_success": "parent_ok", "child_success": "child_ok", "cases": "CASES"}


def load(path):
    csv.field_size_limit(2 ** 31 - 1)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="|"))


def summarize(rows):
    """(stage counts, {bucket: Counter of every counter}) from the rows of a funnel CSV."""
    last_stage = {}
    buckets = defaultdict(Counter)
    for row in rows:
        if row["bucket"] == "repo":
            last_stage[row["repository_name"]] = row["stage"]  # file order: started, then how it ended
            continue
        counts = buckets[row["bucket"]]
        for column in MAIN:
            counts[column] += int(row.get(column) or 0)
        for key, value in json.loads(row.get("details") or "{}").items():
            counts[key] += int(value)
    stages = Counter("never finished" if stage == "started" else stage for stage in last_stage.values())
    return stages, buckets


def render(stages, buckets):
    lines = []
    total = sum(stages.values())
    lines.append(f"repositories: {total}  " + "  ".join(f"{name}: {count}" for name, count in sorted(stages.items())))
    if not buckets:
        lines.append("no search rows yet")
        return "\n".join(lines)

    extra = sorted({key for counts in buckets.values() for key in counts if key not in MAIN})
    extra = [key for key in extra if any(counts[key] for counts in buckets.values())]
    columns = MAIN + extra
    names = [SHORT.get(column, column) for column in columns]
    ordered = sorted(buckets, key=lambda b: (int(b) if b.lstrip("-").isdigit() else 99))
    sums = Counter()
    table = [["bucket"] + names]
    for bucket in ordered:
        table.append([bucket] + [str(buckets[bucket][column]) for column in columns])
        sums.update(buckets[bucket])
    table.append(["all"] + [str(sums[column]) for column in columns])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines.append("")
    lines.append("per tenth of the history (0 = oldest), all repositories together:")
    for n, row in enumerate(table):
        lines.append("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))
        if n == 0 or n == len(table) - 2:
            lines.append("  ".join("-" * width for width in widths))

    tests = sums["changed_tests"]
    if tests:
        lines.append("")
        lines.append(f"of {tests} changed tests: {sums['parent_success']} passed at the parent "
                     f"({100 * sums['parent_success'] / tests:.0f}%), {sums['child_success']} also at the child, "
                     f"{sums['cases']} became cases ({100 * sums['cases'] / tests:.0f}%)")
        lost = Counter({k: v for k, v in sums.items() if k.startswith(("parent_", "child_", "override_"))
                        and k not in ("parent_success", "child_success")})
        if lost:
            lines.append("lost: " + ", ".join(f"{key} {value}" for key, value in lost.most_common()))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python funnel_report.py <dataset name>_funnel.csv")
    print(render(*summarize(load(sys.argv[1]))))
