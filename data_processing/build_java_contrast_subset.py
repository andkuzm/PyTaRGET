"""
Java contrast subset from TaRBench (WP2): 150 test cases that the authors' CodeT5+ solved at RANK 1 (plausible repair per their published
verdicts) and 150 it did not, with at most --per-repo cases of a project per group. Population = the cases of the paper's PR evaluation
(a case whose published verdicts are invalid, timeout or dependency error, for 75% or more of its candidates is dropped).
Writes <out>/splits/test.json (the TaRBench rows, one per line, so the LLM runners can read it), <out>/subset_meta.json and
<out>/reference_raw.json (the repaired lines as RAW Java taken from the repaired file, to judge predictions without tokenisation).

  python -m data_processing.build_java_contrast_subset --bench ~/references/TaRGet_Results_download/TaRGet_Results/Best_on_TaRBench \
      --out ~/eval_results_java_llm/contrast300 --size 300
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

INVALID = ("timeout", "dependency_error")


def reference_raw(row):
    """The repaired lines as raw Java: the lines of the repaired test at the target line numbers (None when it cannot be found)."""
    changes = row["hunk"].get("targetChanges") or []
    if not changes:
        return None
    start = row["aSource"]["startLine"]
    lines = row["aSource"]["code"].split("\n")
    picked = []
    for change in changes:
        index = change["lineNo"] - start
        if not 0 <= index < len(lines):
            return None
        picked.append(lines[index])
    return "\n".join(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--per-repo", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    bench = Path(args.bench)
    tests = json.load(open(bench / "test.json"))
    verdicts = json.load(open(bench / "test_verdicts.json"))
    by_id = defaultdict(dict)
    for v in verdicts:
        by_id[v["ID"]][v["rank"]] = v
    rows = {}
    info = {}
    dropped = 0
    for row in tests:
        vs = by_id.get(row["ID"])
        if not vs:
            continue
        invalid = sum(1 for v in vs.values() if v["verdict"]["status"] in INVALID)
        if len(vs) > 1 and invalid / len(vs) >= 0.75:
            dropped += 1
            continue
        raw = reference_raw(row)
        if raw is None:
            continue                                  # the repaired lines cannot be located (needed to judge predictions)
        solved1 = bool(vs[0]["success"])
        solved5 = any(vs[r]["success"] for r in range(5) if r in vs)
        rows[row["ID"]] = row
        info[row["ID"]] = {"repo": row["ID"].split(":")[0], "solved_at_1": solved1, "solved_in_top5": solved5, "reference_raw": raw}
    population = {"solved_at_1": sum(i["solved_at_1"] for i in info.values()),
                  "not_solved_at_1": sum(not i["solved_at_1"] for i in info.values())}
    print("cases in the evaluation population:", len(rows), "(dropped as invalid:", dropped, ")", population)
    rnd = random.Random(args.seed)
    chosen = {}
    for group, flag in (("solved_at_1", True), ("not_solved_at_1", False)):
        ids = [i for i, x in info.items() if x["solved_at_1"] == flag]
        rnd.shuffle(ids)
        per_repo, picked = Counter(), []
        for i in ids:
            if per_repo[info[i]["repo"]] < args.per_repo:
                per_repo[info[i]["repo"]] += 1
                picked.append(i)
            if len(picked) == args.size // 2:
                break
        chosen[group] = picked
    order = sorted(chosen["solved_at_1"] + chosen["not_solved_at_1"])
    out = Path(args.out)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    with open(out / "splits" / "test.json", "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json.dumps(rows[i], ensure_ascii=False) for i in order) + "\n]\n")
    meta = {"population": population, "seed": args.seed, "per_repo": args.per_repo,
            "cases": {str(n): {"id": i, "group": "solved_at_1" if i in set(chosen["solved_at_1"]) else "not_solved_at_1",
                               "repo": info[i]["repo"], "solved_in_top5": info[i]["solved_in_top5"]} for n, i in enumerate(order)}}
    json.dump(meta, open(out / "subset_meta.json", "w"), indent=1)
    json.dump({str(n): info[i]["reference_raw"] for n, i in enumerate(order)}, open(out / "reference_raw.json", "w"), indent=1)
    print("subset:", {g: len(v) for g, v in chosen.items()}, "projects:", len({info[i]["repo"] for i in order}),
          "| of the not_solved_at_1 group, solved within the top 5:", sum(info[i]["solved_in_top5"] for i in chosen["not_solved_at_1"]))


if __name__ == "__main__":
    main()
