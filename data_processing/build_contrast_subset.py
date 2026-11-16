"""
A contrast subset of the matched test set: about half of the cases that at least one of the five reference models solved
(a candidate in its top 5 that passes the checked plausibility rule) and half that none of them solved, with at most
--per-repo cases of one repository per group, so that the subset is not dominated by a few projects.

Writes <out>/splits/test.json (same layout as the matched set: one case per line, so iter_rows streams it) and
<out>/subset_meta.json: for every new ID the matched ID, the group ("solved" or "unsolved"), the repository and how many of
the five models solved it. The groups are NOT representative: the pass rates on the subset must be re-weighted with the
group sizes of the whole matched set (stored in the meta file) to estimate rates on all cases.

  python -m data_processing.build_contrast_subset --out ~/eval_results_perproject_llm/contrast300 --size 300
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from data_processing.plausibility_test import iter_rows

R = Path.home()
MODELS = ["codet5p", "plbart", "codegen", "qwen3coder", "gptoss20b"]


def counts(result, i):
    """Does candidate i pass under the checked rule (the test needs the span, or the candidate still asserts)?"""
    if result["verdicts"][i] != "pass":
        return False
    settled = result.get("span_removed") not in ("pass", "unparseable")
    return settled or bool(result.get("assertion_kept", [False] * 5)[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=300, help="total cases, half per group")
    ap.add_argument("--per-repo", type=int, default=4, help="at most this many cases of a repository per group")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--matched", default=str(R / "eval_results_perproject_llm" / "matched"))
    args = ap.parse_args()

    keys = json.load(open(Path(args.matched) / "matched_keys.json"))
    matched = [tuple(k) for k in keys["matched"]]
    res = {}
    for m in MODELS[:3]:
        pos = {}
        for i, k in enumerate(keys[m]):
            pos.setdefault(tuple(k), i)
        d = json.load(open(R / f"eval_results_perproject/{m}/0.8/plausibility_v3.json"))["results"]
        res[m] = [d.get(str(pos[k])) for k in matched]
    for m in MODELS[3:]:
        d = json.load(open(R / f"eval_llm_plaus/matched_{m}/0.8/plausibility_v3.json"))["results"]
        res[m] = [d.get(str(i)) for i in range(len(matched))]

    info = {}
    for i in range(len(matched)):
        rs = [res[m][i] for m in MODELS]
        if not all(r is not None and r["status"] == "evaluated" for r in rs):
            continue                                     # only cases every model could be scored on
        solved_by = [m for m, r in zip(MODELS, rs) if any(counts(r, j) for j in range(len(r["verdicts"][:5])))]
        info[i] = solved_by
    repo_of = {}
    for i, row in enumerate(iter_rows(Path(args.matched) / "splits" / "test.json")):
        if i in info:
            repo_of[i] = row["repository_name"]

    rnd = random.Random(args.seed)
    chosen, population = {}, {}
    for group, test in (("solved", lambda s: len(s) > 0), ("unsolved", lambda s: len(s) == 0)):
        ids = [i for i, s in info.items() if test(s)]
        population[group] = len(ids)
        rnd.shuffle(ids)
        per_repo, picked = defaultdict(int), []
        for i in ids:                                    # shuffled, so the cap keeps a random case of each repository
            if per_repo[repo_of[i]] < args.per_repo:
                per_repo[repo_of[i]] += 1
                picked.append(i)
            if len(picked) == args.size // 2:
                break
        chosen[group] = picked
    order = sorted(chosen["solved"] + chosen["unsolved"])
    new_id = {i: n for n, i in enumerate(order)}

    out = Path(args.out)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    keep = set(order)
    with open(out / "splits" / "test.json", "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for i, row in enumerate(iter_rows(Path(args.matched) / "splits" / "test.json")):
            if i in keep:
                f.write(("" if first else ",\n") + json.dumps(row, ensure_ascii=False))
                first = False
        f.write("\n]\n")
    meta = {"population": population, "evaluated_for_all_models": len(info), "seed": args.seed, "per_repo": args.per_repo,
            "cases": {str(new_id[i]): {"matched_id": i, "group": "solved" if i in set(chosen["solved"]) else "unsolved",
                                        "repo": repo_of[i], "solved_by": info[i]} for i in order}}
    json.dump(meta, open(out / "subset_meta.json", "w"), indent=1)
    print("population (cases every model was scored on):", population, "of", len(info))
    print("subset:", {g: len(v) for g, v in chosen.items()}, "repositories:", len({repo_of[i] for i in order}))


if __name__ == "__main__":
    main()
