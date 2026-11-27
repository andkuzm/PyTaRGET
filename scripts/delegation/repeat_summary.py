"""Per repo: mean (min-max) of the strong agent's numbers for SOLO runs and for PIPELINE runs with the Inkling agent delegate, from the sub-agent transcripts (so runs whose transcripts are gone only count for
output / calls / work, taken from batch_results.json). Pipeline = tonight's runs with delegate inkling-agent (rep >= 20); invalid or superseded runs are listed in EXCLUDE.  usage: repeat_summary.py"""
import json
import os
import statistics as st
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_solo_lib as c  # noqa: E402

# superseded by a fix (kept in the run table, not in the means)
EXCLUDE = {"shop__deleg-S6-auto-repair__r24": "before fix R1", "real_logicng_b__deleg-S6-auto-repair__r28": "before fix R2", "real_logicng_b__deleg-S6-auto-repair__r33": "before fix R3"}
S = json.load(open(os.path.expanduser("~/delegation_exp/state.json")))["runs"]
KEYS = ("out", "calls", "total_in", "fresh_in", "work")


def numbers(run):
    r = c.stats(run)
    if r: return r
    v = c.R.get(run)
    return {k: v[k] for k in ("out", "calls", "work") if k in v} if v else None


def pick(task, arm):
    out = []
    for run in c.R:
        if not run.startswith(task + "__") or run in EXCLUDE or c.R[run].get("success") is None: continue
        if arm == "solo" and "__solo__" in run: out.append(run)
        if arm == "pipe" and "__deleg-S6" in run and S.get(run, {}).get("delegate_model") == "inkling-agent": out.append(run)
        if arm == "single" and "__deleg-S6" in run and S.get(run, {}).get("delegate_model") == "inkling" and int(run.rsplit("__r", 1)[1]) >= 70: out.append(run)
    return out


def cell(runs, k):
    v = [numbers(r)[k] for r in runs if numbers(r) and k in numbers(r)]
    if not v: return "n/a"
    f = lambda x: f"{x/1000:.1f}k" if k not in ("calls",) and x >= 10000 else f"{x:.0f}" if k != "calls" else f"{x:.1f}"
    return f(st.mean(v)) + (f" ({f(min(v))}-{f(max(v))}, n={len(v)})" if len(v) > 1 else " (n=1)")


tasks = sorted({r.split("__")[0] for r in c.R if "__deleg-S6" in r and S.get(r, {}).get("delegate_model") == "inkling-agent"})
print("| repo | arm | strict pass | output tok | calls | total input | fresh input | work |\n|---|---|---|---|---|---|---|---|")
for t in tasks:
    for arm, name in (("solo", "solo"), ("pipe", "pipeline (Inkling agent)"), ("single", "pipeline (Inkling single prompt)")):
        runs = pick(t, arm)
        if arm == "single" and not runs: continue
        extra = c.FALLBACK.get(t) if arm == "solo" and not runs else None
        if extra:
            print(f"| {t} | solo (earlier pilots) | n/a | {extra[0]} | {extra[1]} | n/a | n/a | {extra[2]} |"); continue
        ok = sum(bool(c.R[r].get("success")) for r in runs)
        print(f"| {t} | {name} | {ok}/{len(runs)} | " + " | ".join(cell(runs, k) for k in KEYS) + " |")
