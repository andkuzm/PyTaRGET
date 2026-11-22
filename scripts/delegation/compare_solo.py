"""Compares S6 runs with the solo runs of the same task: success, output tokens, calls and INPUT tokens of the strong agent (from the sub-agent transcripts).
total input = sum over model calls of the whole context read (input + cache creation + cache read); fresh input = input + cache-creation tokens (not read from cache); work = extra input beyond the first call's context.
usage: compare_solo.py <run id> [...]"""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing_breakdown as rb
FALLBACK = {"gson_a80": (804, 4, 22979), "gson_cbad": (1990, 6, 84716), "ledger_dev": (4194, 5.5, 39764)}   # solo_baseline.md: out, calls, work (pilot transcripts no longer on disk)
R = json.load(open(os.path.expanduser("~/delegation_exp/batch_results.json")))


def stats(run):
    for t in rb.T:
        p = t + rb.M.get(run, "-") + ".output"
        if os.path.exists(p):
            raw = {}
            for l in open(p):
                try: d = json.loads(l)
                except ValueError: continue
                m = d.get("message") or {}
                if d.get("type") == "assistant" and m.get("usage"): raw[m.get("id") or d.get("uuid")] = m["usage"]
            u = list(raw.values()); g = lambda x, k: x.get(k, 0) or 0
            ctx = [g(x, "input_tokens") + g(x, "cache_creation_input_tokens") + g(x, "cache_read_input_tokens") for x in u]
            return {"out": sum(g(x, "output_tokens") for x in u), "calls": len(u), "total_in": sum(ctx), "fresh_in": sum(g(x, "input_tokens") + g(x, "cache_creation_input_tokens") for x in u), "work": sum(c - ctx[0] for c in ctx)}
    return None


def solo(task):
    s = [x for x in (stats(k) for k in R if k.startswith(task + "__solo__")) if x]
    if s: return {k: st.mean(x[k] for x in s) for k in s[0]} | {"n": len(s)}
    return None


print("| Task | success | out tok | calls | total input | fresh input | work |\n|---|---|---|---|---|---|---|")
for run in sys.argv[1:]:
    task = run.split("__")[0]; r = stats(run); b = solo(task)
    if not r: print(f"| {task} | no transcript | | | | | |"); continue
    sc = R.get(run, {}).get("success", "?")
    f = lambda k, fmt="{:.0f}": fmt.format(r[k]) + (" (" + fmt.format(b[k]) + ")" if b else (" (" + str(FALLBACK[task][{"out": 0, "calls": 1, "work": 2}[k]]) + ")" if task in FALLBACK and k in ("out", "calls", "work") else " (n/a)"))
    print(f"| {task} | {sc} | {f('out')} | {f('calls', '{:.1f}')} | {f('total_in')} | {f('fresh_in')} | {f('work')} |")
print("\n(solo mean in brackets; solo n = " + ", ".join(f"{run.split('__')[0]}:{(solo(run.split('__')[0]) or {}).get('n', 'fallback')}" for run in sys.argv[1:]) + ")")
