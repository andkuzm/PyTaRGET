"""Functions of compare_solo.py, importable (generated copy of its definitions)."""
"""Compares S6 runs with the solo runs of the same task: success, output tokens, calls and INPUT tokens of the strong agent (from the sub-agent transcripts).
total input = sum over model calls of the whole context read (input + cache creation + cache read); fresh input = input + cache-creation tokens (not read from cache); work = extra input beyond the first call's context.
usage: compare_solo.py <run id> [...]"""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing_breakdown as rb
FALLBACK = {"shop": (3224, 4, 36803), "gson_a80": (804, 4, 22979), "gson_cbad": (1990, 6, 84716), "ledger_dev": (4194, 5.5, 39764)}   # solo_baseline.md: out, calls, work (pilot transcripts no longer on disk)
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
    # transcripts of older runs are deleted by the system after a while: fall back to the recorded numbers (no input-token columns then)
    s = [v for k, v in R.items() if k.startswith(task + "__solo__") and v.get("out")]
    if s: return {"out": st.mean(v["out"] for v in s), "calls": st.mean(v["calls"] for v in s), "work": st.mean(v["work"] for v in s), "n": len(s)}
    return None


