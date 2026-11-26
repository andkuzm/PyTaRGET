"""One markdown row per pipeline run for docs/results/inkling_pipeline_overnight.md.  usage: overnight_row.py <run id> [...]"""
import json
import os
import statistics as st
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_solo_lib as c  # noqa: E402

H = os.path.expanduser("~")
for run in sys.argv[1:]:
    task = run.split("__")[0]; r = c.stats(run); b = c.solo(task); R = c.R.get(run, {})
    fb = c.FALLBACK.get(task)
    def f(k, i=None):
        base = f"{b[k]:.0f}" if b and k in b else (f"{fb[i]}" if fb and i is not None else "n/a")
        return f"{r[k] if r else R.get(k, '?')} ({base})"
    calls_f = os.path.join(H, "delegation_exp/runs", run, "ws/.delegation/delegate_calls.jsonl")
    d = [json.loads(l) for l in open(calls_f)] if os.path.exists(calls_f) else []
    if d and d[0].get("unit") == "agent":
        deleg = f"{len(d)} agent run(s): {sum(x.get('api_calls') or 0 for x in d)} calls, {sum(x['seconds'] for x in d):.0f} s, {sum(x['tokens_in'] for x in d)//1000}k in / {sum(x['tokens_out'] for x in d)/1000:.1f}k out" + (f"; discarded changes: {sum(len(x.get('ignored', [])) for x in d)}" if any(x.get('ignored') for x in d) else "") + (f"; audit flags {sum(x.get('audit', 0) for x in d)}" if any(x.get('audit') for x in d) else "")
    else:
        deleg = f"{len(d)} units, {sum(x['applied'] for x in d)} applied, {sum(x['tokens_out'] for x in d)/1000:.1f}k out" if d else "not called (no-model fixes only)"
    rep = os.path.join(H, "delegation_exp/runs", run, "ws/.delegation/repair_stdout.txt")
    end = ""
    if os.path.exists(rep):
        t = open(rep).read(); end = ("ACCEPT" if "\nACCEPT" in t else "ESCALATE" if "ESCALATE" in t else "?") + (", round rolled back" if "ROLLED BACK" in t else "")
    strict = "yes" if R.get("success") else "NO " + ",".join(R.get("weakened") or [])
    print(f"| {task} {run.rsplit('__', 1)[1]} | {strict} ({R.get('pass')}/{R.get('total')}) | {f('out', 0)} | {f('calls', 1)} | {f('total_in')} | {f('fresh_in')} | {f('work', 2)} | {end} | {deleg} |")
